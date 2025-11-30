from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
from pathlib import Path
import polars as pl
import json
import os
import csv
from .bitmex_system import BitMEXPortfolioSystem
from ..accounts import BitMEXAccount, create_bitmex_account
from ..agents.bitmex_agent import LLMBitMEXAgent
from ..fetchers.binance_fetcher import BinanceFetcher
from ..utils.datetime_utils import parse_utc_datetime

logger = logging.getLogger(__name__)


class BinancePortfolioSystem(BitMEXPortfolioSystem):
    """
    Binance Portfolio System for managing multiple LLM agents trading perpetual contracts.

    Inherits all BitMEX portfolio behaviors; only market data fetch differs.
    Market data is sourced from S3 parquet files (prices, orderbook, funding).
    """

    def __init__(self, universe_size: int = 15, name: str = "binance") -> None:
        super().__init__(universe_size=universe_size, name=name)
        # Keep the same types for agents/accounts to minimize changes elsewhere
        self.agents: Dict[str, LLMBitMEXAgent] = {}
        self.accounts: Dict[str, BitMEXAccount] = {}
        self.fetcher = BinanceFetcher()
        self.market_data: pl.DataFrame = pl.DataFrame()
        self.lookback_days = 10

    def add_agent(
        self, name: str, initial_cash: float = 10000.0, model_name: str = "gpt-4o-mini"
    ) -> None:
        if name in self.agents:
            return
        agent = LLMBitMEXAgent(name, model_name)
        account = create_bitmex_account(initial_cash)
        self.agents[name] = agent
        self.accounts[name] = account

    def _fetch_market_data(
        self, for_date: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch market data for Binance perpetuals via BinanceFetcher.
        """
        logger.info("Fetching Binance market data via BinanceFetcher...")

        dt = self.fetcher._resolve_target_datetime(for_date) if for_date else None
        if dt is None:
            dt = datetime.now(timezone.utc)
        agg_trades_df = self.fetcher.get_agg_trades(symbol_list=self.universe, dt=dt, count= 4 * 24 * (self.lookback_days + 1))
        funding_rate_df = self.fetcher.get_funding_rate_df(symbol_list=self.universe, dt=dt, count= 4 * 24 * (self.lookback_days + 1))
        ob_df = self.fetcher.get_orderbook_df(symbol_list=self.universe, dt=dt, count= 4 * 24 * (self.lookback_days + 1))

        # Inner join on ['symbol', 'close_time'] across agg_trades, funding_rate, orderbook
        try:
            agg_trades_df = agg_trades_df.drop(["start_time", "end_time", "count"])
            funding_rate_df = funding_rate_df.drop(["start_time", "end_time", "count"])
            ob_df = ob_df.drop(["start_time", "end_time", "count"])

            joined = agg_trades_df.join(funding_rate_df, on=["symbol", "close_time"], how="inner")
            joined = joined.join(ob_df, on=["symbol", "close_time"], how="inner")
            self.market_data = joined
        except Exception as e:
            logger.error(f"Failed to join market data frames: {e}")
            # Fallback to empty DataFrame to avoid stale state
            self.market_data = pl.DataFrame()

        for symbol in self.universe:
            try:
                current_price = joined.filter(joined["symbol"] == symbol).filter(pl.col("close_time") == pl.max("close_time")).select("bam_close_15m").to_series().item()
                # Update position prices in all accounts
                for account in self.accounts.values():
                    account.update_position_price(symbol, float(current_price))
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
       
        logger.info(f"Market data fetched for {len(self.universe)} Binance contracts")

    # S3 helper methods removed; BinanceFetcher provides data access.

    @classmethod
    def get_instance(cls):
        """Get singleton instance (for compatibility with mock systems)."""
        if not hasattr(cls, "_instance"):
            cls._instance = create_binance_portfolio_system()
        return cls._instance

    # -------------------------
    # Price history aggregation
    # -------------------------
    def _append_market_data(self, symbol: str, history: List[Dict[str, Any]], funding_info: Dict[str, Any], ob_info: Dict[str, Any]) -> None:
        """
        Append price history records to self.market_data.
        - history items: {'timestamp': ISO8601, 'price': float}
        - output columns: symbol, close_time (ms), datetime (UTC), price
        """
        if not history:
            return
        rows: List[Dict[str, Any]] = []
        for item in history:
            ts_str = item.get("timestamp")
            price = item.get("price")
            if ts_str is None or price is None:
                continue
            try:
                dt_obj = datetime.fromisoformat(ts_str)
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                else:
                    dt_obj = dt_obj.astimezone(timezone.utc)
                close_ms = int(dt_obj.timestamp() * 1000)
                rows.append(
                    {
                        "symbol": symbol,
                        "close_time": close_ms,
                        "datetime": dt_obj,
                        "price": float(price),
                    }
                )
            except Exception:
                continue
        if not rows:
            return
        df_new = pl.DataFrame(rows)
        if self.market_data.is_empty():
            self.market_data = df_new
        else:
            self.market_data = pl.concat([self.market_data, df_new], how="vertical", rechunk=True)

    def save_accounts_parquet(
        self,
        output_dir: str | Path,
        for_date: str | None = None,
    ) -> Path:
        """
        Save current account snapshots to a Parquet file under output_dir.
        If output_dir is not present, it will be created.
        The file name includes date/time for uniqueness.
        """
        # Build timestamp / close_time epoch ms
        ts_dt = parse_utc_datetime(for_date) if for_date else datetime.now(timezone.utc)
        close_time_ms = int(ts_dt.timestamp() * 1000)

        # Build rows for each agent's account positions; include CASH as USDT at price 1.0
        rows: list[dict] = []
        for agent_name, account in self.accounts.items():
            # Positions
            for symbol, pos in account.positions.items():
                rows.append(
                    {
                        "timestamp": ts_dt,
                        "close_time": close_time_ms,
                        "agent": agent_name,
                        "symbol": symbol,
                        "quantity": float(getattr(pos, "quantity", 0.0)),
                        "average_price": float(getattr(pos, "average_price", 0.0)),
                        "current_price": float(getattr(pos, "current_price", 0.0)),
                        "position_value": float(getattr(pos, "market_value", 0.0)),
                        "allocation": float(account.target_allocations.get(symbol, 0.0)),
                        #"cash_balance": float(account.cash_balance),
                        #"total_value": float(account.get_total_value()),
                    }
                )
            # Cash as USDT
            if account.cash_balance and account.cash_balance != 0.0:
                rows.append(
                    {
                        "timestamp": ts_dt,
                        "close_time": close_time_ms,
                        "agent": agent_name,
                        "symbol": "USDT",
                        "quantity": float(account.cash_balance),
                        "average_price": 1.0,
                        "current_price": 1.0,
                        "position_value": float(account.cash_balance),
                        "allocation": float(account.target_allocations.get("CASH", 0.0)),
                        #"cash_balance": float(account.cash_balance),
                        #"total_value": float(account.get_total_value()),
                    }
                )

        df = pl.DataFrame(rows, strict=False) if rows else pl.DataFrame()

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        now_utc = datetime.now(timezone.utc)
        # Determine filename timestamp string as YYYY-MM-DDTHHMMSS
        if for_date:
            try:
                parsed = parse_utc_datetime(for_date)
                dstr = parsed.strftime("%Y-%m-%dT%H%M%S")
            except Exception:
                dstr = now_utc.strftime("%Y-%m-%dT%H%M%S")
        else:
            dstr = now_utc.strftime("%Y-%m-%dT%H%M%S")
        # Create date-based subdirectory (YYYY-MM-DD)
        date_dir = dstr.split("T")[0] if "T" in dstr else dstr[:10]
        dated_dir_path = output_dir_path / date_dir
        dated_dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dated_dir_path / f"{self.name}_{dstr}.parquet"

        df.write_parquet(str(file_path))
        logger.info(f"Saved allocations to {file_path}")

        # Save LLM input prompts separately as JSONL for efficiency
        llm_jsonl = dated_dir_path / f"{self.name}_{dstr}_llm_input.jsonl"
        try:
            with open(llm_jsonl, "a", encoding="utf-8") as f:
                for agent_name, agent in self.agents.items():
                    llm_input = getattr(agent, "last_llm_input", None)
                    if not llm_input:
                        continue
                    rec = {
                        "timestamp": ts_dt.isoformat(),
                        "agent": agent_name,
                        "model": llm_input.get("model"),
                        "prompt": llm_input.get("prompt"),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(f"Saved LLM inputs to {llm_jsonl}")
        except Exception as e:
            logger.error(f"Failed to write LLM inputs: {e}")

        # Append per-agent summary (cash_balance, total_value, performance) as CSV in output_dir_path
        try:
            csv_path = output_dir_path / f"{self.name}_account_summary.csv"
            need_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
                fieldnames = ["timestamp", "close_time", "agent", "cash_balance", "total_value", "performance"]
                writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
                if need_header:
                    writer.writeheader()
                for agent_name, account in self.accounts.items():
                    total_value = float(account.get_total_value())
                    init_cash = float(getattr(account, "initial_cash", 0.0) or 0.0)
                    performance = (total_value - init_cash) / init_cash * 100.0 if init_cash > 0 else 0.0
                    writer.writerow(
                        {
                            "timestamp": ts_dt.isoformat(),
                            "close_time": close_time_ms,
                            "agent": agent_name,
                            "cash_balance": float(account.cash_balance),
                            "total_value": total_value,
                            "performance": performance,
                        }
                    )
            logger.info(f"Appended account summary to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to append account summary CSV: {e}")
        return file_path


    def attach_price_to_allocations(self, allocations_df: pl.DataFrame) -> pl.DataFrame:
        """
        Join allocations_df with self.price_history on (symbol, close_time) to add price.
        - allocations_df: expected to have columns ['symbol', 'close_time', ...]
        - self.price_history: columns ['symbol', 'close_time', 'datetime', 'price']
        Returns a new DataFrame with 'price' appended (left join).
        """
        if allocations_df is None or allocations_df.is_empty():
            return allocations_df if allocations_df is not None else pl.DataFrame()
        if self.market_data.is_empty():
            assert False, "Market data is empty"

        # Deduplicate price_history on (symbol, close_time), keep last (latest ingested)
        ph = (
            self.market_data
            .sort(["symbol", "close_time"])
            .unique(subset=["symbol", "close_time"], keep="last")
            .select(["symbol", "close_time", "bam_close_15m"])
        )

        # Left join to add price
        joined = allocations_df.join(ph, on=["symbol", "close_time"], how="left")
        # For USDT (cash-like), force price to 1.0
        joined = joined.with_columns(
            pl.when(pl.col("symbol") == "USDT")
            .then(pl.lit(1.0))
            .otherwise(pl.col("bam_close_15m"))
            .alias("bam_close_15m")
        )
        return joined


def create_binance_portfolio_system() -> BinancePortfolioSystem:
    """
    Create a new Binance portfolio system instance.
    """
    return BinancePortfolioSystem()


