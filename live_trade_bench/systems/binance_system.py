from __future__ import annotations

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
from loguru import logger


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
        self.output_dir = Path(os.getenv("BASE_DIR")) / name

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

        agg_trades_df = self.fetcher.get_agg_trades(symbol_list=self.universe, dt=dt, lookback_days=self.lookback_days)

        funding_rate_df = self.fetcher.get_funding_rate_df(symbol_list=self.universe, dt=dt, lookback_days=self.lookback_days)
        ob_df = self.fetcher.get_orderbook_df(symbol_list=self.universe, dt=dt, lookback_days=self.lookback_days)

        # Inner join on ['symbol', 'close_time'] across agg_trades, funding_rate, orderbook
        # Constraint: (symbol, close_time) pairs must be unique in each DataFrame
        try:
            # Drop columns only if they exist
            cols_to_drop = ["start_time", "end_time", "count"]
            agg_trades_df = agg_trades_df.drop([col for col in cols_to_drop if col in agg_trades_df.columns])
            funding_rate_df = funding_rate_df.drop([col for col in cols_to_drop if col in funding_rate_df.columns])
            ob_df = ob_df.drop([col for col in cols_to_drop if col in ob_df.columns])

            # Validate uniqueness constraint: (symbol, close_time) must be unique
            def check_uniqueness(df: pl.DataFrame, name: str) -> None:
                duplicates = df.group_by(["symbol", "close_time"]).agg(pl.count().alias("count")).filter(pl.col("count") > 1)
                if not duplicates.is_empty():
                    error_msg = f"Violation: (symbol, close_time) must be unique in {name}. Found duplicates:\n{duplicates}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            check_uniqueness(agg_trades_df, "agg_trades_df")
            check_uniqueness(funding_rate_df, "funding_rate_df")
            check_uniqueness(ob_df, "ob_df")

            # Sanity check: Remove duplicate columns before join
            def remove_duplicate_columns(df: pl.DataFrame, existing_df: pl.DataFrame, name: str) -> pl.DataFrame:
                """Remove columns from df that already exist in existing_df (except join keys)"""
                existing_cols = set(existing_df.columns)
                join_keys = ["symbol", "close_time"]
                cols_to_remove = []
                
                for col in df.columns:
                    # Skip join keys
                    if col in join_keys:
                        continue
                    # Check if column exists in existing_df
                    if col in existing_cols:
                        cols_to_remove.append(col)
                        logger.warning(f"Removing duplicate column '{col}' from {name} before join")
                
                if cols_to_remove:
                    df = df.drop(cols_to_remove)
                
                return df

            joined = agg_trades_df.join(funding_rate_df, on=["symbol", "close_time"], how="inner")
            
            # Before second join, check for potential column conflicts
            # 1. Remove columns from ob_df that already exist in joined (except join keys)
            ob_df = remove_duplicate_columns(ob_df, joined, "ob_df")
            
            # 2. Check for columns with _right suffix that might conflict
            # If joined already has columns ending with _right, ob_df columns with same base name
            # would try to get _right suffix again, causing DuplicateError
            joined_right_cols = [col for col in joined.columns if col.endswith("_right")]
            if joined_right_cols:
                logger.debug(f"Found columns with '_right' suffix in joined: {joined_right_cols}")
                # Remove base columns from ob_df that would conflict
                ob_df_cols_to_remove = []
                for right_col in joined_right_cols:
                    base_col = right_col.replace("_right", "")
                    if base_col in ob_df.columns:
                        ob_df_cols_to_remove.append(base_col)
                        logger.warning(f"Removing '{base_col}' from ob_df to avoid conflict with existing '{right_col}' in joined")
                if ob_df_cols_to_remove:
                    ob_df = ob_df.drop(ob_df_cols_to_remove)
            
            # 3. Also check if ob_df has columns that would create _right suffix conflicts
            # If ob_df has a column that would become 'X_right' but 'X_right' already exists in joined
            joined_cols_set = set(joined.columns)
            ob_df_cols_to_remove_conflict = []
            for ob_col in ob_df.columns:
                if ob_col in ["symbol", "close_time"]:
                    continue
                potential_right_col = f"{ob_col}_right"
                if potential_right_col in joined_cols_set:
                    ob_df_cols_to_remove_conflict.append(ob_col)
                    logger.warning(f"Removing '{ob_col}' from ob_df to avoid creating duplicate '{potential_right_col}'")
            if ob_df_cols_to_remove_conflict:
                ob_df = ob_df.drop(ob_df_cols_to_remove_conflict)
            
            joined = joined.join(ob_df, on=["symbol", "close_time"], how="inner")
            
            # Validate uniqueness after join (join should preserve uniqueness)
            check_uniqueness(joined, "joined")
            
            self.market_data = joined
        except Exception as e:
            logger.error(f"Failed to join market data frames: {e}")
            # Fallback to empty DataFrame to avoid stale state
            self.market_data = pl.DataFrame()
            raise

        # Update position prices: since (symbol, close_time) is unique, we can safely get latest price
        for symbol in self.universe:
            try:
                symbol_df = joined.filter(joined["symbol"] == symbol)
                if symbol_df.is_empty():
                    continue
                # Get the row with maximum close_time (unique constraint ensures single row)
                max_close_time = symbol_df["close_time"].max()
                latest_row = symbol_df.filter(pl.col("close_time") == max_close_time)
                if latest_row.is_empty():
                    continue
                current_price = latest_row.select("bam_close_15m").to_series().item()
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
        for_date: str | None = None,
    ) -> Path:
        """
        Save current account snapshots to a Parquet file under output_dir.
        If output_dir is not present, it will be created.
        The file name includes date/time for uniqueness.
        """
        # Build timestamp / close_time epoch ms
        # 실제 allocation 할당 시점의 시간 사용 (UTC 기준)
        ts_dt = parse_utc_datetime(for_date) if for_date else datetime.now(timezone.utc)
        close_time_ms = int(ts_dt.timestamp() * 1000)
        
        # last_trade_time은 실제 trade 실행 시점의 시간 (UTC)
        last_trade_time = ts_dt

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
                        "last_trade_time": last_trade_time,  # 실제 trade 실행 시점의 시간
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
                        "last_trade_time": last_trade_time,  # 실제 trade 실행 시점의 시간
                        #"cash_balance": float(account.cash_balance),
                        #"total_value": float(account.get_total_value()),
                    }
                )

        df = pl.DataFrame(rows, strict=False) if rows else pl.DataFrame()

        output_dir_path = self.output_dir
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
        file_path = dated_dir_path / f"{dstr}.parquet"

        df.write_parquet(str(file_path))
        logger.info(f"Saved allocations to {file_path}")

        # Save additional account metadata as JSON for state restoration
        metadata_path = dated_dir_path / f"{dstr}_metadata.json"
        try:
            metadata = {
                "timestamp": ts_dt.isoformat(),
                "close_time": close_time_ms,
                "agents": {}
            }
            for agent_name, account in self.accounts.items():
                metadata["agents"][agent_name] = {
                    "cash_balance": float(account.cash_balance),
                    "total_value": float(account.get_total_value()),
                    "initial_cash": float(getattr(account, "initial_cash", 0.0)),
                    "target_allocations": {k: float(v) for k, v in account.target_allocations.items()},
                    "total_fees": float(getattr(account, "total_fees", 0.0)),
                    "total_funding_fees": float(getattr(account, "total_funding_fees", 0.0)),
                    "last_rebalance": account.last_rebalance,
                }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved account metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to write account metadata: {e}")

        # Save allocation_history for each agent (append mode to preserve full history)
        # Store in a single file per agent to accumulate history across all trades
        for agent_name, account in self.accounts.items():
            allocation_history_path = self.output_dir / f"{agent_name}_allocation_history.json"
            try:
                # Load existing history if file exists
                existing_history = []
                if allocation_history_path.exists():
                    try:
                        with open(allocation_history_path, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)
                            existing_history = existing_data.get("history", [])
                    except Exception:
                        existing_history = []
                
                # Merge with current history (avoid duplicates based on timestamp)
                existing_timestamps = {entry.get("timestamp") for entry in existing_history}
                new_entries = [
                    entry for entry in account.allocation_history
                    if entry.get("timestamp") not in existing_timestamps
                ]
                
                merged_history = existing_history + new_entries
                
                # Save merged history
                allocation_history_data = {
                    "last_updated": ts_dt.isoformat(),
                    "last_close_time": close_time_ms,
                    "history": merged_history
                }
                with open(allocation_history_path, "w", encoding="utf-8") as f:
                    json.dump(allocation_history_data, f, indent=2, ensure_ascii=False, default=str)
                logger.debug(f"Saved allocation_history for {agent_name} ({len(merged_history)} total entries)")
            except Exception as e:
                logger.error(f"Failed to write allocation_history for {agent_name}: {e}")

        # Save transactions for each agent (append mode to preserve full history)
        # Store in a single file per agent to accumulate transactions across all trades
        for agent_name, account in self.accounts.items():
            transactions_path = self.output_dir / f"{agent_name}_transactions.json"
            try:
                # Load existing transactions if file exists
                existing_transactions = []
                if transactions_path.exists():
                    try:
                        with open(transactions_path, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)
                            existing_transactions = existing_data.get("transactions", [])
                    except Exception:
                        existing_transactions = []
                
                # Convert current transactions to dicts
                current_transactions_dicts = []
                for tx in account.transactions:
                    tx_dict = {
                        "transaction_id": str(tx.transaction_id),
                        "ticker": tx.ticker,
                        "quantity": float(tx.quantity),
                        "price": float(tx.price),
                        "transaction_type": tx.transaction_type,
                        "timestamp": tx.timestamp.isoformat() if isinstance(tx.timestamp, datetime) else str(tx.timestamp),
                    }
                    current_transactions_dicts.append(tx_dict)
                
                # Merge with existing transactions (avoid duplicates based on transaction_id)
                existing_ids = {tx.get("transaction_id") for tx in existing_transactions}
                new_transactions = [
                    tx for tx in current_transactions_dicts
                    if tx.get("transaction_id") not in existing_ids
                ]
                
                merged_transactions = existing_transactions + new_transactions
                
                # Save merged transactions
                transactions_data = {
                    "last_updated": ts_dt.isoformat(),
                    "last_close_time": close_time_ms,
                    "transactions": merged_transactions
                }
                with open(transactions_path, "w", encoding="utf-8") as f:
                    json.dump(transactions_data, f, indent=2, ensure_ascii=False, default=str)
                logger.debug(f"Saved transactions for {agent_name} ({len(merged_transactions)} total entries)")
            except Exception as e:
                logger.error(f"Failed to write transactions for {agent_name}: {e}")

        # Save LLM input prompts separately as JSONL for efficiency
        llm_jsonl = dated_dir_path / f"{dstr}_llm_input.jsonl"
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
            csv_path = output_dir_path / "account_summary.csv"
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

    def get_latest_trade_date(self) -> Optional[datetime]:
        """
        Find the latest trade date from saved parquet files.
        
        Returns:
            Latest trade datetime (UTC) or None if no files found
        """
        if not self.output_dir.exists():
            return None
        
        latest_timestamp = None
        latest_dt = None
        
        # Search for parquet files in date subdirectories
        for date_dir in self.output_dir.iterdir():
            if not date_dir.is_dir():
                continue
            
            for parquet_file in date_dir.glob("*.parquet"):
                # Skip LLM input files if any
                if "_llm_input" in parquet_file.stem:
                    continue
                
                file_stem = parquet_file.stem
                # File format: YYYY-MM-DDTHHMMSS.parquet
                if len(file_stem) == 17 and "T" in file_stem:
                    try:
                        file_dt = datetime.strptime(file_stem, "%Y-%m-%dT%H%M%S")
                        file_dt = file_dt.replace(tzinfo=timezone.utc)
                        if latest_dt is None or file_dt > latest_dt:
                            latest_dt = file_dt
                            latest_timestamp = file_stem
                    except ValueError:
                        continue
        
        if latest_dt:
            logger.info(f"Found latest trade date: {latest_timestamp} ({latest_dt})")
        else:
            logger.info("No trade files found")
        
        return latest_dt

    def load_accounts_from_parquet(self) -> bool:
        """
        Load account state from the latest saved parquet file.
        Restores positions, cash balance, and target allocations.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        if not self.output_dir.exists():
            logger.warning(f"Output directory does not exist: {self.output_dir}")
            return False
        
        # Find the latest parquet file
        latest_dt = self.get_latest_trade_date()
        if latest_dt is None:
            logger.info("No saved trade files found, starting fresh")
            return False
        
        # Find the file path
        date_dir = latest_dt.strftime("%Y-%m-%d")
        dated_dir_path = self.output_dir / date_dir
        file_stem = latest_dt.strftime("%Y-%m-%dT%H%M%S")
        file_path = dated_dir_path / f"{file_stem}.parquet"
        
        if not file_path.exists():
            logger.warning(f"Latest trade file not found: {file_path}")
            return False
        
        try:
            # Load parquet file
            df = pl.read_parquet(str(file_path))
            if df.is_empty():
                logger.warning(f"Parquet file is empty: {file_path}")
                return False
            
            logger.info(f"Loading account state from {file_path} ({len(df)} rows)")
            
            # Group by agent and restore state
            for agent_name in df["agent"].unique().to_list():
                if agent_name not in self.accounts:
                    logger.warning(f"Agent {agent_name} not found in system, skipping")
                    continue
                
                account = self.accounts[agent_name]
                agent_df = df.filter(pl.col("agent") == agent_name)
                
                # Clear existing positions
                account.positions.clear()
                
                # Restore positions and cash
                cash_balance = 0.0
                target_allocations: Dict[str, float] = {}
                
                for row in agent_df.iter_rows(named=True):
                    symbol = row["symbol"]
                    quantity = float(row["quantity"])
                    average_price = float(row["average_price"])
                    current_price = float(row["current_price"])
                    allocation = float(row.get("allocation", 0.0))
                    
                    if symbol == "USDT":
                        # USDT represents cash balance
                        cash_balance = quantity
                        target_allocations["CASH"] = allocation
                    else:
                        # Regular position
                        if quantity > 0.01:  # Only restore meaningful positions
                            from ..accounts import Position
                            account.positions[symbol] = Position(
                                symbol=symbol,
                                quantity=quantity,
                                average_price=average_price,
                                current_price=current_price,
                                url=None,  # URL is not saved in parquet
                            )
                        target_allocations[symbol] = allocation
                
                # Restore account state
                account.cash_balance = cash_balance
                account.target_allocations = target_allocations
                
                logger.info(
                    f"Restored {agent_name}: {len(account.positions)} positions, "
                    f"cash={cash_balance:.2f}, total_value={account.get_total_value():.2f}"
                )
            
            # Try to load additional metadata if available
            metadata_path = dated_dir_path / f"{file_stem}_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    
                    for agent_name, agent_meta in metadata.get("agents", {}).items():
                        if agent_name not in self.accounts:
                            continue
                        
                        account = self.accounts[agent_name]
                        # Restore additional fields from metadata
                        account.total_fees = float(agent_meta.get("total_fees", 0.0))
                        account.total_funding_fees = float(agent_meta.get("total_funding_fees", 0.0))
                        account.last_rebalance = agent_meta.get("last_rebalance")
                        
                        # Ensure target_allocations is complete (may have been partially restored from parquet)
                        meta_allocations = agent_meta.get("target_allocations", {})
                        if meta_allocations:
                            # Merge with parquet-loaded allocations (metadata takes precedence)
                            account.target_allocations.update({k: float(v) for k, v in meta_allocations.items()})
                    
                    logger.info(f"Loaded additional metadata from {metadata_path}")
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {metadata_path}: {e}")

            # Try to load allocation_history from agent-specific files (accumulated across all trades)
            for agent_name, account in self.accounts.items():
                allocation_history_path = self.output_dir / f"{agent_name}_allocation_history.json"
                if allocation_history_path.exists():
                    try:
                        with open(allocation_history_path, "r", encoding="utf-8") as f:
                            allocation_history_data = json.load(f)
                        
                        # Restore allocation_history (already a list of dicts)
                        history_list = allocation_history_data.get("history", [])
                        account.allocation_history = history_list
                        logger.debug(f"Restored {len(history_list)} allocation history entries for {agent_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load allocation_history for {agent_name}: {e}")
                else:
                    logger.debug(f"No allocation_history file found for {agent_name}")

            # Try to load transactions from agent-specific files (accumulated across all trades)
            for agent_name, account in self.accounts.items():
                transactions_path = self.output_dir / f"{agent_name}_transactions.json"
                if transactions_path.exists():
                    try:
                        with open(transactions_path, "r", encoding="utf-8") as f:
                            transactions_data = json.load(f)
                        
                        # Restore transactions by converting dicts back to Transaction objects
                        from ..accounts import Transaction
                        import uuid
                        
                        transactions_list = transactions_data.get("transactions", [])
                        restored_transactions = []
                        for tx_dict in transactions_list:
                            try:
                                tx = Transaction(
                                    transaction_id=uuid.UUID(tx_dict["transaction_id"]),
                                    ticker=tx_dict["ticker"],
                                    quantity=float(tx_dict["quantity"]),
                                    price=float(tx_dict["price"]),
                                    transaction_type=tx_dict["transaction_type"],
                                    timestamp=parse_utc_datetime(tx_dict["timestamp"]) if isinstance(tx_dict["timestamp"], str) else tx_dict["timestamp"],
                                )
                                restored_transactions.append(tx)
                            except Exception as e:
                                logger.warning(f"Failed to restore transaction for {agent_name}: {e}")
                                continue
                        
                        account.transactions = restored_transactions
                        logger.debug(f"Restored {len(restored_transactions)} transactions for {agent_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load transactions for {agent_name}: {e}")
                else:
                    logger.debug(f"No transactions file found for {agent_name}")
            
            logger.info(f"Successfully loaded account state from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load account state from {file_path}: {e}")
            return False


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