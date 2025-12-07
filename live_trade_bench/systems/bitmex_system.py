"""
BitMEX Portfolio System for managing multiple LLM agents trading perpetual contracts.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List
import asyncio
import time
 
from datetime import timezone

from ..accounts import BitMEXAccount, create_bitmex_account
from ..agents.bitmex_agent import LLMBitMEXAgent
from ..fetchers.bitmex_fetcher import BitMEXFetcher
from ..fetchers.news_fetcher import fetch_news_data
import polars as pl
from ..utils.datetime_utils import parse_utc_datetime

logger = logging.getLogger(__name__)


class StepProfiler:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def add(self, name: str, seconds: float) -> None:
        # Accumulate seconds if the step already exists
        for r in self.records:
            if r.get("name") == name:
                r["seconds"] = float(r.get("seconds", 0.0)) + float(seconds)
                return
        self.records.append({"name": name, "seconds": float(seconds)})

    def print_profile(self, logger: logging.Logger, total_seconds: float | None = None) -> None:
        total = total_seconds if total_seconds is not None else sum(r["seconds"] for r in self.records)
        logger.info("=== Profile Report ===")
        logger.info(f"Total elapsed: {total:.2f}s")
        for r in self.records:
            share = (r["seconds"] / total * 100.0) if total > 0 else 0.0
            logger.info(f"- {r['name']}: {r['seconds']:.2f}s ({share:.1f}%)")
        logger.info("======================")


class BitMEXPortfolioSystem:
    """
    Portfolio system for BitMEX perpetual contract trading.

    Manages multiple LLM agents, each with independent accounts trading
    crypto perpetual contracts with 4x daily rebalancing.
    """

    def __init__(self, universe_size: int = 15, name: str = "BitMEX") -> None:
        """
        Initialize BitMEX portfolio system.

        Args:
            universe_size: Number of contracts to track (default 15)
        """
        self.agents: Dict[str, LLMBitMEXAgent] = {}
        self.accounts: Dict[str, BitMEXAccount] = {}
        self.universe: List[str] = []
        self.contract_info: Dict[str, Dict[str, Any]] = {}
        self.cycle_count = 0
        self.universe_size = universe_size
        self.fetcher = BitMEXFetcher()
        self.name = name
        # Keep the latest generated allocations for saving/analytics
        self.last_allocations: Dict[str, Dict[str, float]] = {}
        self.profiler = StepProfiler()

    def initialize_for_live(self) -> None:
        """Initialize for live trading by fetching trending contracts."""
        trending = self.fetcher.get_trending_contracts(limit=self.universe_size)
        symbols = [contract["symbol"] for contract in trending]
        self.set_universe(symbols)
        logger.info(f"Initialized {self.name} system with {len(symbols)} contracts")

    def initialize_for_backtest(self, trading_days: List[datetime]) -> None:
        """
        Initialize for backtesting.

        Args:
            trading_days: List of trading dates
        """
        trending = self.fetcher.get_trending_contracts(limit=self.universe_size)
        symbols = [contract["symbol"] for contract in trending]
        self.set_universe(symbols)

    def set_universe(self, symbols: List[str]) -> None:
        """
        Set the universe of tradable contracts.

        Args:
            symbols: List of BitMEX contract symbols (e.g., ["XBTUSD", "ETHUSD"])
        """
        self.universe = symbols
        self.contract_info = {symbol: {"name": symbol} for symbol in symbols}

    def add_agent(
        self, name: str, initial_cash: float = 10000.0, model_name: str = "gpt-4o-mini"
    ) -> None:
        """
        Add a new LLM agent with dedicated account.

        Args:
            name: Agent display name
            initial_cash: Starting capital (default $10,000)
            model_name: LLM model identifier
        """
        if name in self.agents:
            return
        agent = LLMBitMEXAgent(name, model_name)
        account = create_bitmex_account(initial_cash)
        self.agents[name] = agent
        self.accounts[name] = account

    def run_cycle(self, for_date: str | None = None) -> None:
        """
        Execute one trading cycle for all agents.

        Fetches market data, generates allocations, and updates accounts.

        Args:
            for_date: Optional date for backtesting (YYYY-MM-DD format)
        """

        profiler = self.profiler
        total_start = time.perf_counter()
        logger.info(f"Cycle {self.cycle_count + 1} started for BitMEX System")
        if for_date:
            logger.info(f"Backtest mode - Date: {for_date}")
            current_time_str = for_date
        else:
            logger.info("Live Trading Mode (UTC)")
            current_time_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        self.cycle_count += 1
        logger.info("Fetching data for BitMEX perpetual contracts...")

        t0 = time.perf_counter()
        self._fetch_market_data(current_time_str if for_date else None)
        market_data = self.market_data.filter(pl.col("symbol").is_in(self.universe))
        profiler.add("fetch_market_data", time.perf_counter() - t0)

        t0 = time.perf_counter()
        news_data = self._fetch_news_data(market_data, current_time_str if for_date else None)
        profiler.add("fetch_news_data", time.perf_counter() - t0)

        t0 = time.perf_counter()
        allocations = self._generate_allocations(market_data, news_data, current_time_str)
        profiler.add("generate_allocations", time.perf_counter() - t0)

        t0 = time.perf_counter()
        self._update_accounts(allocations, market_data, current_time_str)
        profiler.add("update_accounts", time.perf_counter() - t0)

        self.last_allocations = allocations
        t0 = time.perf_counter()
        self.save_accounts_parquet(for_date=current_time_str)
        profiler.add("save_accounts_parquet", time.perf_counter() - t0)
        pass

    def _fetch_news_data(
        self, market_data: Dict[str, Any], for_date: str | None
    ) -> Dict[str, Any]:
        """
        Fetch crypto news for all contracts.

        Args:
            market_data: Market data dictionary
            for_date: Optional date for backtesting

        Returns:
            Dictionary mapping symbol to news articles
        """
        print("  - Fetching crypto news data...")
        news_data_map: Dict[str, Any] = {}

        # Map symbols to crypto names for better news queries
        symbol_to_crypto = {
            "XBTUSD": "Bitcoin",
            "XBTUSDT": "Bitcoin",
            "ETHUSD": "Ethereum",
            "ETHUSDT": "Ethereum",
            "SOLUSDT": "Solana",
            "BNBUSDT": "BNB Binance",
            "XRPUSDT": "XRP Ripple",
            "ADAUSDT": "Cardano",
            "DOGEUSDT": "Dogecoin",
            "AVAXUSDT": "Avalanche",
            "LINKUSDT": "Chainlink",
            "LTCUSDT": "Litecoin",
        }

        try:
            if for_date:
                ref = parse_utc_datetime(for_date) - timedelta(days=1)
            else:
                ref = datetime.now(timezone.utc)

            start_dt = ref - timedelta(days=3)
            # Include time component (UTC) in start/end strings
            start_date = start_dt.strftime("%Y-%m-%d") #  %H:%M:%S UTC")
            end_date = ref.strftime("%Y-%m-%d") # %H:%M:%S UTC")

            for symbol in list(self.universe):
                crypto_name = symbol_to_crypto.get(symbol, symbol)
                query = f"{crypto_name} crypto news"
                news_data_map[symbol] = fetch_news_data(
                    query,
                    start_date,
                    end_date,
                    max_pages=1,
                    ticker=symbol,
                    target_date=for_date,
                )
        except Exception as e:
            print(f"    - Crypto news data fetch failed: {e}")

        return news_data_map

    def _fetch_social_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch social media posts for crypto contracts.

        Returns:
            Dictionary mapping symbol to list of social posts
        """
        logger.info("Fetching crypto social media data...")
        from ..fetchers.reddit_fetcher import RedditFetcher

        social_data_map: Dict[str, List[Dict[str, Any]]] = {}
        fetcher = RedditFetcher()

        # Map symbols to searchable crypto terms
        symbol_to_search = {
            "XBTUSD": "Bitcoin",
            "XBTUSDT": "Bitcoin",
            "ETHUSD": "Ethereum",
            "ETHUSDT": "Ethereum",
            "ETH_XBT": "Ethereum",
            "SOLUSDT": "Solana",
            "SOL_USDT": "Solana",
            "BNBUSDT": "BNB",
            "XRPUSDT": "XRP",
            "ADAUSDT": "Cardano",
            "DOGEUSDT": "Dogecoin",
            "AVAXUSDT": "Avalanche",
            "LINKUSDT": "Chainlink",
            "LINK_USDT": "Chainlink",
            "LTCUSDT": "Litecoin",
            "BCHUSDT": "Bitcoin Cash",
            "PEPEUSDT": "Pepe",
            "FLOKIUSDT": "Floki",
            "BONK_USDT": "Bonk",
            "SHIBUSDT": "Shiba Inu",
            "SUIUSDT": "Sui",
            "ARBUSDT": "Arbitrum",
            "PUMPUSDT": "Pump",
            "STLS_USDT": "Starknet",
            "BMEX_USDT": "BitMEX",
        }

        for symbol in self.universe:
            try:
                crypto_name = symbol_to_search.get(symbol, symbol)
                logger.info(f"Fetching social data for crypto: {crypto_name} ({symbol})")

                # Fetch Reddit posts by crypto name query
                posts = fetcher.fetch(
                    category="crypto", query=crypto_name, max_limit=10
                )
                logger.info(f"Fetched {len(posts)} social posts for {symbol}")

                formatted_posts = []
                for post in posts:
                    formatted_posts.append({
                        "id": post.get("id", ""),
                        "title": post.get("title", ""),
                        "content": post.get("content", ""),
                        "author": post.get("author", "Unknown"),
                        "platform": "Reddit",
                        "url": post.get("url", ""),
                        "created_at": post.get("created_utc", ""),
                        "subreddit": post.get("subreddit", ""),
                        "upvotes": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "tag": symbol,
                    })
                social_data_map[symbol] = formatted_posts
            except Exception as e:
                logger.error(f"Failed to fetch social data for {symbol}: {e}")
                social_data_map[symbol] = []

        return social_data_map

    def _generate_allocations(
        self,
        market_data: Any,
        news_data: Dict[str, Any],
        for_date: str | None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate allocations for all agents concurrently using asyncio.gather.
 
        Args:
            market_data: Market data (Polars DataFrame or dict)
            news_data: News data dictionary
            for_date: Optional date string
 
        Returns:
            Dictionary mapping agent name to allocation
        """
        logger.info("Generating allocations for all agents (async)...")
 
        async def _gather_allocations_async() -> Dict[str, Dict[str, float]]:
            tasks: list[asyncio.Future] = []
            names: list[str] = []
            accounts: list[BitMEXAccount] = []
 
            for agent_name, agent in self.agents.items():
                logger.info(f"Queueing agent for allocation: {agent_name}")
                account = self.accounts[agent_name]
                account_data = account.get_account_data()
                # Run sync generate_allocation in a thread to allow concurrency
                task = asyncio.to_thread(
                    agent.generate_allocation, market_data, account_data, for_date, news_data
                )
                tasks.append(task)
                names.append(agent_name)
                accounts.append(account)
 
            results = await asyncio.gather(*tasks, return_exceptions=True)
            aggregated: Dict[str, Dict[str, float]] = {}
            for name, acct, res in zip(names, accounts, results):
                if isinstance(res, Exception):
                    logger.error(f"Allocation failed for {name}: {res}")
                    aggregated[name] = acct.target_allocations
                    continue
                allocation = res
                if allocation:
                    aggregated[name] = allocation
                    logger.info(
                        f"Allocation for {name}: "
                        f"{ {k: f'{v:.1%}' for k, v in list(allocation.items())[:5]} }"
                    )
                else:
                    logger.warning(
                        f"No allocation generated for {name}, keeping previous target"
                    )
                    aggregated[name] = acct.target_allocations
            return aggregated
 
        # Run the async gather in a fresh event loop (we're in a sync context)
        try:
            all_allocations = asyncio.run(_gather_allocations_async())
        except RuntimeError:
            # If already in an event loop (unlikely here), fallback to sequential
            logger.warning("Event loop already running; falling back to sequential allocation.")
            all_allocations: Dict[str, Dict[str, float]] = {}
            for agent_name, agent in self.agents.items():
                logger.info(f"Processing agent (fallback): {agent_name}")
                account = self.accounts[agent_name]
                account_data = account.get_account_data()
                try:
                    allocation = agent.generate_allocation(
                        market_data, account_data, for_date, news_data=news_data
                    )
                except Exception as e:
                    logger.error(f"Allocation failed for {agent_name}: {e}")
                    allocation = None
                if allocation:
                    all_allocations[agent_name] = allocation
                else:
                    all_allocations[agent_name] = account.target_allocations
        else:
            logger.info("All allocations generated")
        return all_allocations

    def _update_accounts(
        self,
        allocations: Dict[str, Dict[str, float]],
        market_data: Any,
        for_date: str | None = None,
    ) -> None:
        """
        Update all accounts with new allocations.

        Args:
            allocations: Dictionary mapping agent name to allocation
            market_data: Market data dictionary
            for_date: Optional date string
        """
        logger.info("Updating all accounts...")
        # Build price map supporting both dict and Polars DataFrame market_data
        price_map: Dict[str, float] = {}
        metadata_map: Dict[str, Dict[str, Any]] | None = None
        try:
            import polars as pl  # type: ignore
            if isinstance(market_data, pl.DataFrame):
                df = market_data
                if "symbol" in df.columns:
                    # Determine best available price column
                    candidate_cols = [
                        c for c in [
                            "current_price",
                            "trdp_last_15m",
                            "vwap_15m",
                            "mark_price_15m",
                            "bam_close_15m",
                            "index_price_15m",
                            "settle_price_15m",
                        ] if c in df.columns
                    ]
                    if candidate_cols:
                        price_expr = pl.coalesce([pl.col(c) for c in candidate_cols]).alias("price")
                        if "close_time" in df.columns:
                            latest = df.group_by("symbol").agg(pl.col("close_time").max().alias("max_ct"))
                            df_latest = (
                                df.join(latest, on="symbol", how="inner")
                                .filter(pl.col("close_time") == pl.col("max_ct"))
                                .select([pl.col("symbol"), price_expr])
                            )
                        else:
                            df_latest = df.select([pl.col("symbol"), price_expr]).unique(subset=["symbol"], keep="last")
                        for sym, px in df_latest.iter_rows():
                            try:
                                if px is not None:
                                    price_map[str(sym)] = float(px)
                            except Exception:
                                continue
                metadata_map = {}  # No dict-style metadata when using DataFrame
            elif isinstance(market_data, dict):
                price_map = {s: d.get("current_price") for s, d in market_data.items()}
                metadata_map = market_data
            else:
                metadata_map = {}
        except Exception:
            # Fallback: try dict path
            if isinstance(market_data, dict):
                price_map = {s: d.get("current_price") for s, d in market_data.items()}
                metadata_map = market_data
            else:
                metadata_map = {}

        for agent_name, allocation in allocations.items():
            account = self.accounts[agent_name]
            account.target_allocations = allocation

            try:
                account.apply_allocation(
                    allocation, price_map=price_map, metadata_map=metadata_map
                )

                # Capture LLM input/output for audit trail
                llm_input = None
                llm_output = None
                agent = self.agents.get(agent_name)
                if agent is not None:
                    llm_input = getattr(agent, "last_llm_input", None)
                    llm_output = getattr(agent, "last_llm_output", None)

                account.record_allocation(
                    metadata_map=metadata_map,
                    backtest_date=for_date,
                    llm_input=llm_input,
                    llm_output=llm_output,
                )

                logger.info(
                    f"Account for {agent_name} updated. "
                    f"New Value: ${account.get_total_value():,.2f}, "
                    f"Cash: ${account.cash_balance:,.2f}"
                )
            except Exception as e:
                logger.error(f"Failed to update account for {agent_name}: {e}")

        logger.info("All accounts updated")

    @classmethod
    def get_instance(cls):
        """Get singleton instance (for compatibility with mock systems)."""
        if not hasattr(cls, "_instance"):
            cls._instance = create_bitmex_portfolio_system()
        return cls._instance

    # -------------------------
    # Allocation export helpers
    # -------------------------
    def allocations_to_dataframe(self, for_date: str | None = None) -> pl.DataFrame:
        """
        Convert the latest allocations into a Polars DataFrame.
        Columns: ['timestamp', 'close_time', 'cycle', 'agent', 'symbol', 'weight']
        """
        if not self.last_allocations:
            return pl.DataFrame(
                {
                    "timestamp": [],
                    "close_time": [],
                    "cycle": [],
                    "agent": [],
                    "symbol": [],
                    "weight": [],
                }
            )

        # Build UTC datetime for timestamp; handle multiple input formats if for_date is provided
        now_utc = datetime.now(timezone.utc)
        if for_date:
            parsed = None
            for fmt in ("%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H%M%S"):
                try:
                    parsed = datetime.strptime(for_date, fmt)
                    break
                except Exception:
                    continue
            if parsed is None:
                # Fallback: keep current time if parsing fails
                ts_dt = now_utc
            else:
                # Assume UTC; if naive, attach UTC
                if parsed.tzinfo is None:
                    ts_dt = parsed.replace(tzinfo=timezone.utc)
                else:
                    ts_dt = parsed.astimezone(timezone.utc)
        else:
            ts_dt = now_utc

        rows: List[Dict[str, Any]] = []
        for agent_name, allocation in self.last_allocations.items():
            for symbol, weight in allocation.items():
                rows.append(
                    {
                        # datetime object (UTC)
                        "timestamp": ts_dt,
                        # epoch milliseconds (int)
                        "close_time": int(ts_dt.timestamp() * 1000),
                        "cycle": self.cycle_count,
                        "agent": agent_name,
                        "symbol": "USDT" if symbol == "CASH" else symbol,
                        "weight": float(weight),
                    }
                )
        
        return pl.DataFrame(rows, strict=False)


def create_bitmex_portfolio_system() -> BitMEXPortfolioSystem:
    """
    Create a new BitMEX portfolio system instance.

    Returns:
        Initialized BitMEXPortfolioSystem
    """
    return BitMEXPortfolioSystem()
