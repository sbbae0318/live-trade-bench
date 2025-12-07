"""
Binance Fetcher backed by S3 parquet minute-slice datasets.

Reads closest 15m slice per target datetime and provides a BitMEXFetcher-like API:
- get_trending_contracts
- get_price
- get_price_history
- get_price_with_history
- get_funding_rate
- get_orderbook
- get_recent_trades
- fetch (unified)
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import polars as pl
from dotenv.main import logger

from .base_fetcher import BaseFetcher
from ..utils.lru_cache import LRUCache

def _parse_interval(s: str) -> timedelta:
    try:
        unit = s[-1].lower()
        value = int(s[:-1])
        if unit == "m":
            return timedelta(minutes=value)
        if unit == "h":
            return timedelta(hours=value)
        if unit == "d":
            return timedelta(days=value)
    except Exception:
        pass
    # Fallback to 1h
    return timedelta(hours=1)


class BinanceFetcher(BaseFetcher):
    """Fetcher for Binance perpetuals using S3 minute-slice parquet data."""

    def __init__(
        self,
        min_delay: float = 0.2,
        max_delay: float = 0.6,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        df_cache_capacity: int = 3000,
    ):
        super().__init__(min_delay, max_delay)
        self.bucket = bucket or os.environ.get("BINANCE_S3_BUCKET", "horizonquant")
        self.prefix = (prefix or os.environ.get("BINANCE_S3_PREFIX", "collector")).rstrip("/")
        self.cache_dir = os.environ.get(
            "BINANCE_CACHE_DIR", os.path.join(os.getcwd(), ".binance_cache")
        )
        self.file_cache_size = 1000
        # In-memory DataFrame LRU cache (avoid repeated I/O / deserialization)
        self._df_cache: LRUCache[str, pl.DataFrame] = LRUCache(capacity=df_cache_capacity)

    # ---------- Public API ----------

    def get_trending_contracts(
        self, limit: int = 15, for_date: Optional[str] = None, test: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Return top contracts by base_volume_15m from closest AggTrades15m slice.
        """
        if test:
            # Static test set with heuristic scores (higher = more trending)
            trending_pairs = [
                ("BTCUSDT", 4),  # CoinGecko +2, Coinranking +2
                ("ETHUSDT", 1),  # CryptoSlate +1
                ("SOLUSDT", 1),  # CryptoSlate +1
                ("XRPUSDT", 1),  # CryptoSlate +1
                ("BNBUSDT", 1),  # Coinranking +1
                #("ADAUSDT", 0),
                #("DOGEUSDT", 1),  # CryptoSlate +1
                #("XMRUSDT", 0),
                #("AVAXUSDT", 0),
                #("LINKUSDT", 1),  # CryptoSlate +1
                #("TRXUSDT", 1),   # CryptoSlate +1
                #("ZECUSDT", 1),   # CoinGecko +1
                #("HYPEUSDT", 0),  # Hyperliquid (placeholder symbol for testing)
                #("XLMUSDT", 1),   # CryptoSlate +1
                #("BCHUSDT", 0),
            ]
            trending_pairs.sort(key=lambda x: (-x[1], x[0]))
            return [{"symbol": sym, "score": score} for sym, score in trending_pairs[:limit]]

        target_dt = self._resolve_target_datetime(for_date)
        price_key = self._create_parquet_key(
            self._price_base_prefix(target_dt), target_dt
        )
        if not price_key:
            return []

        df = self._load_parquet_pl(price_key)
        if df is None or df.height == 0:
            return []

        try:
            # Validate if schema available
            self._ensure_hq_system_on_path()
            from trade.binance_engine.schema.agg_trades_schema import AggTrades15mSchema

            df = AggTrades15mSchema.validate(df)
        except Exception:
            pass

        # Sort by base_volume_15m desc if exists, else vwap_15m desc
        sort_cols = [c for c in ["base_volume_15m", "vwap_15m"] if c in df.columns]
        if sort_cols:
            df = df.sort(by=sort_cols, descending=True)

        results: List[Dict[str, Any]] = []
        for row in df.head(limit).iter_rows(named=True):
            results.append(
                {
                    "symbol": row.get("symbol"),
                    "base_volume_15m": float(row.get("base_volume_15m") or 0.0),
                    "vwap_15m": float(row.get("vwap_15m") or 0.0),
                    "trdp_last_15m": float(row.get("trdp_last_15m") or 0.0),
                }
            )
        return results

    def get_price(self, symbol: str, price_type: str = "last", for_date: Optional[str] = None) -> float:
        """
        Get current-like price for symbol from closest AggTrades15m slice.
        - price_type: "last" uses trdp_last_15m; "vwap" uses vwap_15m
        """
        target_dt = self._resolve_target_datetime(for_date)
        price_key = self._create_parquet_key(
            self._price_base_prefix(target_dt), target_dt
        )
        df = self._load_parquet_pl_cached(price_key) if price_key else None
        if df is None or df.height == 0:
            raise ValueError("No price data available")
        try:
            self._ensure_hq_system_on_path()
            from trade.binance_engine.schema.agg_trades_schema import AggTrades15mSchema

            df = AggTrades15mSchema.validate(df)
        except Exception:
            pass
        sub = df.filter(df["symbol"] == symbol)
        if sub.height == 0:
            raise ValueError(f"No price data for {symbol}")
        row = sub.row(0, named=True)
        if price_type == "vwap":
            val = row.get("vwap_15m")
        else:
            val = row.get("trdp_last_15m", row.get("vwap_15m"))
        if val is None:
            raise ValueError(f"No {price_type} price for {symbol}")
        return float(val)

    def get_price_history(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> List[Dict[str, Any]]:
        """
        Minimal stub: returns empty history for now.
        Can be extended to aggregate multiple 15m slices into requested interval.
        """
        return []

    def get_agg_trades(self, symbol_list:List[str], dt:datetime, lookback_days:int = 10) -> pl.DataFrame:
        """
        Get agg trades using hybrid approach:
        - Daily data (lookback_days+1 days ago ~ 1 day ago): from daily_data/agg_trades_15m/{year}/{month}/{date}_15m.parquet
        - Recent data (today 00:00 ~ current time): from 15-minute slices using existing method
        """
        logger.info(f"Getting agg trades for {symbol_list} from {dt} with {lookback_days} days lookback")
        return self._get_market_data_hybrid(
            dt=dt,
            lookback_days=lookback_days,
            base_prefix_func=self._price_base_prefix,
            data_type="agg_trades_15m"
        )

    def get_price_with_history(
        self,
        symbol: str,
        dt: datetime,
        interval: str = "1h",
        count: int = 10,
        price_type: str = "last",
    ) -> Dict[str, Any]:
        """
        Return current-like price from closest 15m slice and build price_history
        sampled every `interval` going back `count` points from `dt` (inclusive).
        - dt: timezone-aware datetime (naive will be assumed UTC)
        - interval: e.g., "15m", "1h", "4h", "1d" (default "1h")
        - count: number of points including dt (default 10)
        """
        # Normalize dt to UTC
        if dt.tzinfo is None:
            target_dt = dt.replace(tzinfo=timezone.utc)
        else:
            target_dt = dt.astimezone(timezone.utc)

        # Parse interval string to timedelta
        
        step = _parse_interval(interval or "1h")
        # Build timeline (oldest -> newest)
        timeline: List[datetime] = [target_dt - step * i for i in range(count)]
        timeline.reverse()

        price_history: List[Dict[str, Any]] = []
        current_price: Optional[float] = None

        for ts in timeline:
            # Find closest 15m slice near ts
            price_key = self._create_parquet_key(self._price_base_prefix(ts), ts)
            df = self._load_parquet_pl_cached(price_key) if price_key else None
            if df is None or df.height == 0:
                continue
            try:
                self._ensure_hq_system_on_path()
                from trade.binance_engine.schema.agg_trades_schema import AggTrades15mSchema
                df = AggTrades15mSchema.validate(df)
            except Exception:
                pass
            sub = df.filter(df["symbol"] == symbol)
            if sub.height == 0:
                continue
            row = sub.row(0, named=True)
            if price_type == "vwap":
                val = row.get("vwap_15m")
            else:
                val = row.get("trdp_last_15m", row.get("vwap_15m"))
            if val is None:
                continue
            price_val = float(val)
            price_history.append(
                {
                    "timestamp": ts.isoformat(),
                    "price": price_val,
                }
            )
            current_price = price_val  # Overwrite until last successful point (dt)

        # If timeline failed to yield current price, try direct get_price at dt
        if current_price is None:
            current_price = self.get_price(
                symbol, "vwap" if price_type == "vwap" else "last", for_date=target_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            )

        return {
            "symbol": symbol,
            "current_price": current_price,
            "price_type": price_type,
            "price_history": price_history,
            "interval": interval,
            "count": count,
        }

    def get_funding_rate_df(self, symbol_list: List[str], dt:datetime, lookback_days:int = 10) -> pl.DataFrame:
        """
        Get funding rate using hybrid approach:
        - Daily data (lookback_days+1 days ago ~ 1 day ago): from daily_data/funding_rate_15m/{year}/{month}/{date}_15m.parquet
        - Recent data (today 00:00 ~ current time): from 15-minute slices using existing method
        """
        logger.info(f"Getting funding rate for {symbol_list} from {dt} with {lookback_days} days lookback")
        return self._get_market_data_hybrid(
            dt=dt,
            lookback_days=lookback_days,
            base_prefix_func=self._funding_base_prefix,
            data_type="funding_rate_15m"
        )

    def get_orderbook_df(self, symbol_list: List[str], dt:datetime, lookback_days:int = 10) -> pl.DataFrame:
        """
        Get orderbook using hybrid approach:
        - Daily data (lookback_days+1 days ago ~ 1 day ago): from daily_data/orderbook_summary_15m/{year}/{month}/{date}_15m.parquet
        - Recent data (today 00:00 ~ current time): from 15-minute slices using existing method
        """
        logger.info(f"Getting orderbook for {symbol_list} from {dt} with {lookback_days} days lookback")
        return self._get_market_data_hybrid(
            dt=dt,
            lookback_days=lookback_days,
            base_prefix_func=self._orderbook_base_prefix,
            data_type="orderbook_summary_15m"
        )


    def get_funding_rate(self, symbol: str, for_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get funding rate from closest FundingRates15m slice.
        """
        target_dt = self._resolve_target_datetime(for_date)
        key = self._create_parquet_key(self._funding_base_prefix(target_dt), target_dt)
        df = self._load_parquet_pl(key) if key else None
        if df is None or df.height == 0:
            return {"symbol": symbol, "funding_rate": 0.0}
        try:
            self._ensure_hq_system_on_path()
            from trade.binance_engine.schema.funding_rates_schema import FundingRates15mSchema

            df = FundingRates15mSchema.validate(df)
        except Exception:
            pass
        sub = df.filter(df["symbol"] == symbol)
        if sub.height == 0:
            return {"symbol": symbol, "funding_rate": 0.0}
        row = sub.row(0, named=True)
        return {
            "symbol": symbol,
            "funding_rate": float(row.get("funding_rate_15m") or 0.0),
            "next_funding_time": row.get("next_funding_time_15m"),
        }

    def get_orderbook(self, symbol: str, for_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Return orderbook summary (not full L2) from OrderbookSummary15m slice.
        """
        target_dt = self._resolve_target_datetime(for_date)
        key = self._create_parquet_key(self._orderbook_base_prefix(target_dt), target_dt)
        df = self._load_parquet_pl(key) if key else None
        if df is None or df.height == 0:
            return {"symbol": symbol, "bids": [], "asks": [], "bid_depth": 0.0, "ask_depth": 0.0}
        try:
            self._ensure_hq_system_on_path()
            from trade.binance_engine.schema.orderbook_summary_schema import OrderbookSummary15mSchema

            df = OrderbookSummary15mSchema.validate(df)
        except Exception:
            pass
        sub = df.filter(df["symbol"] == symbol)
        if sub.height == 0:
            return {"symbol": symbol, "bids": [], "asks": [], "bid_depth": 0.0, "ask_depth": 0.0}
        row = sub.row(0, named=True)
        # Notional depth = base volume at depth-20 × close price (15m)
        bam_close = row.get("bam_close_15m")
        bidvol = row.get("bidvol_d20_15m")
        askvol = row.get("askvol_d20_15m")
        bid_depth = float(bidvol) * float(bam_close) if bidvol is not None and bam_close is not None else 0.0
        ask_depth = float(askvol) * float(bam_close) if askvol is not None and bam_close is not None else 0.0
        return {
            "symbol": symbol,
            "bids": [],
            "asks": [],
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_recent_trades(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]:
        """
        Not implemented for S3-backed fetcher. Returns empty list.
        """
        return []

    def fetch(self, mode: str, **kwargs) -> Any:
        if mode == "trending":
            return self.get_trending_contracts(
                limit=kwargs.get("limit", 15),
                for_date=kwargs.get("for_date"),
                test=kwargs.get("test", False),
            )
        elif mode == "price":
            return self.get_price(
                symbol=kwargs["symbol"], price_type=kwargs.get("price_type", "last"), for_date=kwargs.get("for_date")
            )
        elif mode == "price_with_history":
            # Backward-compat: accept either explicit dt or for_date string
            dt = kwargs.get("dt")
            if dt is None:
                for_date = kwargs.get("for_date")
                if for_date:
                    dt = self._resolve_target_datetime(for_date)
                else:
                    dt = datetime.now(timezone.utc)
            return self.get_price_with_history(
                symbol=kwargs["symbol"],
                dt=dt,
                interval=kwargs.get("interval", "1h"),
                count=kwargs.get("count", 10),
                price_type=kwargs.get("price_type", "last"),
            )
        elif mode == "history":
            return self.get_price_history(
                symbol=kwargs["symbol"],
                start_date=kwargs["start_date"],
                end_date=kwargs["end_date"],
                interval=kwargs.get("interval", "1d"),
            )
        elif mode == "funding":
            return self.get_funding_rate(symbol=kwargs["symbol"], for_date=kwargs.get("for_date"))
        elif mode == "orderbook":
            return self.get_orderbook(symbol=kwargs["symbol"], for_date=kwargs.get("for_date"))
        elif mode == "trades":
            return self.get_recent_trades(symbol=kwargs["symbol"], count=kwargs.get("count", 100))
        else:
            raise ValueError(f"Unknown fetch mode: {mode}")

    # ---------- Internal helpers ----------

    def _resolve_target_datetime(self, for_date: Optional[str]) -> datetime:
        if not for_date:
            return datetime.now(timezone.utc)
        s = for_date.strip()
        # Normalize common variants
        if s.endswith(" UTC"):
            s = s[:-4].strip()
        if "T" in s and len(s) == 15:  # e.g., 2025-11-25T011500
            try:
                dt = datetime.strptime(s, "%Y-%m-%dT%H%M%S")
                return dt.replace(tzinfo=timezone.utc)
            except Exception:
                pass
        # Replace 'T' with space for ISO-like strings without timezone
        s_space = s.replace("T", " ")
        # Try most specific to least specific
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s_space, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
        # Last resort: fromisoformat on cleaned string
        try:
            dt = datetime.fromisoformat(s_space)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    def _day_suffix(self, dt: datetime) -> str:
        """
        Build day-based path segment. If the timestamp is exactly 00:00:00,
        roll forward to the next calendar day so that a 00:00 slice is stored
        under the following day's folder.
        """
        d = dt
        # Normalize to UTC for consistent day boundaries
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        else:
            d = d.astimezone(timezone.utc)
        if d.hour == 0 and d.minute == 0 and d.second == 0 and d.microsecond == 0:
            d = d - timedelta(days=1)
        return f"{d.strftime('%Y')}/{d.strftime('%m')}/{d.strftime('%d')}/15m/"

    def _price_base_prefix(self, dt: datetime) -> str:
        return f"{self.prefix}/MinuteSliceFuturesAggTradesCollectionManager/{self._day_suffix(dt)}"

    def _orderbook_base_prefix(self, dt: datetime) -> str:
        return f"{self.prefix}/MinuteSliceFuturesOrderbookSummaryCollectionManager/{self._day_suffix(dt)}"

    def _funding_base_prefix(self, dt: datetime) -> str:
        return f"{self.prefix}/MinuteSliceFuturesFundingRateCollectionManager/{self._day_suffix(dt)}"

    def _parse_slice_timestamp(self, key: str) -> Optional[datetime]:
        try:
            import re
            filename = key.rsplit("/", 1)[-1]
            # Strip .parquet and any trailing suffix like ".2" before extension
            if filename.endswith(".parquet"):
                filename = filename[: -len(".parquet")]
            # Extract strict pattern YYYY-MM-DDTHHMMSS
            m = re.search(r"\d{4}-\d{2}-\d{2}T\d{6}", filename)
            if not m:
                return None
            ts_str = m.group(0)
            dt = datetime.strptime(ts_str, "%Y-%m-%dT%H%M%S")
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def _create_daily_parquet_key(self, target_date: datetime, data_type: str = "agg_trades_15m") -> Optional[str]:
        """
        Create daily parquet key in format: daily_data/{data_type}/{year}/{month}/{year}-{month}-{day}_15m.parquet
        Example: daily_data/agg_trades_15m/2025/11/2025-11-04_15m.parquet
        Example: daily_data/funding_rate_15m/2025/11/2025-11-04_15m.parquet
        Example: daily_data/orderbook_summary_15m/2025/11/2025-11-04_15m.parquet
        """
        try:
            # Normalize to UTC
            if target_date.tzinfo is None:
                dt = target_date.replace(tzinfo=timezone.utc)
            else:
                dt = target_date.astimezone(timezone.utc)
            # Use date at 00:00:00
            date_only = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            year = date_only.strftime("%Y")
            month = date_only.strftime("%m")
            day = date_only.strftime("%d")
            filename = f"{year}-{month}-{day}_15m.parquet"
            return f"daily_data/{data_type}/{year}/{month}/{filename}"
        except Exception:
            return None

    def _get_market_data_hybrid(
        self,
        dt: datetime,
        lookback_days: int,
        base_prefix_func,
        data_type: str = "agg_trades_15m"
    ) -> pl.DataFrame:
        """
        Common function to get market data using hybrid approach:
        - Daily data (lookback_days+1 days ago ~ 1 day ago): from daily_data/{data_type}/...
        - Recent data (today 00:00 ~ current time): from 15-minute slices using existing method
        
        Args:
            dt: Target datetime
            lookback_days: Number of days to look back
            base_prefix_func: Function to get base prefix for 15-minute slices (e.g., self._price_base_prefix)
            data_type: Data type for daily path (e.g., "agg_trades_15m", "funding_rate_15m", "orderbook_summary_15m")
        """
        if dt.tzinfo is None:
            target_dt = dt.replace(tzinfo=timezone.utc)
        else:
            target_dt = dt.astimezone(timezone.utc)

        # Calculate today 00:00:00 UTC
        today_start = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Collect keys: Daily data for past days + 15-minute slices for today
        keys: List[str] = []
        
        # Daily data: (lookback_days+1) days ago ~ 1 day ago
        # Example: if lookback_days=10, get data from 11 days ago to 1 day ago (inclusive)
        for day_offset in range(lookback_days + 1, 0, -1):  # From (lookback_days+1) to 1 (inclusive)
            daily_date = today_start - timedelta(days=day_offset)
            daily_key = self._create_daily_parquet_key(daily_date, data_type=data_type)
            if daily_key:
                keys.append(daily_key)
        
        # Recent data: Today 00:15 ~ current time using 15-minute slices
        # Skip 00:00 as it's already included in yesterday's daily data
        # Generate 15-minute intervals from 00:15 to target_dt
        step_15m = timedelta(minutes=15)
        current_ts = today_start + step_15m  # Start from 00:15, skip 00:00
        while current_ts <= target_dt:
            key = self._create_parquet_key(base_prefix_func(current_ts), current_ts)
            if key:
                keys.append(key)
            current_ts += step_15m

        logger.info(f"Loading {len(keys)} parquet files in parallel ({lookback_days} days lookback, type: {data_type})")
        df = self._load_parquet_many_async_entry(keys)
        try:
            import polars as pl
            if df is None or (hasattr(df, "is_empty") and df.is_empty()):
                return pl.DataFrame()
            return df
        except Exception:
            return df

    def _create_parquet_key(self, base_prefix: str, target_dt: datetime) -> Optional[str]:
        """
        Build parquet key by rule without probing S3.
        Align to 15-minute boundaries and format as:
        {base_prefix}{YYYY-MM-DDTHHMMSS}.parquet
        """
        try:
            # Normalize to UTC
            if target_dt.tzinfo is None:
                dt = target_dt.replace(tzinfo=timezone.utc)
            else:
                dt = target_dt.astimezone(timezone.utc)
            # Align to 15-minute slice
            dt_aligned = self._align_to_interval_minutes(dt, 15)
            filename = f"{dt_aligned.strftime('%Y-%m-%dT%H%M%S')}.parquet"
            return f"{base_prefix}{filename}"
        except Exception:
            return None

    def _previous_parquet_key(self, key: str) -> Optional[str]:
        """
        Given an existing parquet key, return the previous 15-minute slice key.
        Directory rule:
        - If the target timestamp becomes exactly 00:00:00 UTC, the directory path
          uses the previous calendar day (while the filename keeps the true timestamp).
        """
        try:
            ts = self._parse_slice_timestamp(key)
            if ts is None:
                return None
            prev_ts = ts - timedelta(minutes=15)
            # Extract manager base prefix up to the YYYY segment
            parts = key.split("/")
            year_idx = None
            for i, p in enumerate(parts):
                if len(p) == 4 and p.isdigit():
                    year_idx = i
                    break
            if year_idx is None or year_idx < 2:
                return None
            base_manager_prefix = "/".join(parts[:year_idx]) + "/"
            base_prefix = f"{base_manager_prefix}{self._day_suffix(prev_ts)}"
            filename = f"{prev_ts.strftime('%Y-%m-%dT%H%M%S')}.parquet"
            return f"{base_prefix}{filename}"
        except Exception:
            return None

    def _find_latest_parquet_key_for_day(self, base_prefix: str, day_dt: datetime) -> Optional[str]:
        """
        Build latest slice key for the given day by rule: align to 23:45 UTC (15m cadence).
        """
        try:
            # Normalize to UTC date
            if day_dt.tzinfo is None:
                d = day_dt.replace(tzinfo=timezone.utc)
            else:
                d = day_dt.astimezone(timezone.utc)
            # Target last 15m slice of the day: 23:45:00
            d_last = d.replace(hour=23, minute=45, second=0, microsecond=0)
            filename = f"{d_last.strftime('%Y-%m-%dT%H%M%S')}.parquet"
            return f"{base_prefix}{filename}"
        except Exception:
            return None

    def _load_parquet_pl(self, key: str):
        try:
            import boto3
            import polars as pl
            # in-memory LRU cache
            df_mem = self._df_cache.get(key)
            if df_mem is not None:
                return df_mem

            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=self.bucket, Key=key)
            body = obj["Body"].read()
            df = pl.read_parquet(io.BytesIO(body))
            self._df_cache.put(key, df)
            return df
        except Exception:
            return None

    def _load_parquet_pl_cached(self, key: str):
        """
        Load parquet with local caching. If exists locally, reuse without re-downloading.
        """
        if not key:
            return None
        try:
            import polars as pl
            # Build cache path
            cache_path = self._cache_path_for_key(key)
            df_mem = self._df_cache.get(key)
            if df_mem is not None:
                return df_mem
            if os.path.exists(cache_path):
                df = pl.read_parquet(cache_path)
                self._df_cache.put(key, df)
                return df

            # Download and cache
            import boto3

            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=self.bucket, Key=key)
            body = obj["Body"].read()
            # Ensure dir exists and write
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(body)

            logger.info(f"Cached {key} to {cache_path}")
            df = pl.read_parquet(cache_path)
            self._df_cache.put(key, df)
            return df
        except Exception:
            return None

    # ---------- Async high-throughput loading ----------
    def _get_common_columns(self, dfs: List[pl.DataFrame]) -> List[str]:
        """
        Get common columns across all DataFrames.
        Returns empty list if no DataFrames or no common columns.
        """
        if not dfs:
            return []
        # Start with first DataFrame's columns
        common_cols = set(dfs[0].columns)
        # Intersect with all other DataFrames
        for df in dfs[1:]:
            common_cols &= set(df.columns)
        return sorted(list(common_cols))

    def _concat_with_common_columns(self, dfs: List[pl.DataFrame]) -> Optional[pl.DataFrame]:
        """
        Concatenate DataFrames using only common columns.
        Normalizes Datetime column types to microseconds to avoid type mismatch errors.
        Returns None if no DataFrames or no common columns.
        """
        if not dfs:
            return None
        import polars as pl
        common_cols = self._get_common_columns(dfs)
        if not common_cols:
            logger.warning("No common columns found among DataFrames, returning empty DataFrame")
            return pl.DataFrame()
        
        # Select only common columns and normalize Datetime types
        dfs_filtered = []
        for df in dfs:
            df_selected = df.select(common_cols)
            # Normalize Datetime columns to microseconds to avoid type mismatch
            # Check for Datetime columns and normalize their time_unit
            cast_exprs = []
            for col in df_selected.columns:
                dtype = df_selected[col].dtype
                # Check if dtype is Datetime (using string representation for safety)
                dtype_str = str(dtype)
                if dtype_str.startswith("Datetime"):
                    # Extract time_unit and time_zone from dtype
                    time_unit = getattr(dtype, "time_unit", "us")
                    time_zone = getattr(dtype, "time_zone", None)
                    # Convert to microseconds (most compatible format)
                    if time_unit != "us":
                        cast_exprs.append(
                            pl.col(col).cast(pl.Datetime("us", time_zone))
                        )
            if cast_exprs:
                df_selected = df_selected.with_columns(cast_exprs)
            dfs_filtered.append(df_selected)
        
        return pl.concat(dfs_filtered, how="vertical", rechunk=True)

    def _load_parquet_many_async_entry(self, keys: List[str]):
        """
        Synchronous entrypoint that runs async loader when possible.
        Falls back to sequential sync loading if event loop is already running
        or aioboto3 is unavailable.
        """
        if not keys:
            return None
        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                # Fallback: sequential sync to avoid nested event loop issues
                dfs = []
                for k in keys:
                    df = self._load_parquet_pl_cached(k)
                    if df is None or getattr(df, "height", 0) == 0:
                        # try previous slice (ffill-like)
                        prev_k = self._previous_parquet_key(k)
                        if prev_k:
                            df = self._load_parquet_pl_cached(prev_k)
                    if df is not None and getattr(df, "height", 0) > 0:
                        dfs.append(df)
                if not dfs:
                    return None
                return self._concat_with_common_columns(dfs)
            else:
                return asyncio.run(self._load_parquet_many_async(keys))
        except Exception:
            # Last-resort fallback: sequential sync
            dfs = []
            for k in keys:
                df = self._load_parquet_pl_cached(k)
                if df is None or getattr(df, "height", 0) == 0:
                    prev_k = self._previous_parquet_key(k)
                    if prev_k:
                        df = self._load_parquet_pl_cached(prev_k)
                if df is not None and getattr(df, "height", 0) > 0:
                    dfs.append(df)
            if not dfs:
                return None
            return self._concat_with_common_columns(dfs)

    async def _load_parquet_many_async(self, keys: List[str]):
        """
        Load multiple parquet objects concurrently using aioboto3, with local caching.
        """
        if not keys:
            return None
        try:
            import aioboto3  # type: ignore
            import asyncio

            session = aioboto3.Session()
            async with session.client("s3") as s3:
                tasks = [self._fetch_parquet_with_cache_async(s3, k) for k in keys]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                dfs = []
                missing_indices = []
                for idx, res in enumerate(results):
                    if hasattr(res, "height") and getattr(res, "height", 0) > 0:
                        dfs.append(res)
                    else:
                        missing_indices.append(idx)
                # Try previous slice for missing ones (ffill-like)
                if missing_indices:
                    prev_tasks = []
                    prev_map = {}
                    for idx in missing_indices:
                        prev_k = self._previous_parquet_key(keys[idx])
                        if prev_k:
                            prev_map[idx] = prev_k
                            prev_tasks.append(self._fetch_parquet_with_cache_async(s3, prev_k))
                    if prev_tasks:
                        prev_results = await asyncio.gather(*prev_tasks, return_exceptions=True)
                        for res in prev_results:
                            if hasattr(res, "height") and getattr(res, "height", 0) > 0:
                                dfs.append(res)
            if not dfs:
                return None
            return self._concat_with_common_columns(dfs)
        except Exception as e:
            logger.error(f"Error loading parquet files: {e}")
            return None

    async def _fetch_parquet_with_cache_async(self, s3, key: str):
        """
        Async fetch with caching: if cached parquet exists, read it off-thread;
        otherwise download via aioboto3 and cache to disk before reading.
        """
        if not key:
            return None
        try:
            import os
            import polars as pl
            import asyncio
            # no-op
            cache_path = self._cache_path_for_key(key)
            df_mem = self._df_cache.get(key)
            if df_mem is not None:
                return df_mem
            if os.path.exists(cache_path):
                df = await asyncio.to_thread(pl.read_parquet, cache_path)
                self._df_cache.put(key, df)
                return df

            # Ensure parent dir
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            # Download asynchronously
            obj = await s3.get_object(Bucket=self.bucket, Key=key)
            body = await obj["Body"].read()
            # Write and read off-thread to avoid blocking the event loop
            await asyncio.to_thread(self._write_bytes, cache_path, body)
            df = await asyncio.to_thread(pl.read_parquet, cache_path)
            self._df_cache.put(key, df)
            return df
        except Exception as e:
            logger.error(f"Error fetching parquet file key: {key}, {e}")
            return None

    def _write_bytes(self, path: str, data: bytes) -> None:
        with open(path, "wb") as f:
            f.write(data)

    def _cache_path_for_key(self, key: str) -> str:
        # mirror bucket/key under cache dir
        safe_key_path = key.replace("/", os.sep)
        return os.path.join(self.cache_dir, self.bucket, safe_key_path)

    def _ensure_hq_system_on_path(self) -> None:
        """
        Ensure external/hq-system is importable at runtime for schema imports.
        """
        try:
            import sys
            here = os.path.abspath(os.path.dirname(__file__))
            project_root = os.path.abspath(os.path.join(here, "..", ".."))
            hq_path = os.path.join(project_root, "external", "hq-system")
            if hq_path not in sys.path and os.path.isdir(hq_path):
                sys.path.insert(0, hq_path)
        except Exception:
            pass

    # ---------- Time alignment helpers ----------
    def _align_to_interval_minutes(self, dt: datetime, minutes: int) -> datetime:
        """
        Floor a datetime to nearest lower interval in minutes. Keeps UTC tzinfo.
        """
        if minutes <= 0:
            return dt.replace(second=0, microsecond=0)
        minute = (dt.minute // minutes) * minutes
        return dt.replace(minute=minute, second=0, microsecond=0)


