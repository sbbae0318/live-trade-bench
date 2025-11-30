"""
BitMEX LLM Agent for crypto perpetual contract trading.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import polars as pl
from ..accounts import BitMEXAccount
from .base_agent import BaseAgent


class LLMBitMEXAgent(BaseAgent[BitMEXAccount, pl.DataFrame]):
    """LLM-powered trading agent for BitMEX perpetual contracts."""

    def __init__(self, name: str, model_name: str = "gpt-4o-mini") -> None:
        super().__init__(name, model_name)

    def _prepare_market_analysis(self, market_data: pl.DataFrame) -> str:
        """
        Prepare market analysis for BitMEX perpetual contracts.

        Includes crypto-specific data:
        - BTC/USD price formatting
        - Funding rates
        - Order book depth
        - Open interest
        """
        df = market_data
        from datetime import timedelta

        needed = {
            "symbol",
            "close_time",
            "trdp_last_15m",
            "vwap_15m",
            "base_volume_15m",
            "funding_rate_15m",
        }
        cols = [c for c in df.columns if c in needed]
        df_sel = df.select(cols) if cols else df

        df_sel = df_sel.with_columns(
            [
                pl.col("close_time")
                .cast(pl.Datetime(time_unit="ms", time_zone="UTC"))
                .dt.date()
                .alias("date"),
                pl.coalesce([pl.col("trdp_last_15m"), pl.col("vwap_15m")]).alias("price_pref"),
            ]
        )

        latest_date = df_sel.select(pl.col("date").max()).to_series().item()
        if latest_date is None:
            return "MARKET ANALYSIS:\n"

        start_date = latest_date - timedelta(days=(self.lookback_days))
        df_10d = df_sel.filter(pl.col("date") >= pl.lit(start_date))

        daily = (
            df_10d.group_by(["symbol", "date"])
            .agg(
                [
                    pl.col("vwap_15m").mean().alias("daily_vwap"),
                    pl.col("base_volume_15m").sum().alias("daily_volume"),
                    pl.col("price_pref").sort_by("close_time").last().alias("daily_close"),
                    pl.col("close_time").max().alias("last_ct"),
                ]
            )
            .sort(["symbol", "date"])
        )

        # Compute previous day metrics and deltas using window functions (no row-by-row loops)
        daily_changes = (
            daily.sort(["symbol", "date"])
            .with_columns(
                [
                    pl.col("daily_close").shift(1).over("symbol").alias("prev_close"),
                    pl.col("daily_volume").shift(1).over("symbol").alias("prev_volume"),
                    pl.col("daily_vwap").shift(1).over("symbol").alias("prev_vwap"),
                ]
            )
            .with_columns(
                [
                    (pl.col("daily_close") - pl.col("prev_close")).alias("close_abs_chg"),
                    pl.when(pl.col("prev_close") > 0)
                    .then((pl.col("daily_close") - pl.col("prev_close")) / pl.col("prev_close") * 100.0)
                    .otherwise(None)
                    .alias("close_pct_chg"),
                    (pl.col("daily_volume") - pl.col("prev_volume")).alias("volume_abs_chg"),
                    pl.when(pl.col("prev_volume") > 0)
                    .then((pl.col("daily_volume") - pl.col("prev_volume")) / pl.col("prev_volume") * 100.0)
                    .otherwise(None)
                    .alias("volume_pct_chg"),
                    (pl.col("daily_vwap") - pl.col("prev_vwap")).alias("vwap_abs_chg"),
                    pl.when(pl.col("prev_vwap") > 0)
                    .then((pl.col("daily_vwap") - pl.col("prev_vwap")) / pl.col("prev_vwap") * 100.0)
                    .otherwise(None)
                    .alias("vwap_pct_chg"),
                ]
            )
            .with_columns(
                [
                    pl.col("close_abs_chg").fill_null(0.0),
                    pl.col("close_pct_chg").fill_null(0.0),
                    pl.col("volume_abs_chg").fill_null(0.0),
                    pl.col("volume_pct_chg").fill_null(0.0),
                    pl.col("vwap_abs_chg").fill_null(0.0),
                    pl.col("vwap_pct_chg").fill_null(0.0),
                    (pl.col("date") == pl.lit(latest_date)).alias("is_latest"),
                ]
            )
        )
        
        cutoff_date = latest_date - timedelta(days=(self.lookback_days - 1))
        daily_changes = daily_changes.filter(pl.col("date") >= pl.lit(cutoff_date))

        latest_by_symbol = (
            df_sel.group_by("symbol")
            .agg(
                [
                    pl.col("close_time").max().alias("max_ct"),
                    pl.col("funding_rate_15m").sort_by("close_time").last().alias("current_funding"),
                    pl.col("price_pref").sort_by("close_time").last().alias("current_price"),
                ]
            )
        )

        # Build per-day formatted lines using expressions (no Python inner loops)
        sign_abs = pl.when(pl.col("close_abs_chg") >= 0).then(pl.lit("+")).otherwise(pl.lit(""))
        sign_pct = pl.when(pl.col("close_pct_chg") >= 0).then(pl.lit("+")).otherwise(pl.lit(""))
        sign_v_abs = pl.when(pl.col("volume_abs_chg") >= 0).then(pl.lit("+")).otherwise(pl.lit(""))
        sign_v_pct = pl.when(pl.col("volume_pct_chg") >= 0).then(pl.lit("+")).otherwise(pl.lit(""))
        sign_w_abs = pl.when(pl.col("vwap_abs_chg") >= 0).then(pl.lit("+")).otherwise(pl.lit(""))
        sign_w_pct = pl.when(pl.col("vwap_pct_chg") >= 0).then(pl.lit("+")).otherwise(pl.lit(""))

        date_hdr = pl.format("  - {}: ", pl.col("date"))
        price_line = pl.format(
            "      price: {} (Change: {}{} ({}{}%))",
            pl.col("daily_close").round(4),
            sign_abs,
            pl.col("close_abs_chg").round(2),
            sign_pct,
            pl.col("close_pct_chg").round(2),
        )
        vol_line = pl.format(
            "      24h trading volume: {} (Change: {}{} ({}{}%))",
            pl.col("daily_volume").round(0),
            sign_v_abs,
            pl.col("volume_abs_chg").round(0),
            sign_v_pct,
            pl.col("volume_pct_chg").round(2),
        )
        vwap_line = pl.format(
            "      24h VWAP: {} (Change: {}{} ({}{}%))",
            pl.col("daily_vwap").round(4),
            sign_w_abs,
            pl.col("vwap_abs_chg").round(4),
            sign_w_pct,
            pl.col("vwap_pct_chg").round(2),
        )
        # For the latest date (today), omit the 24h volume/VWAP lines
        vol_line_final = pl.when(pl.col("is_latest")).then(pl.lit("")).otherwise(vol_line)
        vwap_line_final = pl.when(pl.col("is_latest")).then(pl.lit("")).otherwise(vwap_line)

        daily_text = (
            daily_changes
            .with_columns(
                pl.concat_str(
                    [
                        date_hdr,
                        pl.lit("\n"),
                        price_line,
                        pl.lit("\n"),
                        vol_line_final,
                        pl.lit("\n"),
                        vwap_line_final,
                        pl.lit("\n\n"),
                    ],
                    separator=""
                ).alias("day_block")
            )
            .group_by("symbol")
            .agg(pl.col("day_block").str.concat(delimiter="").alias("daily_block"))
        )

        # Header per symbol (current price and funding)
        header = (
            latest_by_symbol
            .with_columns(
                [
                    pl.format(
                        "{}: Current price is ${}",
                        pl.col("symbol"),
                        pl.col("current_price").round(2),
                    ).alias("h1"),
                    pl.when(pl.col("current_funding").is_not_null())
                    .then(pl.format("  - Current Funding rate: {}%", (pl.col("current_funding") * 100.0).round(4)))
                    .otherwise(pl.lit(None))
                    .alias("h2"),
                    pl.lit("  - Past 24hrs Trading volume").alias("h3"),
                    pl.lit("  - Past 24hrs VWAP").alias("h4"),
                ]
            )
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("h1"),
                        pl.lit("\n"),
                        pl.when(pl.col("h2").is_not_null()).then(pl.col("h2") + pl.lit("\n")).otherwise(pl.lit("")),
                        pl.col("h3"), pl.lit("\n"),
                        pl.col("h4"), pl.lit("\n\n"),
                    ],
                    separator=""
                ).alias("header_text")
            )
            .select(["symbol", "header_text"])
        )

        # Merge header with daily aggregated text
        per_symbol_text = (
            header.join(daily_text, on="symbol", how="left")
            .with_columns(
                pl.concat_str([pl.col("header_text"), pl.col("daily_block").fill_null("")], separator="").alias("full_block")
            )
            .select(["full_block"])
        )

        # Build final MARKET ANALYSIS with a single join over blocks
        blocks = per_symbol_text["full_block"].to_list()
        return "MARKET ANALYSIS:\n" + "\n".join(blocks)

    def _create_news_query(self, ticker: str, data: Dict[str, Any]) -> str:
        """
        Create crypto-specific news query.

        Maps contract symbols to readable cryptocurrency names.
        """
        # Map common BitMEX symbols to crypto names
        symbol_map = {
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

        crypto_name = symbol_map.get(ticker, ticker)
        return f"{crypto_name} crypto news"

    def _get_portfolio_prompt(
        self,
        analysis: str,
        market_data: pl.DataFrame,
        date: Optional[str] = None,
    ) -> str:
        """
        Generate LLM prompt for BitMEX portfolio allocation.

        Includes crypto-specific considerations:
        - 24/7 market volatility
        - Funding rate carry costs
        - Market liquidity from order book depth
        """
        current_date_str = f"Today is {date} (UTC)." if date else ""
        try:
            symbols = (
                market_data.select(pl.col("symbol").unique()).to_series().to_list()
                if "symbol" in market_data.columns
                else []
            )
        except Exception:
            symbols = []
        contract_list = symbols
        contract_list_str = ", ".join(contract_list)
        sample = [
            contract_list[i] if i < len(contract_list) else f"CONTRACT_{i+1}"
            for i in range(3)
        ]

        return (
            f"{current_date_str}\n\n"
            "You are a professional crypto derivatives trader managing a perpetual contract portfolio on BitMEX. "
            "Analyze the market data and generate a complete portfolio allocation.\n\n"
            f"{analysis}\n\n"
            "PORTFOLIO MANAGEMENT OBJECTIVE:\n"
            "- Maximize risk-adjusted returns by selecting contracts with favorable risk/reward profiles.\n"
            "- Consider funding rates as they affect carry costs (paid every 8 hours).\n"
            "- Outperform equal-weight baseline over 1-2 week timeframes.\n"
            "- CRITICAL: Preserve capital during downtrends - significantly reduce crypto exposure and increase CASH when markets decline.\n"
            "- In strong downtrends (>5% decline): Move to 60-80% CASH for capital preservation.\n"
            "- In strong uptrends (>5% gain): Increase crypto exposure to 60-80% to capture momentum.\n\n"
            "CRYPTO MARKET CONSIDERATIONS:\n"
            "- Markets trade 24/7 with high volatility.\n"
            "- Funding rates create carry costs/profits (positive rate = longs pay shorts).\n"
            "- Order book depth indicates liquidity and slippage risk.\n"
            "- Open interest shows market positioning and potential squeeze points.\n"
            "- Correlation risk: many crypto assets move together.\n\n"
            "EVALUATION CRITERIA:\n"
            "- Prefer contracts with positive expected returns after funding costs.\n"
            "- Consider momentum, volatility, and liquidity.\n"
            "- Diversify across different crypto assets when possible.\n"
            "- Monitor funding rates for carry trade opportunities.\n\n"
            "PORTFOLIO PRINCIPLES:\n"
            "- Diversify across major cryptocurrencies when favorable.\n"
            "- Consider market momentum and technical patterns.\n"
            "- Balance between high-beta and stable contracts.\n"
            "- Account for funding rate impacts on carry.\n"
            "- Total allocation must equal 1.0.\n"
            "- CASH is a valid asset for capital preservation.\n\n"
            f"AVAILABLE CONTRACTS: {contract_list_str}, CASH\n\n"
            "CRITICAL: Return ONLY valid JSON. No extra text.\n\n"
            "REQUIRED JSON FORMAT:\n"
            "{\n"
            ' "reasoning": "Your detailed analysis here",\n'
            ' "allocations": {\n'
            f'   "{sample[0]}": <weight>,\n'
            f'   "{sample[1]}": <weight>,\n'
            f'   "{sample[2]}": <weight>,\n'
            '   "CASH": <weight>\n'
            " }\n"
            "}\n"
            "Where <weight> is a float between 0.0 and 1.0, and all weights sum to 1.0.\n\n"
            "RULES:\n"
            "1. Return ONLY the JSON object.\n"
            "2. Allocations must sum to 1.0.\n"
            "3. Consider funding rates when allocating.\n"
            "4. CASH allocation should reflect crypto market risk.\n"
            "5. Use double quotes for strings.\n"
            "6. No trailing commas.\n"
            "7. No extra text outside the JSON.\n"
            "Your objective is to maximize returns while managing crypto-specific risks including funding costs and 24/7 volatility."
        )


def create_bitmex_agent(
    name: str, model_name: str = "gpt-4o-mini"
) -> LLMBitMEXAgent:
    """
    Create a new BitMEX trading agent.

    Args:
        name: Agent display name
        model_name: LLM model identifier

    Returns:
        Initialized LLMBitMEXAgent instance
    """
    return LLMBitMEXAgent(name, model_name)
