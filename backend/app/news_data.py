import json
from typing import Any, Optional
from .config import NEWS_DATA_FILE


def update_news_data(
    stock: bool = True,
    polymarket: bool = True,
    crypto: bool = True,
    *,
    stock_system: Optional[Any] = None,
    polymarket_system: Optional[Any] = None,
    crypto_system: Optional[Any] = None,
) -> None:
    """
    Update news data selectively based on market flags.
    - stock: update stock news
    - polymarket: update polymarket news
    - crypto: update crypto (bitmex/binance-backed) news
    """
    print("📰 Updating news data (flags:", {"stock": stock, "polymarket": polymarket, "crypto": crypto}, ")")

    all_news_data = {"stock": [], "polymarket": [], "bitmex": []}

    try:
        # Fetch systems per flag to avoid import errors when not needed
        bitmex_like_system = None  # could be BitMEX or Binance-backed (Binance testbed)

        if stock and stock_system is None:
            try:
                from .main import get_stock_system  # type: ignore
                stock_system = get_stock_system()
            except Exception:
                print("⚠️ Stock system not available; skipping stock news.")

        if polymarket and polymarket_system is None:
            try:
                from .main import get_polymarket_system  # type: ignore
                polymarket_system = get_polymarket_system()
            except Exception:
                print("⚠️ Polymarket system not available; skipping polymarket news.")

        if crypto:
            if crypto_system is not None:
                bitmex_like_system = crypto_system
            else:
                # Prefer BitMEX from main; if not available, skip (Binance testbed should pass crypto_system)
                try:
                    from .main import get_bitmex_system  # type: ignore
                    bitmex_like_system = get_bitmex_system()
                except Exception:
                    print("⚠️ Crypto system not provided and BitMEX unavailable; skipping crypto news.")
            
        if not any([stock_system, polymarket_system, bitmex_like_system]):
            print("❌ No systems available for requested flags")
            return

        # Initialize systems if not already done
        if stock_system and not getattr(stock_system, "universe", None):
            stock_system.initialize_for_live()
        if polymarket_system and not getattr(polymarket_system, "universe", None):
            polymarket_system.initialize_for_live()
        if bitmex_like_system and not getattr(bitmex_like_system, "universe", None):
            bitmex_like_system.initialize_for_live()

        # Fetch market data
        stock_market_data = stock_system._fetch_market_data(for_date=None) if stock_system else {}
        polymarket_market_data = polymarket_system._fetch_market_data(for_date=None) if polymarket_system else {}
        bitmex_market_data = bitmex_like_system._fetch_market_data(for_date=None) if bitmex_like_system else {}

        # Fetch news data
        stock_news = stock_system._fetch_news_data(stock_market_data, for_date=None) if stock_system else {}
        polymarket_news = (
            polymarket_system._fetch_news_data(polymarket_market_data, for_date=None)
            if polymarket_system
            else {}
        )
        bitmex_news = (
            bitmex_like_system._fetch_news_data(bitmex_market_data, for_date=None)
            if bitmex_like_system
            else {}
        )

        if stock_system:
            all_news_data["stock"] = [item for sublist in stock_news.values() for item in sublist]
        if polymarket_system:
            all_news_data["polymarket"] = [item for sublist in polymarket_news.values() for item in sublist]
        if bitmex_like_system:
            all_news_data["bitmex"] = [item for sublist in bitmex_news.values() for item in sublist]

        with open(NEWS_DATA_FILE, "w") as f:
            json.dump(all_news_data, f, indent=4)
        print(f"✅ News data updated and saved to {NEWS_DATA_FILE}")

    except Exception as e:
        print(f"❌ Error updating news data: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    update_news_data()
