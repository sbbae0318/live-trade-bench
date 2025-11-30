import json
import logging
import os
import shutil

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from live_trade_bench.systems import BinancePortfolioSystem
from backend.app.price_data import update_binance_prices_and_values
from .news_data import update_news_data

from backend.app.config import (
    ALLOWED_ORIGINS,
    MODELS_DATA_FILE,
    MODELS_DATA_HIST_FILE,
    MODELS_DATA_INIT_FILE,
    get_base_model_configs,
    get_test_model_configs,
)
from backend.app.config import NEWS_DATA_FILE, SOCIAL_DATA_FILE, SYSTEM_DATA_FILE
from backend.app.routers.router_utils import read_json_or_404, slice_limit
from backend.app.counter_data import get_visit_count, increment_visit_count
from typing import Any, Dict, List, Optional

load_dotenv()

# Global system instance
binance_system = BinancePortfolioSystem.get_instance()
scheduler = None

# Add agents (paper trading default capital similar to BitMEX)
#for display_name, model_id in get_base_model_configs():
#    binance_system.add_agent(display_name, 1000.0, model_id)
for display_name, model_id in get_test_model_configs():
    binance_system.add_agent(display_name, 1000.0, model_id)

# Initialize system for live trading
binance_system.initialize_for_live()


def get_binance_system():
    """Get the Binance system instance."""
    global binance_system
    return binance_system


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Live Trade Bench - Binance Testbed",
    description="Binance-only testbed API for Live Trade Bench",
    version="1.0.0",
)

allowed_origins = list(ALLOWED_ORIGINS)
frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api")
async def api_root():
    """API information endpoint."""
    return {
        "message": "Live Trade Bench - Binance Testbed",
        "version": "1.0.0",
        "endpoints": {
            "models": "/api/models",
            "news": "/api/news",
            "social": "/api/social",
            "system": "/api/system",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }

@app.get("/api/models", include_in_schema=False)
@app.get("/api/models/")
def get_models() -> List[Dict[str, Any]]:
    logger.info(f"Getting models data from {MODELS_DATA_FILE}")
    return read_json_or_404(MODELS_DATA_FILE)


@app.get("/api/news/{market_type}")
def get_news(market_type: str, limit: Optional[int] = 100) -> List[Dict[str, Any]]:
    # Binance 전용. 그 외 요청은 빈 리스트 반환
    data = read_json_or_404(NEWS_DATA_FILE)
    news_items = slice_limit(data.get("bitmex", []), limit, 100, 500)
    return news_items


@app.get("/api/social/{market_type}")
def get_social(market_type: str, limit: Optional[int] = 100) -> List[Dict[str, Any]]:
    # Binance 전용. 그 외 요청은 빈 리스트 반환
    data = read_json_or_404(SOCIAL_DATA_FILE)
    social_items = slice_limit(data.get("bitmex", []), limit, 100, 500)
    return social_items


@app.get("/api/system")
def get_system_status() -> Dict[str, Any]:
    return read_json_or_404(SYSTEM_DATA_FILE)


@app.get("/api/views")
def get_views() -> Dict[str, int]:
    return {"views": get_visit_count()}


@app.post("/api/views")
def post_views() -> Dict[str, int]:
    return {"views": increment_visit_count()}


# bitmex 항목이 없어서 빈 함수로 처리.
@app.get("/api/schedule/next-price-update")
def get_next_price_update() -> Dict[str, Optional[str]]:
    """Expose the next scheduled realtime price update."""
    response = {}
    #stock_time = get_next_price_update_time("stock")
    #poly_time = get_next_price_update_time("polymarket")

    #response = {
    #    "stock": stock_time.isoformat() if stock_time else None,
    #    "polymarket": poly_time.isoformat() if poly_time else None,
    #}

    ## Backward compatibility for older clients expecting single field
    #response["next_run_time"] = response["stock"]

    return response



def load_backtest_as_initial_data():
    """Load backtest data as initial trading data if no live data exists."""
    if not os.path.exists(MODELS_DATA_FILE) and os.path.exists(MODELS_DATA_INIT_FILE):
        try:
            from .models_data import _create_compact_model_data

            with open(MODELS_DATA_INIT_FILE, "r") as f:
                init_data = json.load(f)

            # Copy full data to hist file
            shutil.copy(MODELS_DATA_INIT_FILE, MODELS_DATA_HIST_FILE)
            os.chmod(MODELS_DATA_HIST_FILE, 0o644)
            logger.info("📚 Created complete historical data file")

            compact_data = [_create_compact_model_data(model) for model in init_data]
            with open(MODELS_DATA_FILE, "w") as f:
                json.dump(compact_data, f, indent=4)
            os.chmod(MODELS_DATA_FILE, 0o644)

            logger.info(
                "📊 Loaded backtest data as initial trading data (compact frontend + full hist)"
            )
        except Exception as e:
            logger.error(f"❌ Failed to load backtest data: {e}")


def safe_generate_binance_cycle():
    """Run Binance trading cycle (24/7 crypto markets)."""
    logger.info("🔄 Running Binance cycle...")
    try:
        binance_system.run_cycle()
        logger.info("✅ Binance cycle completed")
    except Exception as e:
        logger.error(f"❌ Binance cycle failed: {e}")


def schedule_background_tasks(scheduler: BackgroundScheduler):
    binance_price_update_interval = 900  # 15 minutes
    binance_trade_interval = 3600  # 60 minutes
    
    logger.info(
        f"📈 Scheduled Binance trading cycle for every {binance_trade_interval} seconds ({binance_trade_interval//60} minutes)"
    )
    scheduler.add_job(
        safe_generate_binance_cycle,
        "interval",
        seconds=binance_trade_interval,
        id="binance_trading_cycle",
        replace_existing=True,
    )
    scheduler.add_job(
        update_binance_prices_and_values,
        "interval",
        seconds=binance_price_update_interval,
        id="update_binance_prices",
        replace_existing=True,
    )


@app.on_event("startup")
def startup_event():
    logger.info("🚀 FastAPI app (Binance Testbed) starting up...")

    # Ensure initial models data exists before any scheduled jobs run
    load_backtest_as_initial_data()

    global scheduler
    executors = {"default": ThreadPoolExecutor(max_workers=2)}
    scheduler = BackgroundScheduler(executors=executors)
    schedule_background_tasks(scheduler)
    scheduler.start()

    logger.info("✅ Background scheduler started.")
    logger.info("🚀 Binance Testbed startup completed")

    # Update only crypto news for the Binance testbed (inject system to avoid circular imports)
    update_news_data(stock=False, polymarket=False, crypto=True, crypto_system=binance_system)

    safe_generate_binance_cycle()
    update_binance_prices_and_values()


@app.get("/health")
def health_check():
    return {"status": "ok"}


static_files_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "frontend", "build", "static"
)
if os.path.exists(static_files_path):
    app.mount(
        "/static",
        StaticFiles(directory=static_files_path),
        name="static",
    )


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    index_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "frontend", "build", "index.html"
    )

    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=404,
            detail="Frontend not found. Make sure to build the frontend first.",
        )

    return FileResponse(index_path)


