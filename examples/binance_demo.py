#!/usr/bin/env python3
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import sys
import time
from dotenv import load_dotenv
from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from live_trade_bench.systems import BinancePortfolioSystem


# 현 구조 문제점
# 1. 백테스트시 뉴스 문제 : 1시간마다 trading시 + 백테스트일경우 new data가 leak됨 (뉴스 정확한 시간을 가져오기 어려움)
# 2. 뉴스 페치 최적화 필요:
# 3. s3 파일 fetch 최적화 필요:
# 4. 검증용으로 프롬프트 저장 구조 필요:
# 5. Recent 10 allocation이 1시간 단위라서 너무 세세한데, recent 10의 경우 1일치 해상도로 주는게 나을지

# loguru logger is imported as `logger`

def main() -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} {level} {name} {file}:{line} - {message}")
    # Load environment variables from .env if present
    load_dotenv()

    # Create Binance trading system
    system = BinancePortfolioSystem()

    # Add AI agent
    system.add_agent("Binance_gpt-4o-mini", initial_cash=10000.0, model_name="openai/gpt-4o-mini")

    # Initialize system (fetches trending contracts from S3-based dataset)
    system.initialize_for_live()
    print(f"Trading {len(system.universe)} contracts: {system.universe}...")

    # Run trading cycles
    for i in range(5):
        system.run_cycle()

    
    print("Demo finished.")


def demo_test() -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} {level} {name} {file}:{line} - {message}")
    # Load environment variables from .env if present
    load_dotenv()

    # Create Binance trading system
    time_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    system = BinancePortfolioSystem(name="demo_" + time_str)

    # Add AI agent
    system.add_agent("GPT-4o-mini", initial_cash=10000.0, model_name="openai/gpt-4o-mini")
    system.add_agent("GPT-4.1", initial_cash=10000.0, model_name="openai/gpt-4.1")
    system.add_agent("GPT-5-mini", initial_cash=10000.0, model_name="openai/gpt-5-mini-2025-08-07")
    system.add_agent("GPT-o3", initial_cash=10000.0, model_name="openai/o3-2025-04-16")

    # Initialize system (fetches trending contracts from S3-based dataset)
    system.initialize_for_live()
    print(f"Trading {len(system.universe)} contracts: {system.universe}...")

    start_wall = datetime.now(timezone.utc)
    start_perf = time.perf_counter()

    date_str: str = datetime(2025, 11, 25, 1, 0, 0, tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    cycles= 24 * 7 + 1
    delt = timedelta(hours=1)

    cycles = 7
    delt = timedelta(days=1)

    # Run trading cycles
    for i in range(cycles):
        logger.info(f"Running cycle {i} for {date_str}")
        system.run_cycle(for_date=date_str)
        date_str = (datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S UTC") + delt).strftime("%Y-%m-%d %H:%M:%S UTC")

    end_wall = datetime.now(timezone.utc)
    elapsed_s = time.perf_counter() - start_perf
    logger.info(f"Run start (UTC): {start_wall.isoformat()}")
    logger.info(f"Run end   (UTC): {end_wall.isoformat()}")
    logger.info(f"Total elapsed   : {elapsed_s:.2f}s")

    system.profiler.print_profile(logger)
    print("Demo finished.")


def live_test() -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} {level} {name} {file}:{line} - {message}")
    # Load environment variables from .env if present
    load_dotenv()

    # Create Binance trading system
    time_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    system = BinancePortfolioSystem(name="live_" + time_str)

    # Add AI agent
    system.add_agent("GPT-4o-mini", initial_cash=10000.0, model_name="openai/gpt-4o-mini")
    system.add_agent("GPT-4.1", initial_cash=10000.0, model_name="openai/gpt-4.1")
    system.add_agent("GPT-5-mini", initial_cash=10000.0, model_name="openai/gpt-5-mini-2025-08-07")
    system.add_agent("GPT-o3", initial_cash=10000.0, model_name="openai/o3-2025-04-16")

    # Initialize system (fetches trending contracts from S3-based dataset)
    system.initialize_for_live()
    print(f"Trading {len(system.universe)} contracts: {system.universe}...")

    date = datetime(2025, 11, 25, 0, 0, 0, tzinfo=timezone.utc)
    date_end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    delt = timedelta(days=1)

    # Run trading cycles
    while date < date_end:
        date_str = date.strftime("%Y-%m-%d %H:%M:%S UTC")
        system.run_cycle(for_date=date_str)
        date = date + delt

    # Schedule daily run_cycle at UTC 00:01 using APScheduler
    def run_daily_cycle() -> None:
        """Execute run_cycle at scheduled time"""
        run_time = datetime.now(timezone.utc)
        date_str = run_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        logger.info(f"Executing run_cycle at {date_str}")
        system.run_cycle(for_date=date_str)

    scheduler = BlockingScheduler(timezone=timezone.utc)
    scheduler.add_job(
        run_daily_cycle,
        trigger=CronTrigger(hour=0, minute=5, timezone=timezone.utc),
        id="daily_run_cycle",
        name="Daily run_cycle at UTC 00:05"
    )
    
    logger.info("Starting daily scheduler for UTC 00:01")
    scheduler.start()


if __name__ == "__main__":
    live_test()