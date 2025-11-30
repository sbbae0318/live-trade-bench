#!/usr/bin/env python3
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import logging
import sys
import time
from dotenv import load_dotenv
from live_trade_bench.systems import BinancePortfolioSystem


# 현 구조 문제점
# 1. 백테스트시 뉴스 문제 : 1시간마다 trading시 + 백테스트일경우 new data가 leak됨 (뉴스 정확한 시간을 가져오기 어려움)
# 2. 뉴스 페치 최적화 필요:
# 3. s3 파일 fetch 최적화 필요:
# 4. 검증용으로 프롬프트 저장 구조 필요:
# 5. Recent 10 allocation이 1시간 단위라서 너무 세세한데, recent 10의 경우 1일치 해상도로 주는게 나을지

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Load environment variables from .env if present
    load_dotenv()

    # Create Binance trading system
    system = BinancePortfolioSystem(name="demo")

    # Add AI agent
    system.add_agent("GPT-4o-mini", initial_cash=10000.0, model_name="openai/gpt-4o-mini")
    system.add_agent("GPT-4.1", initial_cash=10000.0, model_name="openai/gpt-4.1")
    system.add_agent("GPT-5", initial_cash=10000.0, model_name="openai/gpt-5")
    system.add_agent("GPT-o3", initial_cash=10000.0, model_name="openai/o3-2025-04-16")

    # Initialize system (fetches trending contracts from S3-based dataset)
    system.initialize_for_live()
    print(f"Trading {len(system.universe)} contracts: {system.universe}...")

    start_wall = datetime.now(timezone.utc)
    start_perf = time.perf_counter()

    date_str: str = datetime(2025, 11, 25, 0, 0, 0, tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cycles=50
    # Run trading cycles
    for i in range(cycles):
        logging.info(f"Running cycle {i} for {date_str}")
        system.run_cycle(for_date=date_str)
        date_str = (datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S UTC") + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S UTC")

    end_wall = datetime.now(timezone.utc)
    elapsed_s = time.perf_counter() - start_perf
    logging.info(f"Run start (UTC): {start_wall.isoformat()}")
    logging.info(f"Run end   (UTC): {end_wall.isoformat()}")
    logging.info(f"Total elapsed   : {elapsed_s:.2f}s")

    print("Demo finished.")


if __name__ == "__main__":
    #main()
    demo_test()