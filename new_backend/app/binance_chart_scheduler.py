"""
Binance Chart Scheduler

주기적으로 Binance API를 호출하여 실시간 가격 데이터를 업데이트합니다.
"""
import sys
from typing import List
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from loguru import logger

from .binance_chart_data import BinanceChartDataManager

# loguru 설정: 컬러 출력 활성화 및 형식 설정
logger.remove()  # 기본 핸들러 제거
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True,
    backtrace=True,
    diagnose=True
)


class BinanceChartScheduler:
    """Binance 차트 데이터 스케줄러"""
    
    def __init__(self, update_interval: int = 30):
        """
        Args:
            update_interval: 업데이트 간격 (초)
        """
        self.update_interval = update_interval
        self.manager = BinanceChartDataManager()
        self.scheduler: BackgroundScheduler = None
        
    def start(self):
        """스케줄러를 시작합니다."""
        if self.scheduler is not None and self.scheduler.running:
            logger.warning("Scheduler is already running")
            return
        
        # 초기 히스토리 데이터 로드
        logger.info("Loading initial historical data...")
        self.manager.load_historical_data()[0]  # 튜플 반환값 무시
        
        # 스케줄러 생성
        executors = {"default": ThreadPoolExecutor(max_workers=2)}
        self.scheduler = BackgroundScheduler(executors=executors)
        
        # 주기적 업데이트 작업 추가
        self.scheduler.add_job(
            self._update_all_agents,
            "interval",
            seconds=self.update_interval,
            id="update_binance_chart_data",
            replace_existing=True,
        )
        
        self.scheduler.start()
        logger.info(f"Binance chart scheduler started (interval: {self.update_interval}s)")
    
    def stop(self):
        """스케줄러를 중지합니다."""
        if self.scheduler is not None and self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Binance chart scheduler stopped")
    
    def _update_all_agents(self):
        """모든 에이전트의 실시간 데이터를 업데이트합니다."""
        try:
            # 각 에이전트별로 업데이트
            # update_realtime_price_data 내부에서 load_historical_data를 호출하므로
            # 여기서는 중복 호출하지 않음
            for agent_name in self.manager.historical_data.keys():
                try:
                    # account 정보에서 심볼 가져오기
                    account_info = self.manager.get_account_info_from_data(agent_name)
                    symbols = list(account_info.keys())
                    
                    if not symbols:
                        logger.debug(f"No symbols found for agent {agent_name}")
                        continue
                    
                    # 실시간 데이터 업데이트 (내부에서 항상 새 파일 확인 및 로드 수행)
                    updated_df = self.manager.update_realtime_price_data(
                        agent_name, 
                        symbols
                    )
                    logger.debug(
                        f"Updated {len(updated_df)} rows for agent {agent_name} "
                        f"({len(symbols)} symbols)"
                    )
                except Exception as e:
                    logger.error(f"Failed to update agent {agent_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to update all agents: {e}")



