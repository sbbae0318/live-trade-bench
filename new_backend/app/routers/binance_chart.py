"""
Binance Chart API Router

차트 데이터를 제공하는 API 엔드포인트입니다.
"""
import sys
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from loguru import logger

from ..binance_chart_data import BinanceChartDataManager

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

router = APIRouter()

# 전역 데이터 매니저 인스턴스
_chart_data_manager: Optional[BinanceChartDataManager] = None


def get_chart_data_manager() -> BinanceChartDataManager:
    """차트 데이터 매니저 싱글톤 인스턴스를 반환합니다."""
    global _chart_data_manager
    if _chart_data_manager is None:
        _chart_data_manager = BinanceChartDataManager()
        # 초기 히스토리 데이터 로드
        _chart_data_manager.load_historical_data()[0]  # 튜플 반환값 무시
    return _chart_data_manager


@router.get("/binance/chart/agents")
def get_agents():
    """사용 가능한 모든 에이전트 목록을 반환합니다."""
    manager = get_chart_data_manager()
    manager.load_historical_data()[0]  # 튜플 반환값 무시
    
    agents = list(manager.historical_data.keys())
    return {"agents": agents}


@router.get("/binance/chart/{agent_name}/symbols")
def get_symbols(agent_name: str):
    """특정 에이전트의 심볼 목록을 반환합니다."""
    manager = get_chart_data_manager()
    manager.load_historical_data()[0]  # 튜플 반환값 무시
    
    if agent_name not in manager.historical_data:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    df = manager.historical_data[agent_name]
    symbols = df["symbol"].unique().to_list()
    
    return {"agent": agent_name, "symbols": symbols}


@router.get("/binance/chart/{agent_name}/account")
def get_account_info(agent_name: str):
    """특정 에이전트의 account 정보를 반환합니다."""
    manager = get_chart_data_manager()
    manager.load_historical_data()[0]  # 튜플 반환값 무시
    
    account_info = manager.get_account_info_from_data(agent_name)
    if not account_info:
        raise HTTPException(status_code=404, detail=f"Account info for '{agent_name}' not found")
    
    return {"agent": agent_name, "account": account_info}


@router.get("/binance/chart/{agent_name}/data")
def get_chart_data(
    agent_name: str,
    symbols: Optional[str] = None,
    include_realtime: bool = True
):
    """
    특정 에이전트의 차트 데이터를 반환합니다.
    
    Args:
        agent_name: 에이전트 이름
        symbols: 쉼표로 구분된 심볼 리스트 (예: "BTCUSDT,ETHUSDT")
        include_realtime: 실시간 데이터 포함 여부
    """
    manager = get_chart_data_manager()
    manager.load_historical_data()[0]  # 튜플 반환값 무시
    
    if agent_name not in manager.historical_data:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    # 심볼 파싱
    symbol_list = None
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    
    # 실시간 데이터 포함 여부에 따라 처리
    if include_realtime:
        # 실시간 데이터도 로드
        realtime_df = manager.load_realtime_data(agent_name)
        if not realtime_df.is_empty():
            # 실시간 데이터를 historical_data에 병합
            if agent_name in manager.realtime_data:
                manager.realtime_data[agent_name] = realtime_df
            else:
                manager.realtime_data[agent_name] = realtime_df
    
    chart_data = manager.get_combined_chart_data(agent_name, symbol_list)
    
    return {
        "agent": agent_name,
        "symbols": list(chart_data.keys()),
        "data": chart_data
    }


@router.get("/binance/chart/{agent_name}/total-value")
def get_total_value_chart_data(agent_name: str):
    """
    특정 에이전트의 전체 total 가치 차트 데이터를 반환합니다.
    
    Args:
        agent_name: 에이전트 이름
    """
    manager = get_chart_data_manager()
    manager.load_historical_data()[0]  # 튜플 반환값 무시
    
    if agent_name not in manager.historical_data:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    # 실시간 데이터도 로드
    realtime_df = manager.load_realtime_data(agent_name)
    if not realtime_df.is_empty():
        manager.realtime_data[agent_name] = realtime_df
    
    total_value_data = manager.get_total_value_chart_data(agent_name)
    
    return {
        "agent": agent_name,
        "data": total_value_data
    }


@router.get("/binance/chart/portfolio-weights")
def get_all_portfolio_weights():
    """모든 에이전트의 포트폴리오 비중을 반환합니다."""
    manager = get_chart_data_manager()
    manager.load_historical_data()[0]  # 튜플 반환값 무시
    
    portfolio_weights: Dict[str, Dict[str, Any]] = {}
    
    for agent_name in manager.historical_data.keys():
        try:
            account_info = manager.get_account_info_from_data(agent_name)
            if account_info:
                # 각 심볼별 allocation과 position_value 계산
                total_value = sum(
                    info.get("position_value", 0.0) or 0.0 
                    for info in account_info.values()
                )
                
                portfolio_weights[agent_name] = {
                    "symbols": {},
                    "total_value": total_value
                }
                
                for symbol, info in account_info.items():
                    allocation = info.get("allocation", 0.0) or 0.0
                    position_value = info.get("position_value", 0.0) or 0.0
                    current_price = info.get("current_price", 0.0) or 0.0
                    quantity = info.get("quantity", 0.0) or 0.0
                    
                    # 실제 비중 계산 (position_value / total_value)
                    actual_weight = (position_value / total_value * 100) if total_value > 0 else 0.0
                    
                    portfolio_weights[agent_name]["symbols"][symbol] = {
                        "allocation": allocation * 100,  # 목표 비중 (%)
                        "actual_weight": actual_weight,  # 실제 비중 (%)
                        "position_value": position_value,
                        "current_price": current_price,
                        "quantity": quantity
                    }
        except Exception as e:
            logger.error(f"Failed to get portfolio weights for {agent_name}: {e}")
            continue
    
    return {"portfolio_weights": portfolio_weights}


@router.post("/binance/chart/{agent_name}/update")
def update_realtime_data(agent_name: str, symbols: Optional[str] = None):
    """
    특정 에이전트의 실시간 가격 데이터를 업데이트합니다.
    
    Args:
        agent_name: 에이전트 이름
        symbols: 쉼표로 구분된 심볼 리스트 (None이면 account 정보에서 가져옴)
    """
    manager = get_chart_data_manager()
    # update_realtime_price_data 내부에서 load_historical_data를 호출하므로 여기서는 호출하지 않음
    
    if agent_name not in manager.historical_data:
        # 히스토리 데이터가 없으면 한 번 로드
        manager.load_historical_data()[0]
    
    if agent_name not in manager.historical_data:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    # 심볼 리스트 결정
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    else:
        # account 정보에서 심볼 가져오기
        account_info = manager.get_account_info_from_data(agent_name)
        symbol_list = list(account_info.keys())
    
    if not symbol_list:
        raise HTTPException(status_code=400, detail="No symbols found")
    
    # 실시간 데이터 업데이트 (내부에서 항상 새 파일 확인 및 로드 수행)
    updated_df = manager.update_realtime_price_data(agent_name, symbol_list)
    
    return {
        "agent": agent_name,
        "symbols": symbol_list,
        "updated_rows": len(updated_df),
        "timestamp": updated_df["timestamp"].max().isoformat() if not updated_df.is_empty() else None
    }