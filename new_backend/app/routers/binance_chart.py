"""
Binance Chart API Router

차트 데이터를 제공하는 API 엔드포인트입니다.
"""
import sys
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException
from loguru import logger
import polars as pl

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


@router.get("/binance/chart/snapshot/{close_time}")
def get_snapshot_at_time(close_time: int):
    """
    특정 시점(close_time)의 모든 에이전트의 allocation과 가격 정보를 반환합니다.
    
    Args:
        close_time: 밀리초 단위의 close_time
    """
    manager = get_chart_data_manager()
    manager.load_historical_data()[0]  # 튜플 반환값 무시
    
    snapshot_data: Dict[str, Dict[str, Any]] = {}
    
    for agent_name in manager.historical_data.keys():
        try:
            # 히스토리 데이터에서 해당 close_time의 데이터 찾기
            historical_df = manager.historical_data[agent_name]
            if historical_df.is_empty():
                continue
            
            # 실시간 데이터도 로드
            realtime_df = manager.load_realtime_data(agent_name)
            
            # 히스토리와 실시간 데이터 합치기
            if realtime_df.is_empty():
                combined_df = historical_df
            else:
                combined_df = pl.concat([historical_df, realtime_df], how="vertical", rechunk=True)
            
            # 해당 close_time의 데이터 필터링 (정확히 일치하지 않으면 가장 가까운 시간 찾기)
            snapshot_df = combined_df.filter(pl.col("close_time") == close_time)
            
            # 정확히 일치하는 데이터가 없으면 가장 가까운 close_time 찾기
            if snapshot_df.is_empty():
                # 모든 close_time과의 차이 계산
                time_diffs = combined_df.with_columns([
                    (pl.col("close_time") - close_time).abs().alias("time_diff")
                ])
                # 가장 가까운 close_time 찾기
                closest_df = time_diffs.sort("time_diff").head(1)
                if not closest_df.is_empty():
                    closest_row = closest_df.row(0, named=True)
                    time_diff = closest_row.get("time_diff")
                    closest_close_time = closest_row.get("close_time")
                    # 5분(300000ms) 이내의 차이면 해당 시점으로 간주
                    if time_diff is not None and closest_close_time is not None and time_diff < 300000:
                        snapshot_df = combined_df.filter(pl.col("close_time") == closest_close_time)
                    else:
                        continue
                else:
                    continue
            
            if snapshot_df.is_empty():
                continue
            
            # 각 심볼별 정보 추출
            symbols_data: Dict[str, Dict[str, float]] = {}
            for row in snapshot_df.iter_rows(named=True):
                symbol = row["symbol"]
                symbols_data[symbol] = {
                    "allocation": float(row.get("allocation", 0.0)),
                    "current_price": float(row.get("current_price", 0.0)),
                    "quantity": float(row.get("quantity", 0.0)),
                    "position_value": float(row.get("position_value", 0.0)),
                    "average_price": float(row.get("average_price", 0.0)),
                }
            
            # timestamp 가져오기
            timestamp = snapshot_df["timestamp"].first()
            if timestamp is not None:
                timestamp_str = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
            else:
                timestamp_str = None
            
            # last_trade_time 가져오기
            last_trade_time = None
            if "last_trade_time" in snapshot_df.columns:
                last_trade_time_value = snapshot_df["last_trade_time"].first()
                if last_trade_time_value is not None:
                    if hasattr(last_trade_time_value, "isoformat"):
                        last_trade_time = last_trade_time_value.isoformat()
                    else:
                        last_trade_time = str(last_trade_time_value)
            
            # last_trade_time이 없으면 manager의 메모리에서 가져오기
            if last_trade_time is None:
                last_trade_time_dt = manager.last_trade_times.get(agent_name)
                if last_trade_time_dt:
                    last_trade_time = last_trade_time_dt.isoformat()
                else:
                    # 히스토리 데이터에서 최신 timestamp를 last_trade_time으로 사용
                    if not historical_df.is_empty() and "timestamp" in historical_df.columns:
                        latest_timestamp = historical_df["timestamp"].max()
                        if latest_timestamp:
                            last_trade_time = latest_timestamp.isoformat() if hasattr(latest_timestamp, "isoformat") else str(latest_timestamp)
            
            # last_trade_time이 여전히 없으면 UTC 00:05로 설정
            if last_trade_time is None:
                from datetime import datetime, timezone
                # close_time에서 날짜 추출
                try:
                    date_obj = datetime.fromtimestamp(close_time / 1000, tz=timezone.utc).date()
                except Exception:
                    date_obj = datetime.now(timezone.utc).date()
                default_trade_time = datetime.combine(date_obj, datetime.min.time().replace(hour=0, minute=5), timezone.utc)
                last_trade_time = default_trade_time.isoformat()
            
            snapshot_data[agent_name] = {
                "timestamp": timestamp_str,
                "close_time": close_time,
                "last_trade_time": last_trade_time,
                "symbols": symbols_data,
            }
        except Exception as e:
            logger.error(f"Failed to get snapshot for agent {agent_name} at close_time {close_time}: {e}")
            continue
    
    return {
        "close_time": close_time,
        "snapshot": snapshot_data
    }


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