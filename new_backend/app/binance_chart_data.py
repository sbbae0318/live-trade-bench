"""
Binance Chart Data Manager

주기적으로 parquet 파일을 읽어서 히스토리 데이터를 로드하고,
Binance API를 통해 실시간 가격을 가져와서 차트 데이터를 생성합니다.
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import polars as pl
import requests
from loguru import logger

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


class BinanceChartDataManager:
    """Binance 차트 데이터 관리 클래스"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Args:
            data_dir: parquet 파일이 저장된 디렉토리 경로
                     None이면 BASE_DIR 환경변수 또는 기본 경로 사용
        """
        if data_dir is None:
            base_dir = os.getenv("BASE_DIR", os.getcwd())
            self.data_dir = Path(base_dir) / "binance"
        else:
            self.data_dir = Path(data_dir)
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 실시간 가격 데이터를 저장할 디렉토리
        self.realtime_dir = self.data_dir / "realtime"
        self.realtime_dir.mkdir(parents=True, exist_ok=True)
        
        # 히스토리 데이터 캐시
        self.historical_data: Dict[str, pl.DataFrame] = {}
        self.realtime_data: Dict[str, pl.DataFrame] = {}
        
        # 마지막으로 로드한 파일의 timestamp (YYYY-MM-DDTHHMMSS 형식)
        self.last_loaded_timestamp: Optional[str] = None
        
        # 마지막 거래 파일의 최신 close_time (밀리초, 실시간 데이터 리셋 기준점)
        self.last_trade_close_time_ms: Optional[int] = None
        
        # Binance API 기본 URL
        self.binance_api_url = "https://fapi.binance.com"
        
    def load_historical_data(self, for_date: Optional[str] = None) -> tuple[Dict[str, pl.DataFrame], bool]:
        """
        지정된 디렉토리에서 parquet 파일을 읽어서 히스토리 데이터를 로드합니다.
        처음 호출 시에는 모든 파일을 읽고, 이후에는 마지막으로 읽은 파일 이후의 파일만 읽습니다.
        
        Args:
            for_date: 사용되지 않음 (하위 호환성을 위해 유지)
            
        Returns:
            (agent별로 그룹화된 DataFrame 딕셔너리, 새 파일이 로드되었는지 여부)
        """
        logger.info(f"Loading historical data from {self.data_dir}")
        
        # 모든 하위 디렉토리에서 parquet 파일 재귀적으로 찾기
        all_parquet_files: List[Path] = []
        
        # 재귀적으로 모든 .parquet 파일 찾기
        for parquet_file in self.data_dir.rglob("*.parquet"):
            # realtime 디렉토리는 제외 (실시간 데이터는 별도로 관리)
            if "realtime" in parquet_file.parts:
                continue
            all_parquet_files.append(parquet_file)
        
        if not all_parquet_files:
            logger.warning(f"No parquet files found in {self.data_dir}")
            return {}, False
        
        # 파일명에서 timestamp 추출하여 필터링
        parquet_files: List[Path] = []
        is_initial_load = self.last_loaded_timestamp is None
        if is_initial_load:
            # 처음 로드: 모든 파일 읽기
            parquet_files = all_parquet_files
            logger.info(f"Initial load: Found {len(parquet_files)} parquet files to load")
        else:
            # 이후 로드: 마지막 timestamp 이후의 파일만 읽기
            try:
                # 마지막 timestamp를 datetime으로 변환
                last_dt = datetime.strptime(self.last_loaded_timestamp, "%Y-%m-%dT%H%M%S")
                last_timestamp_str = last_dt.strftime("%Y-%m-%dT%H%M%S")
                
                for file_path in all_parquet_files:
                    # 파일명에서 timestamp 추출 (예: 2025-12-13T074350.parquet -> 2025-12-13T074350)
                    file_stem = file_path.stem  # 확장자 제거
                    
                    # 파일명이 YYYY-MM-DDTHHMMSS 형식인지 확인
                    if len(file_stem) == 17 and "T" in file_stem:
                        try:
                            file_timestamp = datetime.strptime(file_stem, "%Y-%m-%dT%H%M%S")
                            file_timestamp_str = file_timestamp.strftime("%Y-%m-%dT%H%M%S")
                            
                            # 마지막 timestamp 이후의 파일만 포함
                            if file_timestamp_str > last_timestamp_str:
                                parquet_files.append(file_path)
                        except ValueError:
                            # 형식이 맞지 않으면 포함 (하위 호환성)
                            parquet_files.append(file_path)
                    else:
                        # 형식이 맞지 않으면 포함 (하위 호환성)
                        parquet_files.append(file_path)
                
                logger.info(f"Incremental load: Found {len(parquet_files)} new parquet files since {self.last_loaded_timestamp}")
            except Exception as e:
                logger.warning(f"Failed to filter by timestamp, loading all files: {e}")
                parquet_files = all_parquet_files
        
        # 새 파일이 없으면 False 반환
        if not parquet_files:
            logger.debug("No new parquet files to load")
            return (self.historical_data if self.historical_data else {}), False
        
        # 모든 parquet 파일 읽기
        dfs: List[pl.DataFrame] = []
        for file_path in sorted(parquet_files):
            try:
                df = pl.read_parquet(file_path)
                dfs.append(df)
                logger.debug(f"Loaded {file_path.name}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        if not dfs:
            logger.warning("No valid parquet files loaded")
            return {}, False
        
        # 모든 데이터 합치기
        new_df = pl.concat(dfs, how="vertical", rechunk=True)
        
        # 기존 데이터와 합치기
        if self.historical_data:
            # 기존 데이터를 DataFrame 리스트로 변환
            existing_dfs = list(self.historical_data.values())
            combined_df = pl.concat([*existing_dfs, new_df], how="vertical", rechunk=True)
        else:
            combined_df = new_df
        
        # agent별로 그룹화
        agent_data: Dict[str, pl.DataFrame] = {}
        for agent_name in combined_df["agent"].unique().to_list():
            agent_df = combined_df.filter(pl.col("agent") == agent_name)
            agent_data[agent_name] = agent_df.sort("close_time")
        
        self.historical_data = agent_data
        
        # 마지막으로 로드한 파일의 timestamp 업데이트
        latest_timestamp = None
        for file_path in parquet_files:
            file_stem = file_path.stem
            if len(file_stem) == 17 and "T" in file_stem:
                try:
                    file_timestamp = datetime.strptime(file_stem, "%Y-%m-%dT%H%M%S")
                    file_timestamp_str = file_timestamp.strftime("%Y-%m-%dT%H%M%S")
                    if latest_timestamp is None or file_timestamp_str > latest_timestamp:
                        latest_timestamp = file_timestamp_str
                except ValueError:
                    continue
        
        # 마지막 timestamp 업데이트
        if latest_timestamp:
            self.last_loaded_timestamp = latest_timestamp
        
        # 새로 로드한 데이터의 최신 close_time 저장 (실시간 데이터 리셋 기준점)
        # 이미 로드한 new_df에서 최신 close_time을 가져옴 (파일을 다시 읽지 않음)
        if not new_df.is_empty() and "close_time" in new_df.columns:
            latest_close_time_ms = new_df["close_time"].max()
            self.last_trade_close_time_ms = latest_close_time_ms
            if latest_timestamp:
                logger.debug(f"Updated last_loaded_timestamp to {latest_timestamp}, last_trade_close_time_ms to {latest_close_time_ms}")
            else:
                logger.debug(f"Updated last_trade_close_time_ms to {latest_close_time_ms} (initial load)")
        elif latest_timestamp:
            logger.debug(f"Updated last_loaded_timestamp to {latest_timestamp} (no close_time column)")
        
        logger.info(f"Loaded historical data for {len(agent_data)} agents (new files: {len(parquet_files)})")
        return agent_data, True
    
    def get_current_price_from_binance(self, symbol: str) -> Optional[float]:
        """
        Binance API를 통해 현재 가격을 가져옵니다.
        
        Args:
            symbol: 심볼 (예: "BTCUSDT", "USDT")
            
        Returns:
            현재 가격 또는 None
        """
        # USDT는 현물이므로 futures API에서 조회할 수 없음
        # USDT는 항상 1.0으로 고정
        if symbol == "USDT" or symbol.upper() == "USDT":
            return 1.0
        
        try:
            url = f"{self.binance_api_url}/fapi/v1/ticker/price"
            params = {"symbol": symbol}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            return float(data["price"])
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None
    
    def get_account_info_from_data(self, agent_name: str) -> Dict[str, Any]:
        """
        히스토리 데이터에서 account 정보를 추출합니다.
        
        Args:
            agent_name: 에이전트 이름
            
        Returns:
            account 정보 딕셔너리 (symbol별 allocation, quantity 포함)
        """
        
        if agent_name not in self.historical_data:
            return {}
        
        df = self.historical_data[agent_name]
        if df.is_empty():
            return {}
        
        # 최신 close_time 찾기
        latest_close_time = df["close_time"].max()
        
        # 최신 close_time의 모든 행 가져오기 (각 심볼별로)
        latest_df = df.filter(pl.col("close_time") == latest_close_time)
        
        
        account_info: Dict[str, Dict[str, float]] = {}
        for row in latest_df.iter_rows(named=True):
            symbol = row["symbol"]
            quantity = float(row.get("quantity", 0.0))
            current_price = float(row.get("current_price", 0.0))
            position_value = float(row.get("position_value", 0.0))
            # position_value가 없으면 quantity * current_price로 계산
            if position_value == 0.0 and quantity > 0 and current_price > 0:
                position_value = quantity * current_price
            account_info[symbol] = {
                "allocation": float(row.get("allocation", 0.0)),
                "quantity": quantity,
                "current_price": current_price,
                "position_value": position_value,
                "average_price": float(row.get("average_price", 0.0)),  # 구매 평균 가격
            }
        
        
        return account_info
    
    def update_realtime_price_data(
        self, 
        agent_name: str, 
        symbols: List[str]
    ) -> pl.DataFrame:
        """
        Binance API를 통해 실시간 가격을 가져와서 시계열 데이터를 업데이트합니다.
        항상 새 거래 파일을 확인하고, 새 파일이 생성되었으면 실시간 데이터를 리셋합니다.
        
        Args:
            agent_name: 에이전트 이름
            symbols: 심볼 리스트
            
        Returns:
            업데이트된 실시간 데이터 DataFrame
        """
        # 새 거래 파일 확인 및 로드 (항상 수행)
        agent_data, has_new_files = self.load_historical_data()
        if has_new_files:
            logger.info(f"New trade files detected for agent {agent_name}, resetting realtime data")
            # 새 거래 파일이 생성되었으면 실시간 데이터를 리셋
            # 마지막 거래 파일의 최신 close_time 이후의 데이터만 유지
            if agent_name in self.realtime_data and self.last_trade_close_time_ms is not None:
                existing_realtime = self.realtime_data[agent_name]
                if not existing_realtime.is_empty():
                    # 마지막 거래 close_time 이후의 실시간 데이터만 유지
                    filtered_realtime = existing_realtime.filter(
                        pl.col("close_time") > self.last_trade_close_time_ms
                    )
                    if filtered_realtime.is_empty():
                        # 필터링 후 데이터가 없으면 완전히 리셋
                        self.realtime_data[agent_name] = pl.DataFrame()
                        logger.debug(f"Reset all realtime data for agent {agent_name}")
                    else:
                        # 필터링된 데이터만 유지
                        self.realtime_data[agent_name] = filtered_realtime
                        logger.debug(
                            f"Filtered realtime data for agent {agent_name}: "
                            f"kept {len(filtered_realtime)} rows after close_time {self.last_trade_close_time_ms}"
                        )
            else:
                # 실시간 데이터가 없거나 기준점이 없으면 완전히 리셋
                self.realtime_data[agent_name] = pl.DataFrame()
                logger.debug(f"Reset all realtime data for agent {agent_name} (no existing data or no trade close_time)")
        
        now = datetime.now(timezone.utc)
        close_time_ms = int(now.timestamp() * 1000)
        
        # account 정보 가져오기 (최신 히스토리 데이터에서)
        account_info = self.get_account_info_from_data(agent_name)
        
        
        # 각 심볼별 가격 가져오기
        rows: List[Dict[str, Any]] = []
        for symbol in symbols:
            price = self.get_current_price_from_binance(symbol)
            if price is None:
                continue
            
            # account 정보에서 allocation, quantity, average_price 가져오기
            symbol_info = account_info.get(symbol, {})
            allocation = symbol_info.get("allocation", 0.0)
            quantity = symbol_info.get("quantity", 0.0)
            average_price = symbol_info.get("average_price", price)  # 히스토리 데이터의 구매 평균 가격 사용, 없으면 현재 가격 사용
            position_value = quantity * price  # 현재 가격 기준 포지션 가치
            
            
            rows.append({
                "timestamp": now,
                "close_time": close_time_ms,
                "agent": agent_name,
                "symbol": symbol,
                "quantity": quantity,
                "average_price": average_price,  # 히스토리 데이터의 구매 평균 가격 유지
                "current_price": price,  # 실시간 가격만 업데이트
                "position_value": position_value,
                "allocation": allocation,
            })
        
        if not rows:
            return pl.DataFrame()
        
        # 새로운 데이터를 DataFrame으로 변환
        new_df = pl.DataFrame(rows)
        
        
        # 기존 실시간 데이터에 추가
        if agent_name in self.realtime_data and not self.realtime_data[agent_name].is_empty():
            existing_df = self.realtime_data[agent_name]
            
            self.realtime_data[agent_name] = pl.concat(
                [existing_df, new_df],
                how="vertical",
                rechunk=True
            )
            
        else:
            self.realtime_data[agent_name] = new_df
        
        # 실시간 데이터를 CSV 파일로 저장
        self._save_realtime_data(agent_name)
        
        return new_df
    
    def _save_realtime_data(self, agent_name: str) -> None:
        """실시간 데이터를 CSV 파일로 날짜별로 append하여 저장합니다."""
        if agent_name not in self.realtime_data:
            return
        
        df = self.realtime_data[agent_name]
        if df.is_empty():
            return
        
        # 날짜별 디렉토리 생성
        today = datetime.now().date()
        date_dir = self.realtime_dir / today.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # 오늘 날짜의 데이터만 필터링
        if "timestamp" in df.columns:
            # timestamp를 datetime으로 변환 (아직 변환되지 않았을 경우)
            if df["timestamp"].dtype != pl.Datetime:
                try:
                    df = df.with_columns(
                        pl.col("timestamp").str.to_datetime(strict=False, time_zone="UTC")
                    )
                except Exception:
                    # 변환 실패 시 여러 형식 시도
                    formats = [
                        "%Y-%m-%d %H:%M:%S%.f %Z",
                        "%Y-%m-%dT%H:%M:%S%.f%z",
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S",
                    ]
                    for fmt in formats:
                        try:
                            df = df.with_columns(
                                pl.col("timestamp").str.strptime(pl.Datetime, format=fmt, strict=False, time_zone="UTC")
                            )
                            break
                        except Exception:
                            continue
            
            # 오늘 날짜의 데이터만 필터링
            df = df.filter(
                pl.col("timestamp").dt.date() == today
            )
            
            if df.is_empty():
                logger.debug(f"No data for today ({today}) to save for agent {agent_name}")
                return
        
        # CSV 파일 경로 (날짜별로 하나의 파일)
        csv_file_path = date_dir / f"{agent_name}.csv"
        
        # 기존 CSV 파일이 있으면 읽어서 중복 제거 후 append
        if csv_file_path.exists():
            try:
                existing_df = pl.read_csv(str(csv_file_path))
                # timestamp 컬럼을 datetime으로 변환 (CSV에서 읽을 때 문자열로 읽힘)
                if "timestamp" in existing_df.columns:
                    # 먼저 to_datetime을 시도 (자동 형식 감지)
                    try:
                        existing_df = existing_df.with_columns(
                            pl.col("timestamp").str.to_datetime(strict=False, time_zone="UTC")
                        )
                    except Exception:
                        # 실패하면 여러 형식을 순차적으로 시도
                        formats = [
                            "%Y-%m-%d %H:%M:%S%.f %Z",
                            "%Y-%m-%dT%H:%M:%S%.f%z",
                            "%Y-%m-%d %H:%M:%S",
                            "%Y-%m-%dT%H:%M:%S",
                        ]
                        converted = False
                        for fmt in formats:
                            try:
                                existing_df = existing_df.with_columns(
                                    pl.col("timestamp").str.strptime(pl.Datetime, format=fmt, strict=False, time_zone="UTC")
                                )
                                converted = True
                                break
                            except Exception:
                                continue
                        if not converted:
                            logger.warning(f"Failed to parse timestamp column in {csv_file_path}, keeping as string")
                # 기존 데이터와 새 데이터 합치기
                combined_df = pl.concat([existing_df, df], how="vertical", rechunk=True)
                # 중복 제거: 같은 close_time, agent, symbol 조합은 하나만 유지 (최신 timestamp 우선)
                combined_df = (
                    combined_df
                    .sort("timestamp", descending=True)
                    .unique(subset=["close_time", "agent", "symbol"], keep="first")
                    .sort("close_time")
                )
                df = combined_df
            except Exception as e:
                logger.warning(f"Failed to read existing CSV file {csv_file_path}, creating new file: {e}")
        
        try:
            # CSV 파일로 저장 (append 모드가 아니라 전체를 다시 쓰기)
            df.write_csv(str(csv_file_path))
            logger.debug(f"Saved realtime data for {agent_name} to {csv_file_path} ({len(df)} rows)")
            
        except Exception as e:
            logger.error(f"Failed to save realtime data for {agent_name}: {e}")
    
    def load_realtime_data(self, agent_name: str) -> pl.DataFrame:
        """저장된 실시간 데이터를 CSV 파일에서 로드합니다."""
        if not self.realtime_dir.exists():
            return pl.DataFrame()
        
        csv_files: List[Path] = []
        for date_dir in sorted(self.realtime_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            
            csv_file = date_dir / f"{agent_name}.csv"
            if csv_file.exists():
                csv_files.append(csv_file)
        
        if not csv_files:
            return pl.DataFrame()
        
        
        dfs: List[pl.DataFrame] = []
        file_row_counts = {}
        for csv_file_path in sorted(csv_files):
            try:
                df = pl.read_csv(str(csv_file_path))
                # timestamp 컬럼을 datetime으로 변환 (여러 형식 시도)
                if "timestamp" in df.columns:
                    # 먼저 to_datetime을 시도 (자동 형식 감지)
                    try:
                        df = df.with_columns(
                            pl.col("timestamp").str.to_datetime(strict=False, time_zone="UTC")
                        )
                    except Exception:
                        # 실패하면 여러 형식을 순차적으로 시도
                        formats = [
                            "%Y-%m-%d %H:%M:%S%.f %Z",
                            "%Y-%m-%dT%H:%M:%S%.f%z",
                            "%Y-%m-%d %H:%M:%S",
                            "%Y-%m-%dT%H:%M:%S",
                        ]
                        converted = False
                        for fmt in formats:
                            try:
                                df = df.with_columns(
                                    pl.col("timestamp").str.strptime(pl.Datetime, format=fmt, strict=False, time_zone="UTC")
                                )
                                converted = True
                                break
                            except Exception:
                                continue
                        if not converted:
                            logger.warning(f"Failed to parse timestamp column in {csv_file_path}, keeping as string")
                dfs.append(df)
                file_row_counts[csv_file_path.name] = len(df)
            except Exception as e:
                logger.error(f"Failed to load {csv_file_path}: {e}")
                continue
        
        if not dfs:
            return pl.DataFrame()
        
        
        combined_df = pl.concat(dfs, how="vertical", rechunk=True)
        
        # 중복 제거: 같은 close_time, agent, symbol 조합은 하나만 유지 (가장 최신 timestamp 우선)
        if not combined_df.is_empty():
            combined_df = (
                combined_df
                .sort("timestamp", descending=True)  # 최신 timestamp 우선
                .unique(subset=["close_time", "agent", "symbol"], keep="first")  # 첫 번째(가장 최신)만 유지
                .sort("close_time")  # close_time 순으로 정렬
            )
            
        
        return combined_df.sort("close_time")
    
    def get_combined_chart_data(
        self, 
        agent_name: str, 
        symbols: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        히스토리 데이터와 실시간 데이터를 합쳐서 차트용 데이터를 생성합니다.
        
        Args:
            agent_name: 에이전트 이름
            symbols: 심볼 리스트 (None이면 모든 심볼)
            
        Returns:
            심볼별 시계열 데이터 딕셔너리
        """
        # 히스토리 데이터 가져오기
        historical_df = self.historical_data.get(agent_name, pl.DataFrame())
        
        # 실시간 데이터 가져오기
        realtime_df = self.load_realtime_data(agent_name)
        
        # 두 데이터 합치기
        if historical_df.is_empty() and realtime_df.is_empty():
            return {}
        
        if historical_df.is_empty():
            combined_df = realtime_df
        elif realtime_df.is_empty():
            combined_df = historical_df
        else:
            combined_df = pl.concat([historical_df, realtime_df], how="vertical", rechunk=True)
        
        # 중복 제거 (같은 close_time이 있으면 실시간 데이터 우선)
        combined_df = combined_df.sort("close_time").unique(subset=["agent", "symbol", "close_time"], keep="last")
        
        # 심볼 필터링
        if symbols:
            combined_df = combined_df.filter(pl.col("symbol").is_in(symbols))
        
        # 심볼별로 그룹화하여 시계열 데이터 생성
        chart_data: Dict[str, List[Dict[str, Any]]] = {}
        for symbol in combined_df["symbol"].unique().to_list():
            symbol_df = combined_df.filter(pl.col("symbol") == symbol).sort("close_time")
            
            chart_data[symbol] = [
                {
                    "timestamp": row["timestamp"].isoformat() if isinstance(row["timestamp"], datetime) else str(row["timestamp"]),
                    "close_time": row["close_time"],
                    "price": float(row["current_price"]),
                    "quantity": float(row["quantity"]),
                    "allocation": float(row["allocation"]),
                    "position_value": float(row["position_value"]),
                }
                for row in symbol_df.iter_rows(named=True)
            ]
        
        return chart_data
    
    def get_total_value_chart_data(
        self,
        agent_name: str
    ) -> List[Dict[str, Any]]:
        """
        에이전트별 전체 total 가치 시계열 데이터를 생성합니다.
        
        Args:
            agent_name: 에이전트 이름
            
        Returns:
            시계열 데이터 리스트 (timestamp, total_value 포함)
        """
        
        # 히스토리 데이터 가져오기
        historical_df = self.historical_data.get(agent_name, pl.DataFrame())
        
        
        # 실시간 데이터 가져오기
        realtime_df = self.load_realtime_data(agent_name)
        
        
        # 두 데이터 합치기
        if historical_df.is_empty() and realtime_df.is_empty():
            return []
        
        if historical_df.is_empty():
            combined_df = realtime_df
        elif realtime_df.is_empty():
            combined_df = historical_df
        else:
            combined_df = pl.concat([historical_df, realtime_df], how="vertical", rechunk=True)
        
        # 중복 제거 (같은 close_time이 있으면 실시간 데이터 우선)
        combined_df = combined_df.sort("close_time").unique(subset=["agent", "symbol", "close_time"], keep="last")
        
        
        # 각 시간대별로 모든 심볼의 position_value 합계 계산
        total_value_df = (
            combined_df
            .group_by("close_time")
            .agg([
                pl.sum("position_value").alias("total_value"),
                pl.first("timestamp").alias("timestamp"),
            ])
            .sort("close_time")
        )
        
        
        # 시계열 데이터로 변환
        chart_data: List[Dict[str, Any]] = []
        for row in total_value_df.iter_rows(named=True):
            chart_data.append({
                "timestamp": row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
                "close_time": row["close_time"],
                "total_value": float(row["total_value"]),
            })
        
        
        return chart_data



