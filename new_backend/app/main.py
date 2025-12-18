"""
Binance Chart Backend - 독립적인 FastAPI 애플리케이션
"""
import os
import sys
import logging
import socket
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv
from loguru import logger

from .routers import binance_chart
from .binance_chart_scheduler import BinanceChartScheduler

# 환경 변수 로드
load_dotenv()

# loguru 설정: 컬러 출력 활성화 및 형식 설정
# 표준 logging과의 충돌 방지를 위해 기본 핸들러 제거 후 재설정
logger.remove()  # 기본 핸들러 제거
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True,
    backtrace=True,
    diagnose=True
)

# 표준 logging을 loguru로 인터셉트
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)

app = FastAPI(
    title="Binance Chart API",
    description="API for Binance trading chart visualization",
    version="1.0.0",
)

# CORS 디버깅을 위한 미들웨어 (CORS 미들웨어보다 먼저 실행되어야 함)
class CORSDebugMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")
        if origin:
            logger.warning(f"🌐 CORS Request Origin: {origin}")
            logger.warning(f"🌐 CORS Request Method: {request.method}")
            logger.warning(f"🌐 CORS Request Path: {request.url.path}")
            logger.warning(f"🌐 CORS Request Full URL: {request.url}")
        response = await call_next(request)
        # CORS 응답 헤더 확인
        cors_origin = response.headers.get("access-control-allow-origin")
        if cors_origin:
            logger.warning(f"🌐 CORS Response Allow-Origin: {cors_origin}")
        else:
            logger.error("❌ CORS Response Allow-Origin: MISSING!")
        return response

# CORS 설정
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://underdog0318.iptime.org:3000",
    "http://underdog0318.iptime.org",  # 포트 없이도 허용
]

# 환경 변수에서 추가 origin 허용
frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    allowed_origins.append(frontend_url)
    # 포트가 없는 경우 포트 3000 추가
    if ":" not in frontend_url:
        allowed_origins.append(f"{frontend_url}:3000")

# 네트워크 IP 기반 origin 추가 (192.168.x.x:3000 형식)
# 모든 로컬 네트워크 IP 허용 (개발 편의성)
hostname = socket.gethostname()
try:
    local_ip = socket.gethostbyname(hostname)
except socket.gaierror:
    # 호스트명을 IP로 변환할 수 없는 경우 기본값 사용
    local_ip = "127.0.0.1"
# 192.168.x.x 형식의 IP에 대해 여러 포트 허용
if local_ip.startswith("192.168."):
    for port in [3000, 3001, 5000, 5001]:
        allowed_origins.extend([
            f"http://{local_ip}:{port}",
            f"http://{local_ip.split('.')[0]}.{local_ip.split('.')[1]}.{local_ip.split('.')[2]}.x:{port}",
        ])

# 개발 환경에서는 모든 origin 허용 (개발 편의성)
# 프로덕션에서는 특정 origin만 허용하도록 설정
env = os.environ.get("ENV", "development")
logger.info(f"🌍 Environment: {env}")
logger.info(f"🌍 Requested origin will be checked against: {allowed_origins}")

# iptime.org 도메인 전체 허용 (개발 편의성)
# 프로덕션에서도 iptime.org는 허용
iptime_origins = [
    "http://underdog0318.iptime.org:3000",
    "http://underdog0318.iptime.org",
    "https://underdog0318.iptime.org:3000",
    "https://underdog0318.iptime.org",
    # 내부 IP도 추가 (포트포워딩 환경 대응)
    "http://192.168.0.63:3000",
    "http://192.168.0.63",
]

# iptime.org 도메인 패턴 추가
for iptime_origin in iptime_origins:
    if iptime_origin not in allowed_origins:
        allowed_origins.append(iptime_origin)
        logger.info(f"➕ Added CORS origin: {iptime_origin}")

# 중복 제거 및 정렬 (개발/프로덕션 모두 적용)
allowed_origins = sorted(list(set(allowed_origins)))

if env == "development":
    # 개발 환경: 모든 origin 허용
    logger.info("🔓 CORS: Allowing all origins (development mode)")
    logger.info(f"📡 Server IP: {local_ip}")
    logger.info(f"📋 Also configured specific origins ({len(allowed_origins)} total)")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 개발 환경에서만 사용
        allow_credentials=False,  # allow_origins=["*"]일 때는 credentials를 False로 설정해야 함
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
else:
    # 프로덕션 환경: 특정 origin만 허용
    logger.info(f"🔒 CORS: Allowing specific origins ({len(allowed_origins)} total):")
    for origin in allowed_origins:
        logger.info(f"   ✓ {origin}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

# 라우터 등록
app.include_router(binance_chart.router, prefix="/api")

# 전역 스케줄러 인스턴스
binance_chart_scheduler: BinanceChartScheduler = None


@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "Binance Chart API",
        "version": "1.0.0",
        "endpoints": {
            "agents": "/api/binance/chart/agents",
            "symbols": "/api/binance/chart/{agent_name}/symbols",
            "account": "/api/binance/chart/{agent_name}/account",
            "data": "/api/binance/chart/{agent_name}/data",
            "update": "/api/binance/chart/{agent_name}/update",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


@app.get("/health")
def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "ok"}


@app.on_event("startup")
def startup_event():
    """애플리케이션 시작 시 실행"""
    global binance_chart_scheduler
    
    logger.info("🚀 Binance Chart API starting up...")
    
    # 스케줄러 시작
    update_interval = int(os.environ.get("UPDATE_INTERVAL", 30))
    binance_chart_scheduler = BinanceChartScheduler(update_interval=update_interval)
    binance_chart_scheduler.start()
    
    logger.info("✅ Binance Chart API startup completed")


@app.on_event("shutdown")
def shutdown_event():
    """애플리케이션 종료 시 실행"""
    global binance_chart_scheduler
    
    logger.info("🛑 Binance Chart API shutting down...")
    
    if binance_chart_scheduler is not None:
        binance_chart_scheduler.stop()
    
    logger.info("✅ Binance Chart API shutdown completed")



