"""
Binance Chart Backend 실행 스크립트
"""
import os
import uvicorn
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
    )

