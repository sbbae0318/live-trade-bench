# Binance Chart Backend

Binance 거래 데이터를 실시간으로 시각화하기 위한 독립적인 FastAPI 백엔드 애플리케이션입니다.

## 주요 기능

1. **히스토리 데이터 로드**: `save_accounts_parquet`에서 저장한 형식과 동일하게 parquet 파일을 읽어서 히스토리 데이터를 로드합니다.
2. **Account 정보 관리**: 각 모델별 account 정보(심볼별 배분 비율과 quantity)를 관리합니다.
3. **실시간 가격 업데이트**: 주기적으로 Binance API를 통해 각 심볼별 실제 가격을 가져와서 플롯용 시계열 데이터를 생성합니다.
4. **데이터 통합**: 플롯용 데이터와 실제 거래 시 발생한 데이터를 합쳐서 로드합니다.

## 설치

```bash
pip install -r requirements.txt
```

## 환경 변수

`.env` 파일을 생성하여 다음 변수를 설정할 수 있습니다:

```env
BASE_DIR=/path/to/data/directory  # Parquet 파일이 저장된 디렉토리 경로
PORT=5001                         # 서버 포트 (기본값: 5001)
UPDATE_INTERVAL=10                 # 가격 업데이트 간격 (초, 기본값: 10)
FRONTEND_URL=http://localhost:3000 # 프론트엔드 URL (CORS용)
```

## 실행

```bash
python run.py
```

또는 uvicorn을 직접 사용:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5001
```

## 디버깅

VS Code에서 디버깅하려면:

1. **F5** 키를 누르거나 디버그 패널에서 실행
2. 다음 디버그 설정 중 선택:
   - **Python: FastAPI (new_backend)**: uvicorn을 reload 모드로 실행 (코드 변경 시 자동 재시작)
   - **Python: run.py (new_backend)**: run.py를 직접 실행
   - **Python: FastAPI (reload disabled)**: reload 없이 실행 (디버깅에 더 안정적)

### 브레이크포인트 설정

코드에 브레이크포인트를 설정하려면:
- 원하는 줄 번호 왼쪽을 클릭하여 빨간 점 표시
- 디버깅 시작 후 해당 지점에서 실행이 멈춤
- 변수 값 확인, 단계별 실행 등 디버깅 기능 사용 가능

### 환경 변수

디버깅 시 `.env` 파일이 자동으로 로드됩니다. `.env.example`을 참고하여 `.env` 파일을 생성하세요.

## API 엔드포인트

- `GET /` - API 정보
- `GET /health` - 헬스 체크
- `GET /api/binance/chart/agents` - 에이전트 목록
- `GET /api/binance/chart/{agent_name}/symbols` - 심볼 목록
- `GET /api/binance/chart/{agent_name}/account` - Account 정보
- `GET /api/binance/chart/{agent_name}/data` - 차트 데이터
- `POST /api/binance/chart/{agent_name}/update` - 실시간 데이터 업데이트

자세한 API 문서는 `/docs` (Swagger UI) 또는 `/redoc`에서 확인할 수 있습니다.

## 프로젝트 구조

```
new_backend/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI 애플리케이션
│   ├── binance_chart_data.py      # 데이터 관리 클래스
│   ├── binance_chart_scheduler.py # 스케줄러
│   └── routers/
│       ├── __init__.py
│       └── binance_chart.py       # API 라우터
├── run.py                         # 실행 스크립트
├── requirements.txt               # Python 의존성
└── README.md                      # 이 파일
```

