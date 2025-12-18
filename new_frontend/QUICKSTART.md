# 빠른 시작 가이드

## 전체 시스템 실행하기

### 1. 백엔드 실행

```bash
# 터미널 1
cd new_backend
pip install -r requirements.txt
python run.py
```

**확인**: 브라우저에서 `http://localhost:5001/docs` 접속하여 API 문서 확인

### 2. 프론트엔드 실행

```bash
# 터미널 2
cd new_frontend
npm install
npm start
```

**확인**: 브라우저에서 `http://localhost:3000` 접속

### 3. 사용하기

1. 브라우저에서 `http://localhost:3000` 접속
2. 에이전트 선택 (드롭다운에서 선택)
3. 심볼 선택 (체크박스로 다중 선택 가능)
4. 차트 확인 (자동으로 10초마다 업데이트)

## 환경 변수 설정 (선택사항)

### 백엔드 (.env)

```bash
cd new_backend
cat > .env << EOF
BASE_DIR=/Users/sbbae/project/ltb-runs
PORT=5001
UPDATE_INTERVAL=10
FRONTEND_URL=http://localhost:3000
EOF
```

### 프론트엔드 (.env)

```bash
cd new_frontend
cat > .env << EOF
REACT_APP_API_URL=http://localhost:5001
EOF
```

## 주요 엔드포인트

- **프론트엔드**: http://localhost:3000
- **백엔드 API**: http://localhost:5001
- **API 문서**: http://localhost:5001/docs
- **헬스 체크**: http://localhost:5001/health

## 문제 해결

### 백엔드가 시작되지 않음
- Python 가상환경 활성화 확인
- `requirements.txt`의 패키지 설치 확인
- 포트 5001이 사용 중인지 확인

### 프론트엔드가 시작되지 않음
- Node.js 버전 확인 (16 이상 필요)
- `node_modules` 삭제 후 `npm install` 재실행
- 포트 3000이 사용 중인지 확인

### 차트가 표시되지 않음
- 브라우저 개발자 도구(F12)에서 네트워크 탭 확인
- 백엔드 API가 정상 응답하는지 확인
- CORS 오류가 있는지 확인

