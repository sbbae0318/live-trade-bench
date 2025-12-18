# Binance Chart Frontend

Binance 거래 데이터를 실시간으로 시각화하기 위한 독립적인 React Single Page Application입니다.

## 주요 기능

- 에이전트 선택 및 심볼 다중 선택
- 실시간 차트 업데이트 (기본 10초 간격)
- Chart.js를 사용한 시계열 차트 표시
- 가격과 포지션 가치를 동시에 표시

## 사전 요구사항

- Node.js 16 이상
- npm 또는 yarn
- 백엔드 서버가 실행 중이어야 함 (기본 포트: 5001)

## 설치

```bash
cd new_frontend
npm install
```

## 환경 변수 설정

`.env` 파일을 생성하여 백엔드 API URL을 설정할 수 있습니다:

```bash
cd new_frontend
cat > .env << EOF
REACT_APP_API_URL=http://localhost:5001
EOF
```

**참고**: `.env` 파일을 생성하지 않으면 기본값 `http://localhost:5001`이 사용됩니다.

## 실행 방법

### 개발 모드 실행

```bash
cd new_frontend
npm start
```

실행 후 자동으로 브라우저가 열리며 `http://localhost:3000`으로 접속됩니다.

**터미널 출력 예시**:
```
Compiled successfully!

You can now view binance-chart-frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000

Note that the development build is not optimized.
To create a production build, use npm run build.
```

### 수동 접속

브라우저가 자동으로 열리지 않는 경우:
1. 웹 브라우저를 엽니다
2. 주소창에 `http://localhost:3000` 입력
3. Enter 키를 누릅니다

### 포트 변경

기본 포트(3000)가 사용 중인 경우, 다른 포트로 실행할 수 있습니다:

```bash
PORT=3001 npm start
```

또는 환경 변수로 설정:

```bash
# .env 파일에 추가
PORT=3001
```

## 빌드 및 프로덕션 배포

### 프로덕션 빌드

```bash
cd new_frontend
npm run build
```

빌드된 파일은 `build/` 디렉토리에 생성됩니다.

### 정적 파일 서빙

빌드된 파일을 서빙하려면:

```bash
# serve 패키지 사용 (설치 필요: npm install -g serve)
serve -s build -l 3000

# 또는 Python 사용
cd build
python -m http.server 3000
```

## 전체 실행 흐름

### 1단계: 백엔드 실행

```bash
# 터미널 1
cd new_backend
python run.py
```

백엔드가 `http://localhost:5001`에서 실행됩니다.

### 2단계: 프론트엔드 실행

```bash
# 터미널 2
cd new_frontend
npm start
```

프론트엔드가 `http://localhost:3000`에서 실행됩니다.

### 3단계: 브라우저 접속

브라우저에서 `http://localhost:3000`으로 접속하여 차트를 확인합니다.

## 문제 해결

### 백엔드 연결 오류

**증상**: "Failed to fetch agents" 또는 네트워크 오류

**해결 방법**:
1. 백엔드 서버가 실행 중인지 확인: `http://localhost:5001/health` 접속
2. `.env` 파일의 `REACT_APP_API_URL`이 올바른지 확인
3. CORS 설정 확인 (백엔드의 `main.py`에서 프론트엔드 URL 허용 확인)

### 포트 충돌

**증상**: "Port 3000 is already in use"

**해결 방법**:
```bash
# 다른 포트 사용
PORT=3001 npm start
```

### 모듈 설치 오류

**증상**: `npm install` 실패

**해결 방법**:
```bash
# 캐시 클리어 후 재설치
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

## 프로젝트 구조

```
new_frontend/
├── public/
│   └── index.html          # HTML 템플릿
├── src/
│   ├── components/
│   │   ├── BinanceChart.tsx    # 메인 차트 컴포넌트
│   │   └── BinanceChart.css    # 차트 스타일
│   ├── App.tsx             # 메인 앱 컴포넌트
│   ├── App.css             # 앱 스타일
│   ├── index.tsx            # 진입점
│   └── index.css            # 전역 스타일
├── package.json            # 의존성 및 스크립트
├── tsconfig.json           # TypeScript 설정
└── README.md               # 이 파일
```

## 사용 방법

1. 백엔드 서버가 실행 중인지 확인하세요.
2. 프론트엔드를 시작하세요: `npm start`
3. 브라우저에서 에이전트와 심볼을 선택하여 차트를 확인하세요.

## 기술 스택

- React 19
- TypeScript
- Chart.js / react-chartjs-2
- CSS3



