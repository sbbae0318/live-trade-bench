# 문제 해결 가이드

## ERR_CONNECTION_REFUSED 에러

### 원인
백엔드 서버가 실행되지 않았거나, 프론트엔드가 잘못된 URL로 요청을 보내고 있습니다.

### 해결 방법

#### 1. 백엔드 서버 실행 확인

```bash
# 백엔드 디렉토리로 이동
cd new_backend

# 백엔드 실행
python run.py
```

**확인 사항**:
- 터미널에 "Application startup complete" 메시지가 표시되는지 확인
- 포트 5001이 사용 중인지 확인: `lsof -i :5001`

#### 2. 백엔드 접속 테스트

백엔드가 실행 중인 상태에서:

```bash
# 로컬에서 테스트
curl http://localhost:5001/health

# 네트워크에서 테스트 (백엔드 서버 IP로)
curl http://192.168.0.63:5001/health
```

정상 응답: `{"status":"ok"}`

#### 3. 프론트엔드 API URL 설정

프론트엔드가 백엔드의 실제 IP로 요청을 보내도록 설정:

```bash
cd new_frontend

# .env 파일 생성
echo "REACT_APP_API_URL=http://192.168.0.63:5001" > .env

# 프론트엔드 재시작
npm start
```

#### 4. 방화벽 확인

백엔드 서버에서 포트 5001이 열려있는지 확인:

```bash
# macOS 방화벽 확인
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

# 포트가 열려있는지 확인
netstat -an | grep 5001
```

#### 5. 네트워크 연결 확인

프론트엔드와 백엔드가 같은 네트워크에 있는지 확인:

```bash
# 프론트엔드에서 백엔드로 ping
ping 192.168.0.63

# 포트 연결 테스트
nc -zv 192.168.0.63 5001
```

## 전체 실행 순서

### 1단계: 백엔드 실행

```bash
cd new_backend
python run.py
```

**성공 확인**:
- 터미널에 "🚀 Binance Chart API starting up..." 메시지
- "✅ Binance Chart API startup completed" 메시지
- `http://localhost:5001/docs` 접속 가능

### 2단계: 프론트엔드 설정 및 실행

```bash
cd new_frontend

# .env 파일 생성 (백엔드 IP 설정)
echo "REACT_APP_API_URL=http://192.168.0.63:5001" > .env

# 프론트엔드 실행
npm start
```

### 3단계: 브라우저 접속

- 프론트엔드: `http://localhost:3000` 또는 `http://[프론트엔드_IP]:3000`
- 백엔드 API: `http://192.168.0.63:5001`

## 디버깅 팁

### 백엔드 로그 확인

백엔드 터미널에서 다음 메시지들을 확인:
- "🔓 CORS: Allowing all origins (development mode)"
- "📡 Server IP: 192.168.0.63"
- 요청 로그가 표시되는지 확인

### 브라우저 개발자 도구

1. **Console 탭**: JavaScript 에러 확인
2. **Network 탭**: 
   - 요청 URL 확인 (`http://192.168.0.63:5001/api/...`)
   - 응답 상태 코드 확인 (200 OK인지)
   - CORS 헤더 확인

### 네트워크 테스트

```bash
# 백엔드 서버에서
curl http://0.0.0.0:5001/health

# 다른 컴퓨터에서
curl http://192.168.0.63:5001/health
```

두 명령어 모두 성공해야 합니다.

