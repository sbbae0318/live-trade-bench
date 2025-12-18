# 네트워크 접속 설정 가이드

## 문제 상황

백엔드가 `192.168.0.63:5001`에서 실행 중이고, 프론트엔드가 다른 IP에서 접속할 때 CORS 에러가 발생합니다.

## 해결 방법

### 1. 프론트엔드 API URL 설정

프론트엔드가 백엔드의 실제 IP로 요청을 보내도록 설정해야 합니다.

#### 방법 1: .env 파일 생성 (권장)

```bash
cd new_frontend
cat > .env << EOF
REACT_APP_API_URL=http://192.168.0.63:5001
EOF
```

#### 방법 2: 환경 변수로 직접 설정

```bash
cd new_frontend
REACT_APP_API_URL=http://192.168.0.63:5001 npm start
```

### 2. 프론트엔드 재시작

환경 변수를 변경한 후에는 반드시 프론트엔드를 재시작해야 합니다:

```bash
# 기존 프로세스 중지 (Ctrl+C)
# 다시 시작
npm start
```

### 3. 백엔드 확인

백엔드는 이미 다음 설정으로 되어 있습니다:
- `host="0.0.0.0"` - 모든 네트워크 인터페이스에서 리스닝
- CORS: 개발 모드에서 모든 origin 허용

백엔드가 올바르게 실행 중인지 확인:

```bash
# 백엔드 서버에서
curl http://192.168.0.63:5001/health

# 다른 컴퓨터에서
curl http://192.168.0.63:5001/health
```

### 4. 브라우저에서 확인

브라우저 개발자 도구(F12) → Network 탭에서:

1. **요청 URL 확인**: `http://192.168.0.63:5001/api/binance/chart/agents`
2. **응답 헤더 확인**: 
   - `Access-Control-Allow-Origin: *` 또는
   - `Access-Control-Allow-Origin: http://[프론트엔드_IP]:3000`

## 빠른 해결 체크리스트

- [ ] 프론트엔드 `.env` 파일에 `REACT_APP_API_URL=http://192.168.0.63:5001` 설정
- [ ] 프론트엔드 재시작 (`npm start`)
- [ ] 백엔드가 `0.0.0.0:5001`에서 실행 중인지 확인
- [ ] 방화벽이 포트 5001을 허용하는지 확인
- [ ] 브라우저 개발자 도구에서 실제 요청 URL 확인

## 문제 해결

### 여전히 CORS 에러가 발생하는 경우

1. **브라우저 캐시 클리어**: Ctrl+Shift+R (하드 리프레시)
2. **백엔드 재시작**: 백엔드 서버를 재시작하여 CORS 설정 적용
3. **네트워크 확인**: 프론트엔드와 백엔드가 같은 네트워크에 있는지 확인
4. **방화벽 확인**: 포트 5001이 열려있는지 확인

### 디버깅 팁

브라우저 개발자 도구 → Console에서 다음 명령어로 테스트:

```javascript
fetch('http://192.168.0.63:5001/api/binance/chart/agents')
  .then(r => r.json())
  .then(console.log)
  .catch(console.error)
```

이 명령어로 CORS 에러의 정확한 원인을 확인할 수 있습니다.

