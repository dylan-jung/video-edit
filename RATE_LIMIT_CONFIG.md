# OpenAI Rate Limit 설정 가이드

## 문제 상황

```
openai.RateLimitError: Error code: 429 - Rate limit reached for gpt-4.1-mini
Limit: 200000 TPM, Used: 116185, Requested: 88633
```

## 해결책

### 1. 환경변수 설정 (.env.local)

```bash
# OpenAI API Configuration for Rate Limiting
AGENT_MODEL=gpt-4o-mini  # mini 버전 사용으로 비용/Rate Limit 절약
AGENT_MODEL_API_KEY=your_openai_api_key_here
AGENT_MODEL_SERVER=https://api.openai.com/v1

# Rate Limiting Settings
OPENAI_MAX_RETRIES=3
OPENAI_BASE_DELAY=1.0
OPENAI_MAX_TOKENS=8000  # 200000에서 8000으로 축소
OPENAI_REQUEST_TIMEOUT=60

# Video Processing Limits
MAX_FRAMES_PER_VIDEO_CALL=10
TOKENS_PER_FRAME_ESTIMATE=1500
```

### 2. 코드 변경사항 요약

#### A. LLM 초기화 개선 (`src/server/agent/model/model.py`)

- ✅ `RateLimitedChatOpenAI` 클래스 추가
- ✅ 자동 재시도 로직 (exponential backoff)
- ✅ OpenAI 에러 메시지에서 대기시간 자동 추출
- ✅ max_tokens를 200000 → 8000으로 축소

#### B. 비디오 도구 개선 (`src/server/agent/tools/read_video_tool.py`)

- ✅ 프레임 수 제한 (최대 10개)
- ✅ 토큰 사용량 예측 및 표시
- ✅ 사전 검증으로 Rate Limit 방지
- ✅ 사용자에게 최적화 제안

### 3. 사용법 가이드

#### 비디오 분석 시 권장사항:

```python
# ❌ 잘못된 사용 (Rate Limit 유발)
read_video("video1", "00:00:00", "00:05:00", fps=2)  # 600개 프레임 = ~900k 토큰

# ✅ 올바른 사용
read_video("video1", "00:00:00", "00:00:10", fps=1)  # 10개 프레임 = ~15k 토큰
```

#### 긴 비디오 분석 방법:

```python
# 10초씩 나누어서 분석
read_video("video1", "00:00:00", "00:00:10", fps=1)
read_video("video1", "00:00:10", "00:00:20", fps=1)
read_video("video1", "00:00:20", "00:00:30", fps=1)
```

### 4. Rate Limit 모니터링

도구 실행 시 다음 정보가 표시됩니다:

```
📊 예상 토큰 사용량: ~15,000 tokens (10 frames)
✅ 비디오 'video1'에서 00:00:00-00:00:10 구간, 1fps로 10개 프레임을 추출했습니다.
```

### 5. 추가 최적화 옵션

#### OpenAI API 플랜 업그레이드

- **Tier 1**: 200K TPM → **Tier 2**: 1M TPM
- [OpenAI Rate Limits](https://platform.openai.com/account/rate-limits)에서 확인

#### 대안 모델 사용

```bash
# 더 저렴한 모델로 변경
AGENT_MODEL=gpt-3.5-turbo  # 훨씬 저렴하지만 성능 하락
```

#### 프레임 수 추가 제한

```python
# read_video_tool.py에서 max_frames_per_call 조정
self.max_frames_per_call = 5  # 기본값 10에서 5로 감소
```

### 6. 에러 발생 시 자동 대응

1. **자동 재시도**: 3회까지 exponential backoff로 재시도
2. **대기시간 자동 추출**: OpenAI 응답에서 권장 대기시간 사용
3. **프레임 수 자동 제한**: 10개 초과 시 자동으로 잘라냄
4. **사전 검증**: 요청 전 토큰 수 추정하여 제한

이 설정으로 Rate Limit 에러를 크게 줄일 수 있습니다.
