# 비디오 렌더링 스크립트

`editing_state.json` 파일을 기반으로 비디오를 실제로 렌더링하는 스크립트입니다.

## 필요한 소프트웨어

- **FFmpeg**: 비디오 처리를 위해 필요합니다.

  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu/Debian
  sudo apt install ffmpeg

  # Windows (Chocolatey)
  choco install ffmpeg
  ```

- **Python 3**: 스크립트 실행을 위해 필요합니다.

## 사용법

### 1. Python 스크립트 직접 사용

```bash
# 기본 사용법
python3 render_video.py projects/test/editing_state.json

# 출력 파일명 지정
python3 render_video.py projects/test/editing_state.json -o my_video.mp4

# 상세한 출력
python3 render_video.py projects/test/editing_state.json -v
```

### 2. 쉘 스크립트 사용 (권장)

먼저 실행 권한을 부여하세요:

```bash
chmod +x render_project.sh
```

사용법:

```bash
# 프로젝트 목록 보기
./render_project.sh --list

# test 프로젝트 렌더링 (자동 파일명)
./render_project.sh test

# test 프로젝트를 특정 파일명으로 렌더링
./render_project.sh test final_video.mp4

# 도움말
./render_project.sh --help
```

## 동작 원리

1. `editing_state.json` 파일을 읽어서 트랙 정보를 파싱합니다.
2. 각 트랙의 `trimIn`과 `trimOut` 시간을 기반으로 소스 비디오에서 해당 구간을 추출합니다.
3. 추출된 세그먼트들을 시간 순서대로 연결하여 최종 비디오를 생성합니다.

## 예시: editing_state.json 구조

```json
{
  "project_id": "test",
  "tracks": [
    {
      "src": "ea48283a31baa560",
      "start": "00:00:00",
      "end": "00:01:39",
      "duration": "00:01:39",
      "trimIn": "00:06:20",
      "trimOut": "00:07:59"
    },
    {
      "src": "ea48283a31baa560",
      "start": "00:01:39",
      "end": "00:02:41",
      "duration": "00:01:02",
      "trimIn": "00:07:59",
      "trimOut": "00:09:01"
    }
  ]
}
```

이 예시에서는:

- 소스 비디오 `ea48283a31baa560/video.mp4`에서
- 첫 번째 세그먼트: 6분 20초 ~ 7분 59초 구간을 추출
- 두 번째 세그먼트: 7분 59초 ~ 9분 1초 구간을 추출
- 두 세그먼트를 연결하여 최종 비디오 생성

## 출력 파일

- 기본 출력 파일명: `rendered_{project_id}_{timestamp}.mp4`
- 예: `rendered_test_20241211_143021.mp4`

## 문제 해결

### FFmpeg 관련 오류

- FFmpeg가 설치되어 있는지 확인: `ffmpeg -version`
- PATH 환경변수에 FFmpeg가 포함되어 있는지 확인

### 파일 경로 오류

- `editing_state.json` 파일이 올바른 위치에 있는지 확인
- 소스 비디오 파일 `{src}/video.mp4`가 존재하는지 확인

### 권한 오류

- 출력 디렉토리에 쓰기 권한이 있는지 확인
- macOS에서는 터미널에 파일 접근 권한이 있는지 확인

## 성능 최적화

- 기본적으로 `-c copy` 옵션을 사용하여 재인코딩 없이 빠르게 처리합니다.
- 대용량 비디오의 경우 시간이 오래 걸릴 수 있습니다.
- SSD 디스크를 사용하면 더 빠른 처리가 가능합니다.
