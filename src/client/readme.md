# Client Module (Reference Implementation)

> [!IMPORTANT]
> **이 디렉토리는 실제 클라이언트 구현체가 아닙니다.**
> 클라이언트가 수행해야 할 핵심 작업 흐름을 **모사(Simulation)** 한 예시 코드입니다.
> 이 로직과 파이프라인을 참고하여 실제 클라이언트 애플리케이션(Web, Mobile Agent 등)에 해당 기능을 **구현해 주시길 바랍니다.**

이 모듈은 비디오 편집 시스템의 '작업자(Worker)' 역할을 수행해야 하는 클라이언트의 책임을 정의합니다.
주로 **무거운 미디어 처리 작업(FFmpeg)**을 담당하며, 크게 **추출(Extraction)**과 **렌더링(Rendering)** 두 가지 핵심 책임을 가집니다.

## 📂 구조 (Structure)

- **`extract/`**: 원본 비디오를 분석하고, AI가 처리하기 쉬운 형태로 가공하여 업로드하는 파이프라인.
- **`render/`**: AI 또는 사용자의 편집 지시(`editing_state.json`)를 받아 실제 결과물 비디오를 생성하는 파이프라인.

---

## 🚀 1. 인덱싱 및 추출 (Ingestion & Extraction)

새로운 비디오가 들어왔을 때, 서버/AI가 효율적으로 분석할 수 있도록 전처리를 수행하고 클라우드로 업로드합니다.
`src/client/extract/pipeline.py`가 이 과정을 조율합니다.

### 주요 단계
1.  **비디오 샘플링 & 전처리 (Preprocessing)**
    *   **리사이징 (Resizing)**: AI 모델의 입력 크기(예: 240px 등)에 맞춰 비디오 해상도를 조절합니다.
    *   **패딩 (Padding)**: 원본 비율을 유지하면서 타겟 해상도에 맞추기 위해 Letterbox(검은 여백)를 추가할 수 있습니다.
    *   **타임스탬프 (Timestamp)**: 디버깅 및 분석 용이를 위해 영상 상단에 시간 정보를 오버레이합니다.
2.  **오디오 추출 및 정제 (Audio Extraction)**
    *   비디오에서 오디오 트랙을 분리합니다.
    *   **노이즈 제거**: FFmpeg의 `afftdn` 필터 등을 사용하여 음성 인식률을 높이기 위해 배경 소음을 줄입니다.
3.  **메타데이터 추출 (Metadata)**
    *   `ffprobe`를 사용하여 정밀한 파일 정보(생성 시간, 길이, FPS, 코덱 등)를 추출하여 `metadata.json`으로 저장합니다.
4.  **업로드 (Upload)**
    *   처리된 비디오, 오디오, 메타데이터 파일을 Presigned URL(또는 Cloud Storage)을 통해 저장소로 전송합니다.

---

## 🎬 2. 비디오 렌더링 (Rendering)

AI 에이전트나 사용자가 편집을 마친 후, 실제 최종 비디오 파일을 생성하는 단계입니다.
`src/client/render/render_video.py`가 담당합니다.

### 렌더링 프로세스
1.  **편집 상태 읽기 (`editing_state.json`)**
    *   프로젝트 ID와 트랙 정보(어떤 비디오의 몇 분 몇 초 구간을 사용할지)를 로드합니다.
2.  **세그먼트 추출 (Segment Extraction)**
    *   **Smart Cut**: 원본 비디오에서 필요한 구간만 잘라냅니다.
    *   **Stream Copy**: 가능한 경우 재인코딩 없이 코덱 데이터를 그대로 복사하여 화질 저하를 막고 처리 속도를 극대화합니다 (`-c copy`).
3.  **병합 (Concatenation)**
    *   추출된 여러 세그먼트 파일들을 순서대로 이어 붙여 하나의 완성된 `.mp4` 파일을 생성합니다.

## 🛠️ 요구 사항 (Dependencies)
이 모듈은 미디어 처리를 위해 시스템에 다음 도구들이 설치되어 있어야 합니다.
*   **Create Video**: `ffmpeg`
*   **Analyze Video**: `ffprobe`