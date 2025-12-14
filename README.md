# 프로젝트 아키텍처 및 상세 설계 (Advanced Project Architecture & Design)

본 문서는 지능형 비디오 편집 어시스턴트 프로젝트의 구현을 위한 상세 설계 문서입니다. **Modular Monolith** 아키텍처를 채택하여 도메인 간 결합도를 낮추고 확장성을 보장합니다.

---

## 1. 아키텍처 개요 (Architecture Overview)

### 1.1 하이레벨 아키텍처
시스템은 크게 두 가지 영역으로 구분됩니다:
1.  **Server Module (`src/modules/server`)**: 사용자 API 요청을 처리하는 웹 서버 (FastAPI)
2.  **Worker Module (`src/modules/indexing`)**: 비디오 인덱싱 등 무거운 작업을 비동기로 처리하는 백그라운드 워커 (Cloud Run)

```mermaid
graph TD
    User[User / Client] -->|REST / SSE| API_Server[Server Module]
    API_Server -->|Invoke| Chat_Module[Chat Component]
    API_Server -->|Enqueue| MongoDB[(MongoDB: index_jobs)]
    
    Worker[Worker Module] -->|Poll (Periodic)| MongoDB
    Worker -->|Execute| Indexing_Orchestrator[Pipeline Orchestrator]
    
    Chat_Module -->|Read| GCS[(Cloud Storage)]
    Indexing_Orchestrator -->|Read/Write| GCS
```

### 1.2 디렉토리 구조 (Directory Structure)
```text
src/
├── config/                      # 환경설정
├── modules/ (비즈니스 로직 - Bounded Contexts)
│   ├── indexing/                # Indexing Consumer & Worker
│   │   ├── application/         # Orcehstrator & Ports
│   │   ├── domain/              # Scene, Speech Entities
│   │   ├── infrastructure/      # Indexers, Analyzers, Adapters
│   │   └── main.py              # Worker Entrypoint
│   ├── server/                  # API Server
│   │   ├── api/                 # Endpoints (v1)
│   │   ├── application/         # Chat Workflow
│   │   └── main.py              # Server Entrypoint
│   └── chat/                    # (Logic shared by Server)
└── shared/ (공통 커널)
    ├── application/             # Shared Interfaces
    └── infrastructure/          # Shared Impls (GCS, VectorDB)
```

---

## 2. 모듈별 상세 설계 (Detailed Module Design)

### 2.1 Indexing Module (`src/modules/indexing`)

#### 역할
비디오 원본을 분석하여 검색 가능한 멀티모달 Vector Data로 변환, 저장합니다. **Polling-based Outbox Pattern**을 사용하여 안정적인 비동기 처리를 보장합니다.

#### 인덱싱 산출물 (Indexing Artifacts)
파이프라인 실행 시 스토리지(`projects/{project_id}/{video_id}/`)에 다음 파일들이 생성됩니다:

1.  **Media Assets**
    -   `video.mp4`: 원본 비디오 파일.
    -   `audio.wav`: 비디오에서 추출된 오디오 파일 (`16000Hz` Mono).
2.  **Analysis Data**
    -   `transcription.json`: AI 모델을 이용한 전체 타임스탬프 포함 대본.
    -   `scene_descriptions.json`: AI가 30초~5분 단위로 분석한 시각적 장면에 대한 상세 묘사.
    -   `speech_analysis.json`: 대본을 청크 단위로 나누고 요약/맥락을 보강한 데이터.
    -   `project.json`: 프로젝트 메타데이터.
3.  **Vector Store (FAISS)**
    -   `vision_vector_db.faiss`: 장면 묘사(Text) 기반의 시각 정보 검색용 인덱스.
    -   `vision_vector_db.faiss.metadata`: 시각 벡터와 매핑되는 메타데이터 (pickle).
    -   `speech_vector_db.faiss`: 대화 내용(Text) 기반의 청각 정보 검색용 인덱스.
    -   `speech_vector_db.faiss.metadata`: 음성 벡터와 매핑되는 메타데이터 (pickle).

#### 파이프라인 흐름 (Pipeline Orchestrator)
1.  **Download**: 스토리지에서 비디오 다운로드.
2.  **Extract**: 비디오에서 오디오 추출.
3.  **Check Artifacts**: 기존에 생성된 DB나 분석 파일이 있다면 다운로드 (재사용).
4.  **Parallel Execution**:
    -   **Visual Track**: Scene Analysis (`scene_analyzer`) -> Scene Indexing (`scene_indexer`).
    -   **Audio Track**: Speech Analysis (`speech_processor`) -> Speech Indexing (`speech_indexer`).
5.  **Persist**: 생성된 모든 Artifacts와 Vector DB를 스토리지에 업로드.

### 2.2 Server & Chat Module

#### 역할
-   **API Server**: 인덱싱 작업 요청 접수(`POST /videos`), 작업 상태 조회, 채팅 API 제공.
-   **Chat Logic**: LangGraph 기반 ReAct Agent. 인덱싱된 Vector DB를 `Storage`에서 로드하여 사용자 질의에 답변.

### 2.3 Shared Infrastructure (`src/shared`)
-   **GCPVideoRepository**: 스토리지 파일 입출력 전담. Artifact 이름 규칙 관리.
-   **VectorDB / SpeechVectorDB**: FAISS 라이브러리 래퍼. 인덱스 생성, 저장, 로드, 검색 기능 캡슐화.

