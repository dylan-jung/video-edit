# 프로젝트 아키텍처 및 상세 설계 (Advanced Project Architecture & Design)

본 문서는 지능형 비디오 편집 어시스턴트 프로젝트의 구현을 위한 상세 설계 문서입니다. **Modular Monolith** 아키텍처를 채택하여 도메인 간 결합도를 낮추고 확장성을 보장합니다.

---

## 1. 아키텍처 개요 (Architecture Overview)

### 1.1 하이레벨 아키텍처
시스템은 크게 세 가지 영역으로 구분됩니다:
1.  **Server System**: 사용자 요청을 처리하는 API 서버 (FastAPI)
2.  **Worker System**: 비디오 인덱싱 등 무거운 작업을 비동기로 처리하는 백그라운드 워커 (Cloud Run)
3.  **Storage & DB**: 데이터 영속성 계층 (GCS, Firestore, Vector DB)

```mermaid
graph TD
    User[User / Client] -->|REST / SSE| API_Server[Cmd: Server (FastAPI)]
    API_Server -->|Invoke| Chat_Module[Module: Chat]
    API_Server -->|Write| MongoDB[(MongoDB: index_jobs)]
    
    Worker[Cmd: Worker (Cloud Run)] -->|Poll (Periodic)| MongoDB
    Worker -->|Execute| Indexing_Module[Module: Indexing]
    
    Chat_Module -->|Read| VectorDB[(Vector DB)]
    Indexing_Module -->|Write| VectorDB
    Indexing_Module -->|Read/Write| GCS[(Cloud Storage)]
```

### 1.2 디렉토리 구조 (Directory Structure)
```text
src/
├── cmd/ (애플리케이션 진입점)
│   ├── server/
│   │   └── main.py              # FastAPI 서버 엔트리포인트 (uvicorn 실행 대상)
│   └── worker/
│       └── main.py              # Worker 엔트리포인트 (JobPoller 실행)
├── config/                      # 환경설정
│   └── settings.py
├── modules/ (비즈니스 로직 - Bounded Contexts)
│   ├── indexing/
│   │   ├── application/         # 유스케이스 (PipelineOrchestrator)
│   │   ├── domain/              # 엔티티 (Scene, Speech, Video)
│   │   └── infrastructure/      # 구현체 (adapters, repositories, scene_analyzer)
│   ├── chat/
│   │   ├── application/         # 유스케이스 (AgentWorkflow)
│   │   ├── domain/              # 엔티티 (Message, Session)
│   │   └── infrastructure/      # 도구 (tools)
└── shared/ (공통 커널)
    ├── interfaces/              # 공통 인터페이스 (Repository 등)
    ├── infrastructure/          # 공통 인프라 (GCP Clients)
    └── utils/                   # 유틸리티
```

---

## 2. 모듈별 상세 설계 (Detailed Module Design)

### 2.1 Indexing Module (`src/modules/indexing`)

#### 역할
비디오 원본을 분석하여 AI가 이해할 수 있는 형태(Vector)로 변환합니다.

#### 데이터 정합성 및 신뢰성 (Polling-based Outbox Pattern)
시스템은 메시지 유실을 방지하고 데이터 정합성을 보장하기 위해 **DB Polling** 방식을 사용합니다.
- **Why**: 별도의 메시지 큐(Pub/Sub) 관리 비용을 제거하고, DB를 단일 진실 공급원(Source of Truth)으로 활용합니다.
- **Database**: MongoDB (`index_jobs` 컬렉션)
- **Status Flow**: `PENDING` -> `PROCESSING` -> `DONE` / `FAILED`

**프로세스**:
1.  **API 서버**: 클라이언트 요청 시, `index_jobs` 컬렉션에 `PENDING` 상태로 작업을 기록합니다.
2.  **Worker (Poller)**: 백그라운드 워커가 무한 루프를 돌며 DB에서 `PENDING` 작업을 조회(Polling)합니다.
    -   **처리량 조절 (Updates based on Throughput)**: 워커는 자신의 현재 처리 용량(예: Semaphore, Active Job Count)을 확인하여, **감당할 수 있는 속도에 맞춰서** 작업을 가져옵니다(Backpressure).
    -   작업을 가져올 때는 `find_one_and_update`를 사용하여 상태를 `PROCESSING`으로 원자적(Atomic)으로 변경하여 중복 처리를 방지합니다.
3.  **Indexing**: 작업을 수행하고 완료되면 `DONE`, 실패 시 `FAILED`로 업데이트합니다.

#### 주요 클래스 및 흐름
1.  **`JobPoller` (`infrastructure/job_poller.py`)**:
    -   `PENDING` 상태의 작업을 주기적으로 조회합니다.
    -   작업을 발견하면 `PipelineOrchestrator`를 호출합니다.

2.  **`PipelineOrchestrator` (`application/pipeline.py`)**:
    -   **책임**: 전체 인덱싱 트랜잭션 관리, 상태 업데이트, 예외 처리.
    -   **Flow**:
        ```python
        async def run_pipeline(project_id, video_id):
            download_resources()
            await asyncio.gather(
                run_visual_track(), # 시각 정보 분석
                run_audio_track()   # 청각 정보 분석
            )
            persist_results()
        ```

3.  **Domain Services (`infrastructure/`)**:
    -   **`SceneAnalyzer`**: `GPT-4o-mini`를 사용하여 30-300초 단위로 장면을 상세 묘사합니다.
    -   **`SpeechProcessor`**: `Whisper`로 전체 자막을 생성하고, 청크 단위로 나누어 맥락(Context)을 보강합니다.

### 2.2 Chat Module (`src/modules/chat`)

#### 역할
LangGraph 기반의 ReAct 에이전트를 구동하여 사용자의 질문에 대답합니다.

#### 주요 클래스
1.  **`AgentWorkflow` (`application/workflow.py`)**:
    -   `create_react_agent`를 사용하여 에이전트 그래프를 생성합니다.
    -   **상태 관리**: `checkpointer`를 통해 대화 히스토리를 저장 및 로드합니다.

2.  **`ToolRegistry` (`infrastructure/tools.py`)**:
    -   에이전트가 사용할 수 있는 도구들을 정의합니다.
    -   예: `VideoSearchTool`, `TimestampExtractionTool`

3.  **`StreamHandler` (`application/stream.py`)**:
    -   FastAPI의 `EventSourceResponse`를 사용하여, 에이전트의 사고 과정(Thought Process)과 최종 답변을 실시간 스트리밍합니다.

### 2.3 Shared & Infrastructure (`src/shared`)

#### 역할
모든 모듈에서 공통으로 사용하는 인프라 자원을 관리합니다.

1.  **`StorageRepository` (`shared/infrastructure/storage.py`)**:
    -   GCS(Google Cloud Storage) API를 추상화. `upload_file`, `download_file`, `generate_presigned_url` 메서드 제공.
2.  **`VectorDBClient`**:
    -   FAISS 인덱스 파일을 로드하고 검색하는 기능을 제공. 모듈 간 의존성 없이 `shared`에 위치하여 Indexing(Write)과 Chat(Read) 양쪽에서 사용.
