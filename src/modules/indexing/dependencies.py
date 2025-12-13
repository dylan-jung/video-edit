# Background에서 돌아가는 Worker에서는 Depends를 이용한
# FastAPI 의존성 주입(Dependency Injection)을 할 수 없기 때문에 수동으로 DI를 합니다.
# 인프라에 영향을 많이 받는 서비스이므로 이 과정을 통해 최대한 의존성을 낮춰야 합니다.

from src.shared.infrastructure.repository.storage import CloudStorageRepository
from src.shared.infrastructure.ai.ai_repository import AIRepository
from src.modules.indexing.application.orchestrator import PipelineOrchestrator

# Ports
from src.modules.indexing.application.ports.scene_analyzer_port import SceneAnalyzerPort
from src.modules.indexing.application.ports.speech_processor_port import SpeechProcessorPort
from src.modules.indexing.application.ports.embedding_port import EmbeddingPort
from src.modules.indexing.application.ports.job_repository_port import JobRepositoryPort

# Adapters
from src.modules.indexing.infrastructure.adapters.openai_scene_analyzer import OpenAISceneAnalyzer
from src.modules.indexing.infrastructure.adapters.whisper_speech_processor import WhisperSpeechProcessor
from src.modules.indexing.infrastructure.adapters.openai_embedding_service import OpenAIEmbeddingService
from src.modules.indexing.infrastructure.indexing.scene_indexer import SceneIndexer
from src.modules.indexing.infrastructure.indexing.speech_indexer import SpeechIndexer
from src.modules.indexing.infrastructure.repositories.job_repository import MongoIndexingJobRepository

@lru_cache
def get_cloud_storage_repository() -> CloudStorageRepository:
    return CloudStorageRepository()

@lru_cache
def get_ai_repository() -> AIRepository:
    return AIRepository()

@lru_cache
def get_embedding_service() -> EmbeddingPort:
    return OpenAIEmbeddingService()

@lru_cache
def get_scene_analyzer() -> SceneAnalyzerPort:
    return OpenAISceneAnalyzer()

@lru_cache
def get_speech_processor() -> SpeechProcessorPort:
    return WhisperSpeechProcessor()

@lru_cache
def get_scene_indexer() -> SceneIndexer:
    return SceneIndexer(get_embedding_service())

@lru_cache
def get_speech_indexer() -> SpeechIndexer:
    return SpeechIndexer(get_embedding_service())

@lru_cache
def get_indexing_job_repository() -> JobRepositoryPort:
    return MongoIndexingJobRepository()

@lru_cache
def get_pipeline_orchestrator() -> PipelineOrchestrator:
    return PipelineOrchestrator(
        repository=get_cloud_storage_repository(),
        ai_repository=get_ai_repository(),
        scene_analyzer=get_scene_analyzer(),
        speech_processor=get_speech_processor(),
        embedding_service=get_embedding_service(),
        scene_indexer=get_scene_indexer(),
        speech_indexer=get_speech_indexer()
    )
