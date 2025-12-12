from src.shared.infrastructure.repository.storage import CloudStorageRepository
from src.shared.infrastructure.ai.ai_repository import AIRepository

# Ports
from src.modules.indexing.application.ports.scene_analyzer_port import SceneAnalyzerPort
from src.modules.indexing.application.ports.speech_processor_port import SpeechProcessorPort
from src.modules.indexing.application.ports.embedding_port import EmbeddingPort

# Adapters
from src.modules.indexing.infrastructure.adapters.openai_scene_analyzer import OpenAISceneAnalyzer
from src.modules.indexing.infrastructure.adapters.whisper_speech_processor import WhisperSpeechProcessor
from src.modules.indexing.infrastructure.adapters.openai_embedding_service import OpenAIEmbeddingService
from src.modules.indexing.infrastructure.indexing.scene_indexer import SceneIndexer
from src.modules.indexing.infrastructure.indexing.speech_indexer import SpeechIndexer
from src.modules.indexing.infrastructure.repositories.job_repository import MongoIndexingJobRepository
from src.modules.indexing.application.ports.job_repository_port import JobRepositoryPort

def get_cloud_storage_repository() -> CloudStorageRepository:
    return CloudStorageRepository()

def get_ai_repository() -> AIRepository:
    return AIRepository()

def get_embedding_service() -> EmbeddingPort:
    return OpenAIEmbeddingService()

def get_scene_analyzer() -> SceneAnalyzerPort:
    return OpenAISceneAnalyzer()

def get_speech_processor() -> SpeechProcessorPort:
    return WhisperSpeechProcessor()

def get_scene_indexer() -> SceneIndexer:
    # SceneIndexer depends on EmbeddingService. 
    # Since we are not using Depends within SceneIndexer yet (it's not a Pydantic model or API router), 
    # we instantiate it here. OR we can make SceneIndexer a simple class and rely on manual wiring here.
    return SceneIndexer(get_embedding_service())

def get_speech_indexer() -> SpeechIndexer:
    return SpeechIndexer(get_embedding_service())

def get_indexing_job_repository() -> JobRepositoryPort:
    return MongoIndexingJobRepository()
