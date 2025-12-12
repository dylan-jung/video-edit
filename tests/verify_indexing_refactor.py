import sys
import os
from unittest.mock import MagicMock

# Add src to path
# Add src to path
sys.path.append(os.getcwd())

from src.modules.indexing.infrastructure.adapters.openai_scene_analyzer import OpenAISceneAnalyzer
from src.modules.indexing.infrastructure.adapters.whisper_speech_processor import WhisperSpeechProcessor
from src.modules.indexing.infrastructure.adapters.openai_embedding_service import OpenAIEmbeddingService
from src.modules.indexing.infrastructure.indexing.scene_indexer import SceneIndexer
from src.modules.indexing.infrastructure.indexing.speech_indexer import SpeechIndexer
from src.modules.indexing.application.orchestrator import PipelineOrchestrator

def test_imports():
    print("Imports successful.")

def test_instantiation():
    print("Testing instantiation...")
    try:
        # Mock env vars
        os.environ["OPENAI_API_KEY"] = "sk-test"
        
        # Mock internal dependencies if they try to connect to real services
        # OpenAISceneAnalyzer checks API key in init. Valid.
        # It imports Video Utils. Valid.
        
        analyzer = OpenAISceneAnalyzer()
        print("OpenAISceneAnalyzer instantiated.")
        
        processor = WhisperSpeechProcessor()
        print("WhisperSpeechProcessor instantiated.")
        
        # Mock OpenAI Client to avoid API connection in init if it does that
        # OpenAI() client init typically doesn't connect until request, but let's be safe or just let it be.
        # OpenAIEmbeddingService inits OpenAI client.
        embedding = OpenAIEmbeddingService()
        print("OpenAIEmbeddingService instantiated.")
        
        scene_indexer = SceneIndexer(embedding)
        print("SceneIndexer instantiated.")
        
        speech_indexer = SpeechIndexer(embedding)
        print("SpeechIndexer instantiated.")
        
        # Mock Repositories
        repo = MagicMock()
        ai_repo = MagicMock()

        # Test DI
        orchestrator_di = PipelineOrchestrator(
            repository=repo,
            ai_repository=ai_repo,
            scene_analyzer=analyzer,
            speech_processor=processor,
            embedding_service=embedding,
            scene_indexer=scene_indexer,
            speech_indexer=speech_indexer
        )
        print("PipelineOrchestrator instantiated (DI).")
        
        # Verify DI worked
        if orchestrator_di.scene_analyzer != analyzer:
            raise ValueError("DI Failed: scene_analyzer mismatch")
        if orchestrator_di.speech_processor != processor:
            raise ValueError("DI Failed: speech_processor mismatch")
        if orchestrator_di.repository != repo:
            raise ValueError("DI Failed: repository mismatch")
        if orchestrator_di.ai_repository != ai_repo:
            raise ValueError("DI Failed: ai_repository mismatch")
        
        print("DI Verification successful.")
        
        print("Instantiation successful.")
    except Exception as e:
        print(f"Instantiation failed: {e}")
        raise

if __name__ == "__main__":
    test_imports()
    test_instantiation()
