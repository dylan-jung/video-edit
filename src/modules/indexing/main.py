import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.modules.indexing.application.orchestrator import PipelineOrchestrator
from src.modules.indexing.infrastructure.job_poller import JobPoller
from src.modules.indexing.infrastructure.dependencies import (
    get_scene_analyzer,
    get_speech_processor,
    get_embedding_service,
    get_scene_indexer,
    get_speech_indexer,
    get_cloud_storage_repository,
    get_ai_repository,
    get_indexing_job_repository
)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

poller = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global poller
    try:
        logger.info("Initializing Worker...")
        
        # Instantiate Orchestrator with dependencies manually
        orchestrator = PipelineOrchestrator(
            scene_analyzer=get_scene_analyzer(),
            speech_processor=get_speech_processor(),
            embedding_service=get_embedding_service(),
            scene_indexer=get_scene_indexer(),
            speech_indexer=get_speech_indexer(),
            repository=get_cloud_storage_repository(),
            ai_repository=get_ai_repository()
        )
        
        # Instantiate Repository
        repository = get_indexing_job_repository()
        
        # Instantiate Poller
        poller = JobPoller(repository, orchestrator)
        
        # Start Polling (Background Task)
        asyncio.create_task(poller.start())
        logger.info("Worker Initialized and Poller Started.")
        
        yield
        
    finally:
        if poller:
            logger.info("Stopping Poller...")
            poller.stop()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok", "poller_running": poller.running if poller else False}
