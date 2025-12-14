import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.modules.indexing.infrastructure.job_poller import JobPoller
from .api.router import router as indexing_router

from .dependencies import (
    get_indexing_job_repository,
    get_pipeline_orchestrator
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
        
        # Instantiate Orchestrator
        orchestrator = get_pipeline_orchestrator()
        
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

app.include_router(indexing_router, prefix="/api/v1", tags=["indexing"])

@app.get("/health")
async def health_check():
    return {"status": "ok", "poller_running": poller.running if poller else False}
