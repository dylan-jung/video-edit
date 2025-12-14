import asyncio
import logging
from src.modules.indexing.application.ports.job_repository_port import JobRepositoryPort
from src.modules.indexing.application.orchestrator import PipelineOrchestrator
from src.modules.indexing.domain.indexing_job import IndexingJobStatus, IndexingJob

logger = logging.getLogger(__name__)

class JobPoller:
    def __init__(self, 
                 job_repository: JobRepositoryPort,
                 orchestrator: PipelineOrchestrator, 
                 max_concurrency: int = 2,
                 poll_interval: float = 1.0):
        self.job_repository = job_repository
        self.orchestrator = orchestrator
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.poll_interval = poll_interval
        self.running = False

    async def start(self):
        """Starts the polling loop."""
        self.running = True
        logger.info("Starting JobPoller loop...")
        while self.running:
            try:
                # If we are at full capacity, wait a bit
                if self.semaphore.locked():
                    await asyncio.sleep(0.5)
                    continue

                # Try to acquire a job
                job = await self.job_repository.acquire_next_pending_job()
                if job:
                    logger.info(f"Acquired job {job.id}. Starting processing...")
                    await self.semaphore.acquire()
                    # Fire and forget (tracker is inside _process_job)
                    asyncio.create_task(self._process_job(job))
                else:
                    # No jobs, sleep
                    await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in main polling loop: {e}")
                await asyncio.sleep(1)

    async def _process_job(self, job: IndexingJob):
        try:
            logger.info(f"Processing job {job.id} for video {job.video_id}")

            await self.orchestrator.run_pipeline(
                project_id=job.project_id,
                video_id=job.video_id,
            )
            
            await self.job_repository.update_status(job.id, IndexingJobStatus.DONE)
            logger.info(f"Job {job.id} completed successfully.")
            
        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}", exc_info=True)
            await self.job_repository.update_status(job.id, IndexingJobStatus.FAILED, error_message=str(e))
        finally:
            self.semaphore.release()

    def stop(self):
        self.running = False
