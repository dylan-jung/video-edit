from typing import Optional
from datetime import datetime
from pymongo import ReturnDocument
from src.modules.indexing.domain.indexing_job import IndexingJob, IndexingJobStatus
from src.shared.infrastructure.db import get_db
from src.modules.indexing.application.ports.job_repository_port import JobRepositoryPort

class MongoIndexingJobRepository(JobRepositoryPort):
    def __init__(self):
        self.db = get_db()
        self.collection = self.db.index_jobs

    async def create_job(self, job: IndexingJob) -> IndexingJob:
        job_dict = job.model_dump(by_alias=True)
        job_dict["_id"] = job.id
        await self.collection.insert_one(job_dict)
        return job

    async def acquire_next_pending_job(self) -> Optional[IndexingJob]:
        """
        Finds a PENDING job and atomically locks it by setting status to PROCESSING.
        """
        job_doc = await self.collection.find_one_and_update(
            {"status": IndexingJobStatus.PENDING.value},
            {
                "$set": {
                    "status": IndexingJobStatus.PROCESSING.value,
                    "updated_at": datetime.utcnow()
                }
            },
            sort=[("created_at", 1)],  # FIFO
            return_document=ReturnDocument.AFTER
        )
        
        if job_doc:
            return IndexingJob(**job_doc)
        return None

    async def update_status(self, job_id: str, status: IndexingJobStatus, error_message: str = None):
        update_data = {
            "status": status.value,
            "updated_at": datetime.utcnow()
        }
        if error_message:
            update_data["error_message"] = error_message
            
        await self.collection.update_one(
            {"_id": job_id},
            {"$set": update_data}
        )
        
    async def get_job(self, job_id: str) -> Optional[IndexingJob]:
        doc = await self.collection.find_one({"_id": job_id})
        if doc:
            return IndexingJob(**doc)
        return None
