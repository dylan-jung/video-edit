from typing import Optional
from datetime import datetime, timezone
from pymongo import ReturnDocument
from motor.motor_asyncio import AsyncIOMotorDatabase

from src.modules.indexing.domain.indexing_job import IndexingJob, IndexingJobStatus
from src.modules.indexing.application.ports.job_repository_port import JobRepositoryPort

class MongoIndexingJobRepository(JobRepositoryPort):
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db.index_jobs

    async def create_job(self, job: IndexingJob) -> IndexingJob:
        job_dict = job.model_dump()
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
                    "updated_at": datetime.now(timezone.utc)
                }
            },
            sort=[("created_at", 1)],  # FIFO
            return_document=ReturnDocument.AFTER
        )
        
        if job_doc:
            return self.doc_to_indexing_job(job_doc)
        return None

    async def update_status(self, job_id: str, status: IndexingJobStatus, error_message: str = None):
        update_data = {
            "status": status.value,
            "updated_at": datetime.now(timezone.utc)
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
            return self.doc_to_indexing_job(doc)
        return None

    async def find_active_job_by_video_id(self, video_id: str) -> Optional[IndexingJob]:
        """
        Finds a job for the given video_id that is either PENDING or PROCESSING.
        """
        doc = await self.collection.find_one({
            "video_id": video_id,
            "status": {"$in": [IndexingJobStatus.PENDING.value, IndexingJobStatus.PROCESSING.value]}
        })
        if doc:
            return self.doc_to_indexing_job(doc)
        return None

    def doc_to_indexing_job(self, doc: dict) -> IndexingJob:
        _id = doc.pop("_id", None)
        doc["id"] = _id
        return IndexingJob(**doc)
