from abc import ABC, abstractmethod
from typing import Optional
from src.modules.indexing.domain.indexing_job import IndexingJob, IndexingJobStatus

class JobRepositoryPort(ABC):
    @abstractmethod
    async def create_job(self, job: IndexingJob) -> IndexingJob:
        pass

    @abstractmethod
    async def acquire_next_pending_job(self) -> Optional[IndexingJob]:
        pass

    @abstractmethod
    async def update_status(self, job_id: str, status: IndexingJobStatus, error_message: str = None):
        pass

    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[IndexingJob]:
        pass

    @abstractmethod
    async def find_active_job_by_video_id(self, video_id: str) -> Optional[IndexingJob]:
        pass
