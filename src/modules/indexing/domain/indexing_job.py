from enum import Enum, auto
from datetime import datetime, timezone
from typing import Optional, Any, Dict
from dataclasses import dataclass, field


class IndexingJobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class IndexingJob:
    id: str  # MongoDB _id
    project_id: str
    video_id: str
    content_type: str
    status: IndexingJobStatus = IndexingJobStatus.PENDING
    error_message: Optional[str] = None
    retry_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def model_dump(self) -> Dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "video_id": self.video_id,
            "content_type": self.content_type,
            "status": self.status.value,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
