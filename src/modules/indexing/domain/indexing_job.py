from enum import Enum, auto
from datetime import datetime
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field

class IndexingJobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"

class IndexingJob(BaseModel):
    id: str  # MongoDB _id or a separate UUID
    project_id: str
    video_id: str
    bucket: str
    object_name: str
    content_type: str
    status: IndexingJobStatus = IndexingJobStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    
    # Optional: Retry count
    retry_count: int = 0
    
    class Config:
        from_attributes = True
