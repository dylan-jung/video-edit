from pydantic import BaseModel, Field
from typing import Optional

class IndexingJobMessage(BaseModel):
    """
    Message payload for indexing-jobs Pub/Sub topic
    """
    bucket: str = Field(..., description="GCS Bucket Name")
    object_name: str = Field(..., description="GCS Object Path (name)")
    content_type: str = Field(..., description="File Content Type")
    size: int = Field(..., description="File Size in bytes")
    
    # Trace / Correlation
    request_id: Optional[str] = Field(None, description="Unique Request ID for tracking")
    callback_url: Optional[str] = Field(None, description="Callback URL for status updates")
    
    # Metadata
    project_id: Optional[str] = Field(None, description="Project ID derived from path")
    video_id: Optional[str] = Field(None, description="Video ID derived from path")
