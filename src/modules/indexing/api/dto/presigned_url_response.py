from typing import Dict
from pydantic import BaseModel

class PresignedUrlResponse(BaseModel):
    project_id: str
    video_id: str
    urls: Dict[str, str]
