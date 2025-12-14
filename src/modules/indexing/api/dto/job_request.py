from pydantic import BaseModel

class IndexingJobCreateRequest(BaseModel):
    project_id: str
    video_id: str
    content_type: str