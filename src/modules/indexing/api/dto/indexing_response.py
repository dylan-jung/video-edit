from pydantic import BaseModel

class IndexingResponse(BaseModel):
    project_id: str
    video_id: str
    status: str
    message: str
