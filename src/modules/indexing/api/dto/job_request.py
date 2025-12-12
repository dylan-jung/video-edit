from pydantic import BaseModel

class IndexingJobCreateRequest(BaseModel):
    bucket: str
    object_name: str
    content_type: str
