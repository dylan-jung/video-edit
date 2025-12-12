from pydantic import BaseModel

class IndexingRequest(BaseModel):
    pass # No body needed for now as params are in path, but good for future ext
