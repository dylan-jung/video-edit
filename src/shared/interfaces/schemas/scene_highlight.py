from pydantic import BaseModel

class SceneHighlight(BaseModel):
    time: str
    note: str
