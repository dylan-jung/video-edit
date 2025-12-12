from typing import Optional
from pydantic import BaseModel

class SceneObject(BaseModel):
    name: str
    detail: Optional[str] = None
