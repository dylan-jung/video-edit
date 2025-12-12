from typing import List
from pydantic import BaseModel
from .track import Track

class EditingState(BaseModel):
    project_id: str
    tracks: List[Track]
