from typing import List
from dataclasses import dataclass
from .track import Track

@dataclass
class EditingState:
    project_id: str
    tracks: List[Track]
