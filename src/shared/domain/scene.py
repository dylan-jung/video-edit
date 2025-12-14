from typing import List
from dataclasses import dataclass
from .scene_object import SceneObject
from .scene_highlight import SceneHighlight

@dataclass
class Scene:
    background: str
    objects: List[SceneObject]
    ocr_text: List[str]
    actions: List[str]
    emotions: List[str]
    context: str
    highlight: List[SceneHighlight]
    start_time: str
    end_time: str
