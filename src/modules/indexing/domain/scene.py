from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SceneObject:
    label: str
    confidence: float = 0.0
    box_2d: Optional[List[int]] = None

@dataclass
class SceneHighlight:
    start_time: str
    end_time: str
    description: str

@dataclass
class Scene:
    """
    Represents a analyzed scene from a video.
    """
    start_time: str
    end_time: str
    context: str
    background: str = ""
    objects: List[SceneObject] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    emotions: List[str] = field(default_factory=list)
    ocr_text: List[str] = field(default_factory=list)
    highlight: List[SceneHighlight] = field(default_factory=list)
