from typing import Optional
from dataclasses import dataclass

@dataclass
class SceneObject:
    name: str
    detail: Optional[str] = None
