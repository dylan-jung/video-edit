import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Timeline:
    clips: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Project:
    timeline: Timeline = field(default_factory=Timeline)

    def to_dict(self) -> dict:
        return {
            "timeline": {
                "clips": self.timeline.clips
            }
        }

    @staticmethod
    def from_dict(data: dict) -> 'Project':
        timeline_data = data.get("timeline", {})
        clips = timeline_data.get("clips", [])
        return Project(timeline=Timeline(clips=clips))
