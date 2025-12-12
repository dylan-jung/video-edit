from dataclasses import dataclass
from typing import Optional

@dataclass
class Video:
    """
    Represents a video entity in the indexing process.
    """
    project_id: str
    video_id: str
    path: str
    audio_path: Optional[str] = None

    @property
    def base_dir(self) -> str:
        """Returns the base directory for this video's resources."""
        # Using the convention projects/{project_id}/{video_id}
        return f"projects/{self.project_id}/{self.video_id}"
