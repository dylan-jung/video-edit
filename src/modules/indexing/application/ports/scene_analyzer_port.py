from abc import ABC, abstractmethod
from typing import List
from src.modules.indexing.domain.scene import Scene

class SceneAnalyzerPort(ABC):
    @abstractmethod
    def analyze_scenes(self, video_path: str, chunk_duration: int = 300) -> List[Scene]:
        """
        Analyze video scenes and return a list of Scene objects.
        
        Args:
            video_path: Path to the video file
            chunk_duration: Duration of chunks in seconds
            
        Returns:
            List of Scene objects containing analysis results
        """
        pass
