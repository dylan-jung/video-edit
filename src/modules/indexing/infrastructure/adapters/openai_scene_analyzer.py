import os
from typing import List

from src.modules.indexing.domain.scene import Scene
from src.modules.indexing.application.ports.scene_analyzer_port import SceneAnalyzerPort
from src.modules.indexing.infrastructure.scene_analyzer.scene_analyzer import analyze_video_scenes

class OpenAISceneAnalyzer(SceneAnalyzerPort):
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self._check_api_key()

    def _check_api_key(self):
        if "OPENAI_API_KEY" not in os.environ:
             raise EnvironmentError("OPENAI_API_KEY missing - required for OpenAISceneAnalyzer")

    def run(self, video_path: str, chunk_duration: int = 300) -> List[Scene]:
        """
        Analyze video scenes using the refactored infrastructure service.
        Delegates to analyze_video_scenes which handles chunking and GPT analysis.
        """
        return analyze_video_scenes(
            video_path=video_path,
            model=self.model_name,
            chunk_duration=chunk_duration
        )
