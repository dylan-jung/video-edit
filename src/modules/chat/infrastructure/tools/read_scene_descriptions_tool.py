import json
from typing import Annotated, Any, Dict, Union

from langchain_core.tools import StructuredTool, Tool
from langgraph.prebuilt import InjectedState

from src.shared.infrastructure.storage.gcp_video_repository import GCPVideoRepository
from src.shared.application.ports.video_repository import VideoRepositoryPort
# from src.modules.chat.config import PROJECT_ID
from src.shared.utils.get_video_id import get_video_id


class ReadVideoSceneDescriptionsTool:
    """
    Cloud Storage 에 저장된 동영상의 장면 설명(scene_descriptions, JSON)을 읽어 옵니다.
    """
    name = "read_video_scene_descriptions"
    description = (
        "Read scene descriptions for a specific video. "
        "Returns detailed scene-by-scene analysis as JSON, including visual descriptions, "
        "timestamps (start and end times), scene transitions, key objects, actions, "
        "and contextual information for each scene. This data is crucial for understanding "
        "video content structure and planning editing decisions. "
        "Input: video_id (str) - the id of the video to read scene descriptions from "
        "Output: scene_descriptions (JSON) - the scene descriptions of the video"
    )
    
    def __init__(self):
        self.repository: VideoRepositoryPort = GCPVideoRepository()

    def call(self, video_id: str, project_id: str) -> str:
        final_scene_descriptions = self.repository.get_video_scene_descriptions(project_id, video_id)
        return json.dumps(final_scene_descriptions, ensure_ascii=False)
        
    def as_tool(self) -> StructuredTool:
        """Convert the tool to a LangGraph-compatible tool format."""
        def tool_func(video_id: str, project_id: Annotated[str, InjectedState("project_id")]) -> str:
            return self.call(video_id, project_id)
        
        return StructuredTool.from_function(
            func=tool_func,
            name=self.name,
            description=self.description,
        )
