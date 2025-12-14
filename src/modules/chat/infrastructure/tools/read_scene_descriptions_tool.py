import json
from typing import Any, Dict, Union

from langchain_core.tools import Tool

from src.shared.infrastructure.storage.gcp_video_repository import GCPVideoRepository
from src.shared.infrastructure.storage.repository import Repository
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
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.repository: Repository = GCPVideoRepository()

    def _get_scene_descriptions(self, video_id: str) -> str:
        with open(f"projects/{self.project_id}/{video_id}/scene_descriptions.json", "r") as f:
            scene_descriptions = json.load(f)
        # scene_descriptions_bytes = self.repository.get_video_scene_descriptions(self.project_id, video_id)
        # scene_descriptions = json.loads(scene_descriptions_bytes)
        return scene_descriptions

    def call(self, video_id: str) -> str:
        scene_descriptions = self._get_scene_descriptions(video_id)
        return json.dumps(scene_descriptions, ensure_ascii=False)
        
    def as_tool(self) -> Tool:
        """Convert the tool to a LangGraph-compatible tool format."""
        def tool_func(*args, **kwargs) -> str:
            # LangChain이 video_id를 args[0]이나 kwargs에서 전달할 수 있음
            video_id = args[0] if args else kwargs.get('video_id', '')
            return self.call(video_id)
        
        return Tool(
            name=self.name,
            description=self.description,
            func=tool_func
        )
