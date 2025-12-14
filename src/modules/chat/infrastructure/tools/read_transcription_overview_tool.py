import json
from typing import Annotated, Any, Dict, Union

from langchain_core.tools import StructuredTool, Tool
from langgraph.prebuilt import InjectedState

from src.shared.infrastructure.storage.gcp_video_repository import GCPVideoRepository
from src.shared.application.ports.video_repository import VideoRepositoryPort
from src.shared.utils.get_video_id import get_video_id


class ReadVideoTranscriptionOverviewTool:
    name = "read_video_transcription_overview"
    description = (
        "Read a simplified transcription overview for a specific video. "
        "Returns the transcribed speech content as plain text without timestamp information. "
        "This is useful for quickly understanding the spoken content and dialogue in the video. "
        "Note that transcriptions are AI-generated and may not be 100% accurate. "
        "For detailed transcription with timestamps, use read_video_transcription instead. "
        "Input: video_id (str) - the id of the video to read transcription overview from "
        "Output: transcription_overview (str) - the transcription overview of the video"
    )

    def __init__(self):
        self.repository: VideoRepositoryPort = GCPVideoRepository()

    def call(self, video_id: str, project_id: str) -> str:
        transcription_bytes = self.repository.get_video_transcription(project_id, video_id)
        transcription_json = json.loads(transcription_bytes)
        transcription_overview = transcription_json["transcription"]
        return transcription_overview
        
    def as_tool(self) -> StructuredTool:
        """Convert the tool to a LangGraph-compatible tool format."""
        def tool_func(video_id: str, project_id: Annotated[str, InjectedState("project_id")]) -> str:
            return self.call(video_id, project_id)
        
        return StructuredTool.from_function(
            func=tool_func,
            name=self.name,
            description=self.description,
        )
