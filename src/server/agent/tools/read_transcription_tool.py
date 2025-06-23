import json
from typing import Any, Dict, Union

from langchain_core.tools import Tool

from src.repository.cloud_storage_repository import CloudStorageRepository
from src.repository.repository import Repository
from src.server.agent.config import PROJECT_ID
from src.utils.get_video_id import get_video_id


class ReadVideoTranscriptionTool:
    """
    Cloud Storage 에 저장된 동영상의 트랜스크립션(JSON)을 읽어 옵니다.
    """
    name = "read_video_transcription"
    description = (
        "Read detailed transcription data for a specific video. "
        "Returns comprehensive transcription information as JSON, including the complete "
        "transcribed text with precise timestamps (start and end times) for each segment. "
        "This enables accurate synchronization between speech and video timeline for editing. "
        "Note that transcriptions are AI-generated and may not be 100% accurate. "
        "Input: video_id (str) - the id of the video to read transcription from "
        "Output: transcription (JSON) - the transcription of the video"
    )

    def __init__(self):
        self.repository: Repository = CloudStorageRepository()

    def call(self, video_id: str) -> str:
        with open(f"projects/{PROJECT_ID}/{video_id}/transcription.json", "r") as f:
            transcription_json = json.load(f)
        # transcription_bytes = self.repository.get_video_transcription(PROJECT_ID, video_id)
        # transcription_json = json.loads(transcription_bytes)
        transcription = transcription_json["transcription"] 

        return json.dumps(transcription, ensure_ascii=False)
        
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
