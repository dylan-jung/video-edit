import json
from typing import Any, Dict, Union

from src.repository.cloud_storage_repository import CloudStorageRepository
from src.repository.repository import Repository
from src.server.agent.config import PROJECT_ID


class ReadEditingStateTool:
    """
    현재 프로젝트의 프로젝트 파일을 읽어 옵니다.
    """
    name = "read_editing_state"
    description = (
        "Return editing state (as JSON) for a video.",
        "Editing state is a JSON file that contains video editing project information. this file is used to drive the video editing process. so you can use this tool to get the current state of the editing process.",
        "Input: none",
        "Output: editing state (as JSON)"
    )

    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    def call(self) -> str:
        repository: Repository = CloudStorageRepository()
        project_bytes = repository.get_project(PROJECT_ID)
        project = json.loads(project_bytes)

        return json.dumps(project, ensure_ascii=False)
        
    @staticmethod
    def as_tool() -> Dict[str, Any]:
        """Convert the tool to a LangGraph-compatible tool format."""
        return {
            "type": "function",
            "function": {
                "name": ReadEditingStateTool.name,
                "description": "\n".join(ReadEditingStateTool.description),
                "parameters": ReadEditingStateTool.parameters
            }
        }
