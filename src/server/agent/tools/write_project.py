import json
from typing import Any, Dict, Union

from src.repository.cloud_storage_repository import CloudStorageRepository
from src.repository.repository import Repository
from src.server.agent.config import BASE_PATH, PROJECT_ID


class WriteProjectTool:
    """
    프로젝트 파일을 쓰기 위한 도구입니다.
    """
    name = "write_project"
    description = (
        "Write project file",
        "Project file is a JSON file that contains video editing project information. this file is used to drive the video editing process. so you can use this tool to get the current state of the project.",
        "Input: 'diff of project file(JSON)' so that less intelligent LLM can edit the project file.",
        "Output: 'success' if the project file is written successfully."
    )

    parameters = {
        "type": "object",
        "properties": {
            "diff": {
                "type": "string",
                "description": "diff of project file(JSON)"
            }
        },
        "required": ["diff"]
    }

    def call(self, params: Union[str, Dict[str, Any]], **kwargs) -> str:
        """
        Call the tool with parameters.
        
        Args:
            params: Either a JSON string (Qwen-Agent) or a dict (LangGraph)
            
        Returns:
            Success message
        """
        # Handle both string (Qwen-Agent) and dict (LangGraph) params
        if isinstance(params, str):
            args = json.loads(params)
        else:
            args = params
            
        diff: str = args["diff"]
        print(diff)

        with open(f"{BASE_PATH}/project.json", "w") as f:
            f.write(json.dumps(diff, ensure_ascii=False))

        return "success"

        # repository: Repository = CloudStorageRepository()
        # project_bytes = repository.get_project(PROJECT_ID)
        # project = json.loads(project_bytes)

        # return json.dumps(project, ensure_ascii=False)
        
    @staticmethod
    def as_tool() -> Dict[str, Any]:
        """Convert the tool to a LangGraph-compatible tool format."""
        return {
            "type": "function",
            "function": {
                "name": WriteProjectTool.name,
                "description": "\n".join(WriteProjectTool.description),
                "parameters": WriteProjectTool.parameters
            }
        }