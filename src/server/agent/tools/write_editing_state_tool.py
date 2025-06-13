import difflib
import json
import os
from typing import Any, Dict, Union

import jsonpatch
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.server.agent.config import BASE_PATH, PROJECT_ID


class WriteEditingStateInput(BaseModel):
    patch: str = Field(description="JSON patch to apply to the editing state")

class WriteEditingStateTool:
    """
    프로젝트 파일을 쓰기 위한 도구입니다.
    """
    name = "write_editing_state"
    description = (
        "Write or update the video editing state file with JSON patch. "
        "The editing state contains timeline information, clips, and configurations "
        "that drive the video editing process.\n"
        "JSON patch is a list of operations to apply to the editing state. "
        "You can use the following operations: add, remove, replace, move, copy, test. When you remove all tracks, you need to add an empty track.\n"
        "Editing state structure:\n"
        "{\n"
        "  \"project_id\": string,\n"
        "  \"tracks\": [\n"
        "    {\n"
        "      \"src\": string,       // video ID\n"
        # "      \"start\": string,     // start time in track (hh:mm:ss)\n"
        # "      \"end\": string,       // end time in track (hh:mm:ss)\n"
        "      \"duration\": string,  // playback duration (hh:mm:ss)\n"
        "      \"trimIn\": string,    // trim start from source (hh:mm:ss)\n"
        "      \"trimOut\": string    // trim end from source (hh:mm:ss)\n"
        "    }\n"
        "  ]\n"
        "}"
        "Input: patch (JSON patch to apply to the editing state)"
        "Output: None"
    )

    def _get_editing_state_path(self, project_id: str) -> str:
        """Get the editing state file path for the project."""
        return f"projects/{project_id}/editing_state.json"

    def call(self, patch: str) -> str:
        try:
            # Parse patch string to JSON (JSON Patch 형식)
            if not patch:
                return "Error: No patch provided"
            try:
                patch_data = json.loads(patch)
            except json.JSONDecodeError:
                return "Error: Invalid JSON format in patch"

            # Get the editing state file path
            editing_state_path = self._get_editing_state_path(PROJECT_ID)
            full_path = editing_state_path

            # Try to get existing editing state
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        current_state = json.load(f)
                except Exception as e:
                    return f"Error reading existing editing state: {str(e)}"
            else:
                print(f"No existing editing state found at {full_path}, creating new one")
                current_state = {
                    "project_id": PROJECT_ID,
                    "tracks": [],
                }
                os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # JSON Patch 적용
            try:
                patch_obj = jsonpatch.JsonPatch(patch_data)
                updated_state = patch_obj.apply(current_state, in_place=False)
            except Exception as e:
                return f"Error applying JSON Patch: {str(e)}"

            # Save updated state
            try:
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_state, f, ensure_ascii=False, indent=2)
            except Exception as e:
                return f"Error saving updated editing state: {str(e)}"

            return f"Successfully updated editing state at {full_path}"
        except Exception as e:
            return f"Error updating editing state: {str(e)}"

    def as_tool(self) -> StructuredTool:
        """Convert the tool to a LangGraph-compatible tool format."""        
        def tool_func(patch: str) -> str:
            return self.call(patch)
        
        return StructuredTool(
            name=self.name,
            description=self.description,
            func=tool_func,
            args_schema=WriteEditingStateInput
        )