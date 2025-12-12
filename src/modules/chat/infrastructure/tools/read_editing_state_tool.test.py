import json
import os

import pytest

from src.modules.chat.config import PROJECT_ID
from .read_editing_state_tool import ReadEditingStateTool


def test_read_editing_state_tool():
    tool = ReadEditingStateTool()
    result = tool.call()
    print(result)

if __name__ == "__main__":
    test_read_editing_state_tool()