from datetime import datetime
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    project_id: str | None = None
    timestamp: str | None = None
    route: str | None = None
    
    plan: str | None = None
    plan_messages: Annotated[List[BaseMessage], add_messages] | None = None