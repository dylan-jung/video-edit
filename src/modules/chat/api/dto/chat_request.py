from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = Field(default=None, description="Session ID. If empty, a new session is created.")