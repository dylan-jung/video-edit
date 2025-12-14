import logging
import json
from typing import Optional
import uuid
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from src.modules.chat.application.service import AgentService
from src.modules.chat.dependencies import get_agent_service

router = APIRouter()
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = Field(default=None, description="Session ID. If empty, a new session is created.")

@router.post("/projects/{project_id}/chat")
async def chat_endpoint(
    project_id: str, 
    request: ChatRequest,
    agent_service: AgentService = Depends(get_agent_service)
):
    thread_id = request.thread_id or str(uuid.uuid4())
    
    async def event_generator():
        try:
            # Send the thread_id as the first event so client knows the session ID
            yield f"data: {json.dumps({'type': 'session_init', 'thread_id': thread_id})}\n\n"
            
            async for chunk in agent_service.chat(request.message, thread_id):
                # Standard SSE format: "data: <json>\n\n"
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
