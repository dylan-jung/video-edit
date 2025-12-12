import logging
import json
from pydantic import BaseModel
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.modules.chat.application.service import AgentService

router = APIRouter()
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    message: str
    thread_id: str

@router.post("/projects/{project_id}/chat")
async def chat_endpoint(project_id: str, request: ChatRequest):
    agent_service = AgentService(project_id=project_id)
    
    async def event_generator():
        try:
            async for chunk in agent_service.chat(request.message, request.thread_id):
                # Standard SSE format: "data: <json>\n\n"
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
