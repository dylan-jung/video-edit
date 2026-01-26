import logging
import json
from typing import Optional
import uuid
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorClient

from src.modules.chat.api.dto.chat_request import ChatRequest
from src.modules.chat.api.dto.presigned_url_response import PresignedUrlResponse
from src.modules.chat.application.service import AgentService
from src.modules.chat.dependencies import get_agent_service
from src.modules.indexing.domain.video import Video, VideoStatus
from src.modules.indexing.domain.domain_events import VideoUploadedEvent
from src.modules.indexing.domain.outbox_message import OutboxMessage
from src.modules.indexing.dependencies import (
    get_file_storage_repository,
    get_db_client,
    get_video_repository,
    get_outbox_repository
)
from src.shared.application.ports.file_storage import FileStoragePort
from src.modules.indexing.application.ports.video_repository_port import VideoRepositoryPort
from src.modules.indexing.application.ports.outbox_repository_port import OutboxRepositoryPort

router = APIRouter()
logger = logging.getLogger(__name__)

# ============ Chat Endpoints ============

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


# ============ Video Upload Endpoints ============

@router.post("/projects/{project_id}/video/presigned-url", response_model=PresignedUrlResponse)
async def get_presigned_urls(
    project_id: str,
    repository: FileStoragePort = Depends(get_file_storage_repository)
):
    """
    Generate pre-signed URLs for uploading video artifacts.
    """
    try:
        video_id = str(uuid.uuid4())
        content_type = "video/mp4"
        object_name = f"projects/{project_id}/videos/{video_id}.mp4"
        url = repository.generate_upload_signed_url(
            object_name=object_name,
            content_type=content_type
        )
        return PresignedUrlResponse(
            project_id=project_id,
            video_id=video_id,
            url=url
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class VideoUploadRequest(BaseModel):
    file_path: str = Field(..., description="Path to uploaded video file in storage")


@router.post("/projects/{project_id}/videos/{video_id}/upload", status_code=201)
async def upload_video(
    project_id: str,
    video_id: str,
    request: VideoUploadRequest,
    db_client: AsyncIOMotorClient = Depends(get_db_client),
    video_repo: VideoRepositoryPort = Depends(get_video_repository),
    outbox_repo: OutboxRepositoryPort = Depends(get_outbox_repository)
):
    """
    비디오 업로드 처리
    
    비즈니스 데이터(Video)와 이벤트(Outbox)를 동일한 트랜잭션에서 저장하여
    원자성을 보장합니다.
    """
    
    try:
        # 트랜잭션 시작
        async with await db_client.start_session() as session:
            async with session.start_transaction():
                # 1. 비즈니스 엔티티 생성 및 저장
                video = Video(
                    id=video_id,
                    project_id=project_id,
                    file_path=request.file_path,
                    content_type="video/mp4",
                    status=VideoStatus.UPLOADED
                )
                await video_repo.save_in_transaction(video, session)
                
                # 2. 도메인 이벤트 생성
                event = VideoUploadedEvent(
                    video_id=video_id,
                    project_id=project_id,
                    file_path=request.file_path
                )
                
                # 3. Outbox에 이벤트 저장 (같은 트랜잭션)
                outbox_message = OutboxMessage(
                    id=event.event_id,
                    event_type=event.event_type.value,
                    aggregate_id=video_id,
                    payload=event.payload
                )
                await outbox_repo.save_in_transaction(outbox_message, session)
                
                # 트랜잭션 커밋 (자동)
        
        return {
            "video_id": video_id,
            "status": video.status.value,
            "message": "Video uploaded successfully. Indexing will start shortly."
        }
        
    except Exception as e:
        logger.error(f"Failed to upload video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/videos/{video_id}/status")
async def get_video_status(
    project_id: str,
    video_id: str,
    video_repo: VideoRepositoryPort = Depends(get_video_repository)
):
    """비디오 상태 조회"""
    video = await video_repo.find_by_id(video_id)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return {
        "video_id": video.id,
        "project_id": video.project_id,
        "status": video.status.value,
        "uploaded_at": video.uploaded_at.isoformat(),
        "indexed_at": video.indexed_at.isoformat() if video.indexed_at else None,
        "error_message": video.error_message
    }
