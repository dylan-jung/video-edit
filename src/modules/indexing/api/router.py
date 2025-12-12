from fastapi import APIRouter, HTTPException, Body, Depends
import uuid
from src.modules.indexing.api.dto.presigned_url_request import PresignedUrlRequest
from src.modules.indexing.api.dto.presigned_url_response import PresignedUrlResponse
from src.modules.indexing.api.dto.job_request import IndexingJobCreateRequest
from src.modules.indexing.domain.indexing_job import IndexingJob, IndexingJobStatus
from src.shared.infrastructure.repository.storage import CloudStorageRepository
from src.modules.indexing.infrastructure.dependencies import get_cloud_storage_repository, get_indexing_job_repository
from src.modules.indexing.infrastructure.repositories.job_repository import IndexingJobRepository

router = APIRouter()

@router.post("/projects/{project_id}/videos/{video_id}/presigned-url", response_model=PresignedUrlResponse)
async def get_presigned_urls(
    project_id: str,
    video_id: str,
    request: PresignedUrlRequest = Body(...),
    repository: CloudStorageRepository = Depends(get_cloud_storage_repository)
):
    """
    Generate pre-signed URLs for uploading video artifacts.
    """
    try:
        urls = {}
        
        # Define content types for known files to ensure correct handling
        content_type_map = {
            "video.mp4": "video/mp4",
            "audio.wav": "audio/wav",
            "metadata.json": "application/json",
            "scenes.json": "application/json"
        }

        for filename in request.files:
            content_type = content_type_map.get(filename, "application/octet-stream")
            url = repository.generate_upload_signed_url(
                project_id, 
                video_id, 
                filename, 
                content_type=content_type
            )
            urls[filename] = url
        
        return PresignedUrlResponse(
            project_id=project_id,
            video_id=video_id,
            urls=urls
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/videos/{video_id}/jobs", status_code=202)
async def create_indexing_job(
    project_id: str,
    video_id: str,
    request: IndexingJobCreateRequest = Body(...),
    repository: IndexingJobRepository = Depends(get_indexing_job_repository)
):
    """
    Triggers an indexing job by storing it in the Outbox (MongoDB).
    """
    try:
        job = IndexingJob(
            id=str(uuid.uuid4()),
            project_id=project_id,
            video_id=video_id,
            bucket=request.bucket,
            object_name=request.object_name,
            content_type=request.content_type,
            status=IndexingJobStatus.PENDING
        )
        
        created_job = await repository.create_job(job)
        return {"job_id": created_job.id, "status": created_job.status}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
