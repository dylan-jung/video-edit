from fastapi import APIRouter, HTTPException, Body, Depends
import uuid

from src.modules.indexing.api.dto.presigned_url_response import PresignedUrlResponse
from src.modules.indexing.api.dto.job_request import IndexingJobCreateRequest
from src.modules.indexing.domain.indexing_job import IndexingJob, IndexingJobStatus
from src.modules.indexing.dependencies import get_file_storage_repository, get_indexing_job_repository
from src.shared.application.ports.file_storage import FileStoragePort
from src.modules.indexing.application.ports.job_repository_port import JobRepositoryPort

router = APIRouter()

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

@router.post("/projects/{project_id}/videos/{video_id}/jobs", status_code=202)
async def create_indexing_job(
    project_id: str,
    video_id: str,
    repository: JobRepositoryPort = Depends(get_indexing_job_repository)
):
    """
    Triggers an indexing job by storing it in the Outbox (MongoDB).
    """

    try:
        # Check for existing active job
        active_job = await repository.find_active_job_by_video_id(video_id)
        if active_job:
            raise HTTPException(
                status_code=409,
                detail=f"Job for video {video_id} is already {active_job.status}"
            )

        job = IndexingJob(
            id=str(uuid.uuid4()),
            project_id=project_id,
            video_id=video_id,
            content_type="video/mp4",
            status=IndexingJobStatus.PENDING
        )
        
        created_job = await repository.create_job(job)
        return {"job_id": created_job.id, "status": created_job.status}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
