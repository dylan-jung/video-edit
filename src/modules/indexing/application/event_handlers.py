import logging
from typing import Dict, Any

from src.modules.indexing.application.orchestrator import PipelineOrchestrator
from src.modules.indexing.application.ports.video_repository_port import VideoRepositoryPort

logger = logging.getLogger(__name__)


class VideoEventHandlers:
    """비디오 도메인 이벤트 핸들러"""
    
    def __init__(
        self, 
        orchestrator: PipelineOrchestrator,
        video_repo: VideoRepositoryPort
    ):
        self.orchestrator = orchestrator
        self.video_repo = video_repo
    
    async def handle_video_uploaded(self, payload: Dict[str, Any]):
        """VIDEO_UPLOADED 이벤트 처리"""
        video_id = payload["video_id"]
        project_id = payload["project_id"]
        
        logger.info(f"Handling VideoUploadedEvent for video {video_id}")
        
        try:
            # 비디오 상태를 INDEXING으로 변경
            video = await self.video_repo.find_by_id(video_id)
            if video:
                video.mark_as_indexing()
                await self.video_repo.update(video)
            
            # 인덱싱 파이프라인 실행
            await self.orchestrator.run_pipeline(
                project_id=project_id,
                video_id=video_id
            )
            
            # 성공 시 비디오 상태를 INDEXED로 변경
            video = await self.video_repo.find_by_id(video_id)
            if video:
                video.mark_as_indexed()
                await self.video_repo.update(video)
            
            logger.info(f"Video {video_id} indexed successfully")
            
        except Exception as e:
            logger.error(f"Failed to index video {video_id}: {e}", exc_info=True)
            
            # 실패 시 비디오 상태를 FAILED로 변경
            video = await self.video_repo.find_by_id(video_id)
            if video:
                video.mark_as_failed(str(e))
                await self.video_repo.update(video)
            
            raise  # 재시도를 위해 예외 재발생
