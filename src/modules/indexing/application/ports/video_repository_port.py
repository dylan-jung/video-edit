from abc import ABC, abstractmethod
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClientSession

from src.modules.indexing.domain.video import Video


class VideoRepositoryPort(ABC):
    """비디오 저장소 포트"""
    
    @abstractmethod
    async def save_in_transaction(
        self, 
        video: Video, 
        session: AsyncIOMotorClientSession
    ) -> Video:
        """트랜잭션 내에서 비디오 저장"""
        pass
    
    @abstractmethod
    async def find_by_id(self, video_id: str) -> Optional[Video]:
        """ID로 비디오 조회"""
        pass
    
    @abstractmethod
    async def update(self, video: Video) -> None:
        """비디오 업데이트"""
        pass
