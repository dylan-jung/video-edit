from typing import Optional
from datetime import datetime, timezone, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorClientSession
from pymongo import ReturnDocument

from src.modules.indexing.domain.video import Video
from src.modules.indexing.application.ports.video_repository_port import VideoRepositoryPort


class MongoVideoRepository(VideoRepositoryPort):
    """MongoDB 기반 비디오 저장소"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db.videos
    
    async def save_in_transaction(
        self, 
        video: Video, 
        session: AsyncIOMotorClientSession
    ) -> Video:
        """트랜잭션 내에서 비디오 저장"""
        doc = video.to_dict()
        doc["_id"] = video.id
        await self.collection.insert_one(doc, session=session)
        return video
    
    async def find_by_id(self, video_id: str) -> Optional[Video]:
        """ID로 비디오 조회"""
        doc = await self.collection.find_one({"_id": video_id})
        if doc:
            doc["id"] = doc.pop("_id")
            return Video.from_dict(doc)
        return None
    
    async def update(self, video: Video) -> None:
        """비디오 업데이트"""
        doc = video.to_dict()
        doc.pop("id")  # _id는 업데이트하지 않음
        await self.collection.update_one(
            {"_id": video.id},
            {"$set": doc}
        )
