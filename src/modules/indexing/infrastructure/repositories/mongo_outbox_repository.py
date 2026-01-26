from typing import Optional
from datetime import datetime, timezone, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorClientSession
from pymongo import ReturnDocument

from src.modules.indexing.domain.outbox_message import OutboxMessage, OutboxStatus
from src.modules.indexing.application.ports.outbox_repository_port import OutboxRepositoryPort


class MongoOutboxRepository(OutboxRepositoryPort):
    """MongoDB 기반 Outbox 저장소"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db.outbox_messages
    
    async def save_in_transaction(
        self, 
        message: OutboxMessage, 
        session: AsyncIOMotorClientSession
    ) -> OutboxMessage:
        """트랜잭션 내에서 Outbox 메시지 저장"""
        doc = message.to_dict()
        doc["_id"] = message.id
        await self.collection.insert_one(doc, session=session)
        return message
    
    async def acquire_next_pending_message(self) -> Optional[OutboxMessage]:
        """다음 처리할 메시지를 원자적으로 획득"""
        now = datetime.now(timezone.utc)
        
        doc = await self.collection.find_one_and_update(
            {
                "status": OutboxStatus.PENDING.value,
                "$or": [
                    {"next_retry_at": None},
                    {"next_retry_at": {"$lte": now}}
                ]
            },
            {
                "$set": {
                    "status": OutboxStatus.PROCESSING.value,
                    "updated_at": now
                }
            },
            sort=[("created_at", 1)],  # FIFO
            return_document=ReturnDocument.AFTER
        )
        
        if doc:
            doc["id"] = doc.pop("_id")
            return OutboxMessage.from_dict(doc)
        return None
    
    async def mark_as_processed(self, message_id: str) -> None:
        """메시지 처리 완료 표시"""
        await self.collection.update_one(
            {"_id": message_id},
            {
                "$set": {
                    "status": OutboxStatus.PROCESSED.value,
                    "processed_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
    
    async def mark_as_failed_with_retry(
        self, 
        message_id: str,
        error_message: str,
        max_retries: int = 3,
        backoff_seconds: int = 60
    ) -> None:
        """실패 표시 및 재시도 스케줄링"""
        message_doc = await self.collection.find_one({"_id": message_id})
        if not message_doc:
            return
        
        retry_count = message_doc.get("retry_count", 0) + 1
        
        if retry_count >= max_retries:
            # 최대 재시도 초과 → FAILED (DLQ)
            await self.collection.update_one(
                {"_id": message_id},
                {
                    "$set": {
                        "status": OutboxStatus.FAILED.value,
                        "retry_count": retry_count,
                        "error_message": error_message,
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
        else:
            # 재시도 스케줄링 (Exponential Backoff)
            next_retry = datetime.now(timezone.utc) + timedelta(
                seconds=backoff_seconds * (2 ** (retry_count - 1))
            )
            await self.collection.update_one(
                {"_id": message_id},
                {
                    "$set": {
                        "status": OutboxStatus.PENDING.value,
                        "retry_count": retry_count,
                        "next_retry_at": next_retry,
                        "error_message": error_message,
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
    
    async def reset_stale_messages(self, timeout_minutes: int = 30) -> int:
        """
        Visibility Timeout: PROCESSING 상태로 오래 남아있는 메시지를 PENDING으로 복구
        
        Returns:
            복구된 메시지 수
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)
        
        result = await self.collection.update_many(
            {
                "status": OutboxStatus.PROCESSING.value,
                "updated_at": {"$lt": cutoff_time}
            },
            {
                "$set": {
                    "status": OutboxStatus.PENDING.value,
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        return result.modified_count
