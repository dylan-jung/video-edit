from abc import ABC, abstractmethod
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClientSession

from src.modules.indexing.domain.outbox_message import OutboxMessage


class OutboxRepositoryPort(ABC):
    """Outbox 저장소 포트"""
    
    @abstractmethod
    async def save_in_transaction(
        self, 
        message: OutboxMessage, 
        session: AsyncIOMotorClientSession
    ) -> OutboxMessage:
        """트랜잭션 내에서 Outbox 메시지 저장"""
        pass
    
    @abstractmethod
    async def acquire_next_pending_message(self) -> Optional[OutboxMessage]:
        """다음 처리할 메시지를 원자적으로 획득"""
        pass
    
    @abstractmethod
    async def mark_as_processed(self, message_id: str) -> None:
        """메시지 처리 완료 표시"""
        pass
    
    @abstractmethod
    async def mark_as_failed_with_retry(
        self, 
        message_id: str,
        error_message: str,
        max_retries: int = 3,
        backoff_seconds: int = 60
    ) -> None:
        """실패 표시 및 재시도 스케줄링"""
        pass
