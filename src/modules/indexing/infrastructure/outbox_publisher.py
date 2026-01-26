import asyncio
import logging
from typing import Dict, Callable, Any

from src.modules.indexing.application.ports.outbox_repository_port import OutboxRepositoryPort
from src.modules.indexing.domain.outbox_message import OutboxMessage

logger = logging.getLogger(__name__)


class OutboxPublisher:
    """
    Outbox 메시지를 폴링하여 이벤트 핸들러에게 전달
    
    Transactional Outbox Pattern의 Publisher 컴포넌트
    """
    
    def __init__(
        self,
        outbox_repo: OutboxRepositoryPort,
        event_handlers: Dict[str, Callable],
        poll_interval: float = 1.0,
        max_concurrency: int = 5
    ):
        self.outbox_repo = outbox_repo
        self.event_handlers = event_handlers
        self.poll_interval = poll_interval
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.running = False
    
    async def start(self):
        """Publisher 시작"""
        self.running = True
        logger.info("Starting OutboxPublisher...")
        
        while self.running:
            try:
                # 용량 확인
                if self.semaphore.locked():
                    await asyncio.sleep(0.5)
                    continue
                
                # Outbox에서 메시지 획득
                message = await self.outbox_repo.acquire_next_pending_message()
                
                if message:
                    logger.info(f"Publishing event {message.event_type} for {message.aggregate_id}")
                    await self.semaphore.acquire()
                    asyncio.create_task(self._publish_message(message))
                else:
                    # 메시지 없음, 대기
                    await asyncio.sleep(self.poll_interval)
                    
            except Exception as e:
                logger.error(f"Error in publisher loop: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _publish_message(self, message: OutboxMessage):
        """메시지 발행 및 핸들러 호출"""
        try:
            # 이벤트 타입에 맞는 핸들러 찾기
            handler = self.event_handlers.get(message.event_type)
            
            if not handler:
                error_msg = f"No handler for event type: {message.event_type}"
                logger.error(error_msg)
                await self.outbox_repo.mark_as_failed_with_retry(
                    message.id, 
                    error_msg
                )
                return
            
            # 핸들러 실행 (예: 인덱싱 워커 호출)
            await handler(message.payload)
            
            # 성공 시 PROCESSED로 표시
            await self.outbox_repo.mark_as_processed(message.id)
            logger.info(f"Event {message.id} processed successfully")
            
        except Exception as e:
            error_msg = f"Failed to publish message {message.id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # 재시도 스케줄링
            await self.outbox_repo.mark_as_failed_with_retry(
                message.id,
                error_msg
            )
            
        finally:
            self.semaphore.release()
    
    def stop(self):
        """Publisher 중지"""
        self.running = False
        logger.info("OutboxPublisher stopped")
