import asyncio
import logging
from datetime import timedelta

from src.modules.indexing.infrastructure.repositories.mongo_outbox_repository import MongoOutboxRepository

logger = logging.getLogger(__name__)


class VisibilityTimeoutManager:
    """
    Visibility Timeout 관리자
    
    PROCESSING 상태로 오래 남아있는 메시지를 PENDING으로 복구하여
    워커 크래시 시에도 메시지가 재처리되도록 보장
    """
    
    def __init__(
        self, 
        outbox_repo: MongoOutboxRepository,
        timeout_minutes: int = 30,
        check_interval_seconds: int = 300  # 5분마다 체크
    ):
        self.outbox_repo = outbox_repo
        self.timeout_minutes = timeout_minutes
        self.check_interval_seconds = check_interval_seconds
        self.running = False
    
    async def start(self):
        """주기적으로 Stale 메시지 체크 시작"""
        self.running = True
        logger.info(f"Starting VisibilityTimeoutManager (timeout={self.timeout_minutes}min)...")
        
        while self.running:
            try:
                reset_count = await self.outbox_repo.reset_stale_messages(
                    timeout_minutes=self.timeout_minutes
                )
                
                if reset_count > 0:
                    logger.warning(f"Reset {reset_count} stale messages to PENDING")
                
            except Exception as e:
                logger.error(f"Error resetting stale messages: {e}", exc_info=True)
            
            await asyncio.sleep(self.check_interval_seconds)
    
    def stop(self):
        """Timeout Manager 중지"""
        self.running = False
        logger.info("VisibilityTimeoutManager stopped")
