"""
Indexing Worker - Pure Event-Driven Worker

이 모듈은 Outbox 메시지를 폴링하여 비디오 인덱싱을 처리하는 순수 워커입니다.
FastAPI 없이 백그라운드 태스크로만 동작합니다.
"""
import asyncio
import logging
import signal

from src.modules.indexing.infrastructure.outbox_publisher import OutboxPublisher
from src.modules.indexing.infrastructure.visibility_timeout_manager import VisibilityTimeoutManager
from src.modules.indexing.application.event_handlers import VideoEventHandlers
from src.modules.indexing.domain.domain_events import EventType

from src.modules.indexing.dependencies import (
    get_pipeline_orchestrator,
    get_outbox_repository,
    get_video_repository
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global references
publisher = None
timeout_manager = None
running = True


def signal_handler(signum, frame):
    """Graceful shutdown handler"""
    global running
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    running = False
    if publisher:
        publisher.stop()
    if timeout_manager:
        timeout_manager.stop()


async def main():
    """Main worker loop"""
    global publisher, timeout_manager
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("=" * 60)
        logger.info("Starting Indexing Worker")
        logger.info("=" * 60)
        
        # Initialize dependencies
        logger.info("Initializing dependencies...")
        orchestrator = get_pipeline_orchestrator()
        outbox_repo = get_outbox_repository()
        video_repo = get_video_repository()
        
        # Create event handlers
        event_handlers = VideoEventHandlers(orchestrator, video_repo)
        handlers_map = {
            EventType.VIDEO_UPLOADED.value: event_handlers.handle_video_uploaded
        }
        logger.info(f"Registered {len(handlers_map)} event handlers")
        
        # Initialize OutboxPublisher
        publisher = OutboxPublisher(
            outbox_repo=outbox_repo,
            event_handlers=handlers_map,
            poll_interval=1.0,
            max_concurrency=2
        )
        logger.info("OutboxPublisher initialized (poll_interval=1.0s, max_concurrency=2)")
        
        # Initialize VisibilityTimeoutManager
        timeout_manager = VisibilityTimeoutManager(
            outbox_repo=outbox_repo,
            timeout_minutes=30,
            check_interval_seconds=300  # 5분마다 체크
        )
        logger.info("VisibilityTimeoutManager initialized (timeout=30min, check_interval=5min)")
        
        # Start background tasks
        publisher_task = asyncio.create_task(publisher.start())
        timeout_task = asyncio.create_task(timeout_manager.start())
        
        logger.info("=" * 60)
        logger.info("✓ Worker started successfully")
        logger.info("=" * 60)
        
        # Wait for tasks to complete (or until shutdown signal)
        await asyncio.gather(publisher_task, timeout_task, return_exceptions=True)
        
    except Exception as e:
        logger.error(f"Fatal error in worker: {e}", exc_info=True)
        raise
    
    finally:
        logger.info("=" * 60)
        logger.info("Worker shutdown complete")
        logger.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        exit(1)
