"""
MongoDB 인덱스 생성 스크립트

Transactional Outbox Pattern을 위한 최적화된 인덱스 생성
"""
import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from src.config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_indexes():
    """MongoDB 인덱스 생성"""
    settings = get_settings()
    client = AsyncIOMotorClient(settings.MONGO_URI)
    db = client[settings.DB_NAME]
    
    logger.info("Creating MongoDB indexes...")
    
    # 1. Videos 컬렉션 인덱스
    logger.info("Creating indexes for 'videos' collection...")
    
    # Primary key (unique)
    await db.videos.create_index([("id", 1)], unique=True, name="idx_videos_id")
    
    # Project + Status 조회용
    await db.videos.create_index(
        [("project_id", 1), ("status", 1)],
        name="idx_videos_project_status"
    )
    
    # Status별 조회용
    await db.videos.create_index([("status", 1)], name="idx_videos_status")
    
    logger.info("✓ Videos indexes created")
    
    # 2. Outbox Messages 컬렉션 인덱스
    logger.info("Creating indexes for 'outbox_messages' collection...")
    
    # Primary key (unique)
    await db.outbox_messages.create_index(
        [("id", 1)],
        unique=True,
        name="idx_outbox_id"
    )
    
    # Polling 쿼리 최적화 (가장 중요!)
    # acquire_next_pending_message에서 사용
    await db.outbox_messages.create_index(
        [("status", 1), ("created_at", 1)],
        name="idx_outbox_status_created"
    )
    
    # 재시도 스케줄링 쿼리 최적화
    await db.outbox_messages.create_index(
        [("status", 1), ("next_retry_at", 1)],
        name="idx_outbox_status_retry"
    )
    
    # Visibility Timeout 쿼리 최적화
    # reset_stale_messages에서 사용
    await db.outbox_messages.create_index(
        [("status", 1), ("updated_at", 1)],
        name="idx_outbox_status_updated"
    )
    
    # Aggregate (Video) ID로 조회
    await db.outbox_messages.create_index(
        [("aggregate_id", 1), ("event_type", 1)],
        name="idx_outbox_aggregate_event"
    )
    
    logger.info("✓ Outbox messages indexes created")
    
    # 3. 기존 index_jobs 컬렉션 인덱스 (하위 호환성)
    logger.info("Creating indexes for 'index_jobs' collection (legacy)...")
    
    await db.index_jobs.create_index(
        [("status", 1), ("created_at", 1)],
        name="idx_jobs_status_created"
    )
    
    await db.index_jobs.create_index(
        [("video_id", 1), ("status", 1)],
        name="idx_jobs_video_status"
    )
    
    logger.info("✓ Index jobs indexes created")
    
    # 인덱스 목록 출력
    logger.info("\n=== Created Indexes ===")
    for collection_name in ["videos", "outbox_messages", "index_jobs"]:
        indexes = await db[collection_name].list_indexes().to_list(None)
        logger.info(f"\n{collection_name}:")
        for idx in indexes:
            logger.info(f"  - {idx['name']}: {idx.get('key', {})}")
    
    client.close()
    logger.info("\n✓ All indexes created successfully!")


if __name__ == "__main__":
    asyncio.run(create_indexes())
