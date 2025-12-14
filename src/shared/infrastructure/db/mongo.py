import logging
from functools import lru_cache
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from src.config.settings import get_settings

_client: AsyncIOMotorClient | None = None

logger: logging.Logger = logging.getLogger(__name__)

def get_mongo_client() -> AsyncIOMotorClient:
    """
    Get or create a MongoDB client.
    Uses generic global for singleton behavior similar to original ad-hoc implementation,
    but centralized here.
    """
    global _client
    if _client is None:
        settings = get_settings()
        logger.info(settings.MONGO_URI)
        _client = AsyncIOMotorClient(settings.MONGO_URI)
    return _client

def get_db() -> AsyncIOMotorDatabase:
    """
    Get the default database instance.
    """
    client = get_mongo_client()
    settings = get_settings()
    return client[settings.DB_NAME]
