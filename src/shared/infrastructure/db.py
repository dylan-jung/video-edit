from motor.motor_asyncio import AsyncIOMotorClient
from src.config.settings import get_settings

_client = None

def get_mongo_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncIOMotorClient(settings.MONGO_URI)
    return _client

def get_db():
    client = get_mongo_client()
    settings = get_settings()
    return client[settings.DB_NAME]
