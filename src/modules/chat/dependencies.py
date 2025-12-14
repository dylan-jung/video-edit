import os
from functools import lru_cache
from fastapi import Path, Depends
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from src.modules.chat.application.service import AgentService

@lru_cache
def get_mongo_client() -> MongoClient:
    """
    Create a singleton MongoDB client connection.
    This ensures we use connection pooling effectively.
    """
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    return MongoClient(mongo_uri)

def get_agent_service(
    project_id: str = Path(...),
    mongo_client: MongoClient = Depends(get_mongo_client)
) -> AgentService:
    """
    Dependency provider for AgentService.
    Injects project_id and a configured MongoDB checkpointer.
    """
    # Create a checkpointer using the shared client
    checkpointer = MongoDBSaver(mongo_client)
    
    return AgentService(project_id=project_id, checkpointer=checkpointer)
