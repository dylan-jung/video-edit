import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.modules.chat.api.router import router as chat_router
from src.modules.indexing.api.router import router as indexing_router

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Attenz API", version="1.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include Routers
# Prefix can be adjusted as needed, keeping /api/v1 for compatibility
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(indexing_router, prefix="/api/v1", tags=["indexing"])

@app.get("/health")
async def health_check():
    return {"status": "ok"}
