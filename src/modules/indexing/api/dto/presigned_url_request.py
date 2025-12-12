from typing import List
from pydantic import BaseModel

class PresignedUrlRequest(BaseModel):
    files: List[str] = ["video.mp4", "audio.wav", "metadata.json", "scenes.json"]
    # Mapping filename to content_type. If not provided, defaults will be used or guessed (server side logic)
    # But to keep it simple, we can just accept files list and server knows types.
    # User flow says: "client requests", usually client knows what it wants.
