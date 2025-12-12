from pydantic import BaseModel, Field

class Track(BaseModel):
    src: str = Field(..., description="Source video file identifier or path")
    start: str = Field(..., description="Start time in the timeline (HH:MM:SS)")
    end: str = Field(..., description="End time in the timeline (HH:MM:SS)")
    duration: str = Field(..., description="Duration of the segment (HH:MM:SS)")
    trim_in: str = Field(..., alias="trimIn", description="Start time in the source video (HH:MM:SS)")
    trim_out: str = Field(..., alias="trimOut", description="End time in the source video (HH:MM:SS)")
