from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any
import uuid


class EventType(str, Enum):
    """도메인 이벤트 타입"""
    VIDEO_UPLOADED = "VIDEO_UPLOADED"
    VIDEO_INDEXING_STARTED = "VIDEO_INDEXING_STARTED"
    VIDEO_INDEXED = "VIDEO_INDEXED"
    VIDEO_INDEXING_FAILED = "VIDEO_INDEXING_FAILED"


@dataclass
class DomainEvent:
    """도메인 이벤트 베이스 클래스"""
    event_id: str
    event_type: EventType
    aggregate_id: str  # Video ID
    payload: Dict[str, Any]
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "aggregate_id": self.aggregate_id,
            "payload": self.payload,
            "occurred_at": self.occurred_at
        }


@dataclass
class VideoUploadedEvent(DomainEvent):
    """비디오 업로드 이벤트"""
    def __init__(self, video_id: str, project_id: str, file_path: str):
        super().__init__(
            event_id=str(uuid.uuid4()),
            event_type=EventType.VIDEO_UPLOADED,
            aggregate_id=video_id,
            payload={
                "video_id": video_id,
                "project_id": project_id,
                "file_path": file_path
            }
        )


@dataclass
class VideoIndexingStartedEvent(DomainEvent):
    """비디오 인덱싱 시작 이벤트"""
    def __init__(self, video_id: str, project_id: str):
        super().__init__(
            event_id=str(uuid.uuid4()),
            event_type=EventType.VIDEO_INDEXING_STARTED,
            aggregate_id=video_id,
            payload={
                "video_id": video_id,
                "project_id": project_id
            }
        )


@dataclass
class VideoIndexedEvent(DomainEvent):
    """비디오 인덱싱 완료 이벤트"""
    def __init__(self, video_id: str, project_id: str):
        super().__init__(
            event_id=str(uuid.uuid4()),
            event_type=EventType.VIDEO_INDEXED,
            aggregate_id=video_id,
            payload={
                "video_id": video_id,
                "project_id": project_id
            }
        )


@dataclass
class VideoIndexingFailedEvent(DomainEvent):
    """비디오 인덱싱 실패 이벤트"""
    def __init__(self, video_id: str, project_id: str, error_message: str):
        super().__init__(
            event_id=str(uuid.uuid4()),
            event_type=EventType.VIDEO_INDEXING_FAILED,
            aggregate_id=video_id,
            payload={
                "video_id": video_id,
                "project_id": project_id,
                "error_message": error_message
            }
        )
