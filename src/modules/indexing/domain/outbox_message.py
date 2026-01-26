from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional


class OutboxStatus(str, Enum):
    """Outbox 메시지 상태"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


@dataclass
class OutboxMessage:
    """
    Outbox 테이블의 메시지
    
    도메인 이벤트를 영구 저장하여 메시지 유실을 방지
    """
    id: str
    event_type: str
    aggregate_id: str  # Video ID
    payload: Dict[str, Any]
    status: OutboxStatus = OutboxStatus.PENDING
    retry_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환 (MongoDB 저장용)"""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "payload": self.payload,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "processed_at": self.processed_at,
            "next_retry_at": self.next_retry_at,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "OutboxMessage":
        """딕셔너리에서 생성 (MongoDB 조회용)"""
        return cls(
            id=data["id"],
            event_type=data["event_type"],
            aggregate_id=data["aggregate_id"],
            payload=data["payload"],
            status=OutboxStatus(data["status"]),
            retry_count=data.get("retry_count", 0),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            processed_at=data.get("processed_at"),
            next_retry_at=data.get("next_retry_at"),
            error_message=data.get("error_message")
        )
