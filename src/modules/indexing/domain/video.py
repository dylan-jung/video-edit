from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class VideoStatus(str, Enum):
    """비디오 상태"""
    UPLOADED = "UPLOADED"
    INDEXING = "INDEXING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


@dataclass
class Video:
    """
    비디오 도메인 엔티티
    
    비즈니스 로직과 상태 관리를 담당하는 핵심 도메인 모델
    """
    id: str
    project_id: str
    file_path: str
    content_type: str
    status: VideoStatus = VideoStatus.UPLOADED
    uploaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    indexed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def mark_as_indexing(self) -> None:
        """인덱싱 시작 상태로 변경"""
        if self.status != VideoStatus.UPLOADED:
            raise ValueError(f"Cannot start indexing from status: {self.status}")
        self.status = VideoStatus.INDEXING
    
    def mark_as_indexed(self) -> None:
        """인덱싱 완료 상태로 변경"""
        if self.status != VideoStatus.INDEXING:
            raise ValueError(f"Cannot mark as indexed from status: {self.status}")
        self.status = VideoStatus.INDEXED
        self.indexed_at = datetime.now(timezone.utc)
    
    def mark_as_failed(self, error_message: str) -> None:
        """인덱싱 실패 상태로 변경"""
        self.status = VideoStatus.FAILED
        self.error_message = error_message
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환 (MongoDB 저장용)"""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "file_path": self.file_path,
            "content_type": self.content_type,
            "status": self.status.value,
            "uploaded_at": self.uploaded_at,
            "indexed_at": self.indexed_at,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Video":
        """딕셔너리에서 생성 (MongoDB 조회용)"""
        return cls(
            id=data["id"],
            project_id=data["project_id"],
            file_path=data["file_path"],
            content_type=data["content_type"],
            status=VideoStatus(data["status"]),
            uploaded_at=data["uploaded_at"],
            indexed_at=data.get("indexed_at"),
            error_message=data.get("error_message")
        )
