import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class UploadedFile(Base):  # type: ignore
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(
        String(36),
        unique=True,
        index=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    task_id = Column(Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=True)
    # Index is created by migration 20260410_add_index_on_uploaded_files_filename.py
    # to ensure existing databases have the index for URL deduplication queries.
    filename = Column(String(512), nullable=False)
    storage_path = Column(String(2048), nullable=False, unique=True)
    storage_backend = Column(String(64), nullable=True)
    storage_key = Column(String(2048), nullable=True)
    storage_uri = Column(String(4096), nullable=True)
    checksum = Column(String(128), nullable=True)
    etag = Column(String(255), nullable=True)
    workspace_relative_path = Column(String(2048), nullable=True)
    workspace_category = Column(String(64), nullable=True)
    # Provenance marker for uploads created before any task/owner binding
    # exists (currently the task-less public-share path, #973). NULL for the
    # overwhelming majority of rows (task-bound / logged-in draft uploads);
    # only orphan GC keys off this so a coarse "task_id IS NULL" sweep can't
    # reap a logged-in user's un-sent draft attachments.
    upload_source = Column(String(64), nullable=True)
    storage_status = Column(String(32), nullable=False, default="legacy")
    mime_type = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="uploaded_files")
    task = relationship("Task", back_populates="uploaded_files")

    def __repr__(self) -> str:
        return f"<UploadedFile(file_id={self.file_id}, filename='{self.filename}', user_id={self.user_id})>"
