"""
SQLAlchemy models for Feedback.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class Feedback(Base):
    """
    Feedback table for capturing user interaction.
    """

    __tablename__ = "users_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String, index=True, nullable=False)
    message_id = Column(String, index=True, nullable=False)
    thumbs_up = Column(Boolean, nullable=False)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        status = "Up" if self.thumbs_up else "Down"
        return f"<Feedback(id={self.id}, status={status})>"
