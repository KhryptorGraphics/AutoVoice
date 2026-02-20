"""SQLAlchemy models for voice profiles and training samples.

These models define the PostgreSQL schema for persistent storage of
voice profiles and accumulated training data.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class VoiceProfileDB(Base):
    """SQLAlchemy model for voice profiles.

    Stores user voice profiles with metadata for continuous learning.
    """

    __tablename__ = "voice_profiles"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    samples_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    model_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    settings: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Relationships
    samples: Mapped[list["TrainingSampleDB"]] = relationship(
        "TrainingSampleDB",
        back_populates="profile",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    # Indexes
    __table_args__ = (
        Index("ix_voice_profiles_user_created", "user_id", "created"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "created": self.created.isoformat() if self.created else None,
            "updated": self.updated.isoformat() if self.updated else None,
            "samples_count": self.samples_count,
            "model_version": self.model_version,
            "model_path": self.model_path,
            "settings": self.settings,
        }


class TrainingSampleDB(Base):
    """SQLAlchemy model for training samples.

    Stores metadata about audio samples used for voice profile training.
    """

    __tablename__ = "training_samples"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    profile_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("voice_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    audio_path: Mapped[str] = mapped_column(Text, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    sample_rate: Mapped[int] = mapped_column(Integer, nullable=False)
    created: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Processing status
    processed: Mapped[bool] = mapped_column(Integer, nullable=False, default=False)
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    profile: Mapped["VoiceProfileDB"] = relationship(
        "VoiceProfileDB", back_populates="samples"
    )

    # Indexes
    __table_args__ = (
        Index("ix_training_samples_profile_created", "profile_id", "created"),
        Index("ix_training_samples_profile_processed", "profile_id", "processed"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "audio_path": self.audio_path,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "created": self.created.isoformat() if self.created else None,
            "quality_score": self.quality_score,
            "metadata": self.extra_metadata,
            "processed": bool(self.processed),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }
