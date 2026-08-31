from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class TailoredResume(Base):
    """Persisted structured tailored-resume generation."""

    __tablename__ = "tailored_resumes"

    __table_args__ = (
        CheckConstraint(
            "status IN ('generated')",
            name="ck_tailored_resumes_status",
        ),
        Index(
            "ix_tailored_resumes_job_profile_created",
            "job_id",
            "profile_id",
            "created_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )

    job_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "jobs.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    profile_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "candidate_profiles.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="generated",
        server_default="generated",
    )

    structured_content: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
    )

    generator_provider: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    generator_model: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    generation_metadata: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
