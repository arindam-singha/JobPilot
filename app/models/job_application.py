from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class JobApplication(Base):
    """Track a candidate's application and its generated artifacts."""

    __tablename__ = "job_applications"

    __table_args__ = (
        CheckConstraint(
            "status IN ("
            "'draft', "
            "'ready_to_apply', "
            "'applied', "
            "'screening', "
            "'interview', "
            "'offer', "
            "'rejected', "
            "'withdrawn'"
            ")",
            name="ck_job_applications_status",
        ),
        UniqueConstraint(
            "job_id",
            "profile_id",
            name="uq_job_applications_job_profile",
        ),
        Index(
            "ix_job_applications_profile_status",
            "profile_id",
            "status",
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
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    profile_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="draft",
        server_default="draft",
    )

    resume_snapshot: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    cover_letter_snapshot: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    application_url: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    applied_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )