from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.job import Job


class JobRequirements(Base):
    """Structured requirements extracted from a job description."""

    __tablename__ = "job_requirements"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )

    job_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    required_skills: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    preferred_skills: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    required_experience: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    responsibilities: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    education_requirements: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    certifications: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    domain_keywords: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    minimum_experience_years: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    provider: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    model_name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    extraction_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
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

    job: Mapped["Job"] = relationship(
        back_populates="requirements",
    )