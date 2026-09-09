from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class JobApplicationStatus(str, Enum):
    DRAFT = "draft"
    READY_TO_APPLY = "ready_to_apply"
    APPLIED = "applied"
    SCREENING = "screening"
    INTERVIEW = "interview"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class JobApplicationCreate(BaseModel):
    job_id: UUID
    profile_id: UUID
    status: JobApplicationStatus = JobApplicationStatus.DRAFT
    resume_snapshot: dict[str, Any] | None = None
    cover_letter_snapshot: dict[str, Any] | None = None
    application_url: str | None = None
    notes: str | None = None
    applied_at: datetime | None = None

    model_config = ConfigDict(extra="forbid")


class JobApplicationUpdate(BaseModel):
    status: JobApplicationStatus | None = None
    resume_snapshot: dict[str, Any] | None = None
    cover_letter_snapshot: dict[str, Any] | None = None
    application_url: str | None = None
    notes: str | None = None
    applied_at: datetime | None = None

    model_config = ConfigDict(extra="forbid")


class JobApplicationRead(BaseModel):
    id: UUID
    job_id: UUID
    profile_id: UUID
    status: JobApplicationStatus
    resume_snapshot: dict[str, Any] | None
    cover_letter_snapshot: dict[str, Any] | None
    application_url: str | None
    notes: str | None
    applied_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
    )