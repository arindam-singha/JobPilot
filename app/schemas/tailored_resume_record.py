from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.tailored_resume import TailoredResumeDraft


class TailoredResumeRecordRead(BaseModel):
    """Complete persisted tailored-resume representation."""

    id: UUID
    job_id: UUID
    profile_id: UUID
    status: str
    structured_content: TailoredResumeDraft
    generator_provider: str
    generator_model: str
    generation_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
    )


class TailoredResumeRecordSummary(BaseModel):
    """Lightweight tailored-resume list representation."""

    id: UUID
    job_id: UUID
    profile_id: UUID
    status: str
    target_title: str
    generator_provider: str
    generator_model: str
    evidence_count: int = Field(..., ge=0)
    created_at: datetime

    model_config = ConfigDict(extra="forbid")


class TailoredResumeRecordList(BaseModel):
    """Paginated tailored-resume list response."""

    items: list[TailoredResumeRecordSummary] = Field(default_factory=list)
    total: int = Field(..., ge=0)
    limit: int = Field(..., ge=1, le=100)
    offset: int = Field(..., ge=0)

    model_config = ConfigDict(extra="forbid")
