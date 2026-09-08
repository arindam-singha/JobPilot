from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.cover_letter import TailoredCoverLetterDraft
from app.schemas.hybrid_job_candidate_match import HybridJobCandidateMatchRead
from app.schemas.job import JobIngestRequest, JobRead
from app.schemas.job_requirements import JobRequirementsRead
from app.schemas.tailored_resume import TailoredResumeDraft


class CandidateProfilePreparationRead(BaseModel):
    profile_id: UUID
    document_id: UUID
    evidence_records: int = Field(..., ge=0)
    embedded_records: int = Field(..., ge=0)
    ready: bool

    model_config = ConfigDict(extra="forbid")


class ApplicationPackageCreate(JobIngestRequest):
    profile_id: UUID


class ApplicationPackageRead(BaseModel):
    job: JobRead
    requirements: JobRequirementsRead
    match: HybridJobCandidateMatchRead
    resume: TailoredResumeDraft
    cover_letter: TailoredCoverLetterDraft
    resume_html: str
    resume_markdown: str
    cover_letter_markdown: str
    resume_pdf_base64: str
    cover_letter_pdf_base64: str

    model_config = ConfigDict(extra="forbid")
