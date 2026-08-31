from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SupportingEvidence(BaseModel):
    evidence_id: UUID
    evidence_type: str
    title: str
    source_type: str
    score: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class RequirementMatch(BaseModel):
    requirement: str
    matched: bool
    score: float = Field(..., ge=0.0, le=1.0)
    supporting_evidence: list[SupportingEvidence] = Field(
        default_factory=list
    )

    model_config = ConfigDict(extra="forbid")


class MatchCategoryResult(BaseModel):
    category: str
    score: float = Field(..., ge=0.0, le=100.0)
    total_requirements: int = Field(..., ge=0)
    matched_requirements: int = Field(..., ge=0)
    requirements: list[RequirementMatch] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ExperienceYearsMatch(BaseModel):
    required_years: float | None = Field(default=None, ge=0)
    candidate_years: float | None = Field(default=None, ge=0)
    matched: bool | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class JobCandidateMatchRead(BaseModel):
    job_id: UUID
    profile_id: UUID
    overall_score: float = Field(..., ge=0.0, le=100.0)

    required_skills: MatchCategoryResult
    preferred_skills: MatchCategoryResult
    required_experience: MatchCategoryResult
    education: MatchCategoryResult
    certifications: MatchCategoryResult

    experience_years: ExperienceYearsMatch

    matched_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    evidence_considered: int = Field(..., ge=0)

    scoring_method: str = "deterministic_phrase_and_token_matching"

    model_config = ConfigDict(extra="forbid")