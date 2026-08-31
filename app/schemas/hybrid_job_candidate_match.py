from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.job_candidate_match import ExperienceYearsMatch


class HybridSupportingEvidence(BaseModel):
    evidence_id: UUID
    evidence_type: str
    title: str
    source_type: str

    deterministic_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )
    semantic_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )
    hybrid_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    model_config = ConfigDict(extra="forbid")


class HybridRequirementMatch(BaseModel):
    requirement: str
    matched: bool

    deterministic_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )
    semantic_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )
    hybrid_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    supporting_evidence: list[HybridSupportingEvidence] = Field(
        default_factory=list,
    )

    model_config = ConfigDict(extra="forbid")


class HybridCategoryResult(BaseModel):
    category: str
    score: float = Field(..., ge=0.0, le=100.0)
    total_requirements: int = Field(..., ge=0)
    matched_requirements: int = Field(..., ge=0)
    requirements: list[HybridRequirementMatch] = Field(
        default_factory=list,
    )

    model_config = ConfigDict(extra="forbid")


class HybridJobCandidateMatchRead(BaseModel):
    job_id: UUID
    profile_id: UUID

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
    )

    deterministic_overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
    )

    semantic_overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
    )

    required_skills: HybridCategoryResult
    preferred_skills: HybridCategoryResult
    required_experience: HybridCategoryResult
    education: HybridCategoryResult
    certifications: HybridCategoryResult

    experience_years: ExperienceYearsMatch

    matched_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)

    evidence_considered: int = Field(..., ge=0)

    embedding_provider: str
    embedding_model: str

    deterministic_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
    )
    semantic_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
    )

    scoring_method: str = (
        "hybrid_deterministic_and_semantic_matching"
    )

    model_config = ConfigDict(extra="forbid")