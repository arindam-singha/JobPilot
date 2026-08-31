from __future__ import annotations

from uuid import uuid4

import pytest
from app.schemas.tailored_resume import (
    GroundedResumeStatement,
    ResumeGroundingBundle,
    ResumeHeader,
    ResumeRequirementTrace,
    SelectedResumeEvidence,
    TailoredResumeDraft,
)
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def clean_database() -> None:
    """Override the suite-wide database fixture for pure schema tests."""


def _trace() -> ResumeRequirementTrace:
    return ResumeRequirementTrace(
        category="required_skills",
        requirement="Python",
        deterministic_score=1.0,
        semantic_score=0.8,
        hybrid_score=0.92,
        rank=1,
    )


def _evidence(*, profile_id=None, evidence_id=None) -> SelectedResumeEvidence:
    return SelectedResumeEvidence(
        evidence_id=evidence_id or uuid4(),
        profile_id=profile_id or uuid4(),
        evidence_type="skill",
        title="Python",
        content="Built production Python services.",
        source_type="candidate_skill",
        source_id=uuid4(),
        selection_score=0.92,
        requirement_traces=[_trace()],
    )


def test_grounded_statement_requires_provenance() -> None:
    with pytest.raises(ValidationError):
        GroundedResumeStatement(text="Built APIs", evidence_ids=[])


def test_grounded_statement_normalizes_whitespace() -> None:
    statement = GroundedResumeStatement(
        text="  Built   production\nAPIs. ",
        evidence_ids=[uuid4()],
    )

    assert statement.text == "Built production APIs."


def test_grounding_bundle_rejects_duplicate_evidence() -> None:
    profile_id = uuid4()
    evidence_id = uuid4()
    evidence = _evidence(profile_id=profile_id, evidence_id=evidence_id)

    with pytest.raises(ValidationError, match="duplicate evidence IDs"):
        ResumeGroundingBundle(
            job_id=uuid4(),
            profile_id=profile_id,
            hybrid_overall_score=90,
            selected_evidence=[evidence, evidence],
            embedding_provider="fake",
            embedding_model="fake-model",
            minimum_evidence_score=0.6,
            max_evidence_per_requirement=3,
            max_total_evidence=12,
        )


def test_grounding_bundle_rejects_cross_profile_evidence() -> None:
    with pytest.raises(ValidationError, match="requested profile"):
        ResumeGroundingBundle(
            job_id=uuid4(),
            profile_id=uuid4(),
            hybrid_overall_score=90,
            selected_evidence=[_evidence()],
            embedding_provider="fake",
            embedding_model="fake-model",
            minimum_evidence_score=0.6,
            max_evidence_per_requirement=3,
            max_total_evidence=12,
        )


def test_resume_draft_rejects_unknown_statement_evidence() -> None:
    profile_id = uuid4()
    evidence = _evidence(profile_id=profile_id)

    with pytest.raises(ValidationError, match="absent from evidence_catalog"):
        TailoredResumeDraft(
            job_id=uuid4(),
            profile_id=profile_id,
            header=ResumeHeader(full_name="Candidate"),
            target_title="Senior Engineer",
            professional_summary=[
                GroundedResumeStatement(
                    text="Built production systems.",
                    evidence_ids=[uuid4()],
                )
            ],
            evidence_catalog=[evidence],
            generator_provider="fake",
            generator_model="fake-model",
        )


def test_resume_draft_accepts_catalogued_provenance() -> None:
    profile_id = uuid4()
    evidence = _evidence(profile_id=profile_id)
    statement = GroundedResumeStatement(
        text="Built production Python services.",
        evidence_ids=[evidence.evidence_id],
    )

    draft = TailoredResumeDraft(
        job_id=uuid4(),
        profile_id=profile_id,
        header=ResumeHeader(full_name="Candidate"),
        target_title="Senior Engineer",
        professional_summary=[statement],
        evidence_catalog=[evidence],
        generator_provider="ollama",
        generator_model="qwen2.5:7b",
    )

    assert draft.professional_summary == [statement]
