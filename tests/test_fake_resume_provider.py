from __future__ import annotations

from uuid import uuid4

import pytest
from app.llm.fake_resume_provider import FakeResumeGenerationProvider
from app.schemas.tailored_resume import (
    GroundedResumeStatement,
    ResumeGenerationContext,
    ResumeGroundingBundle,
    ResumeHeader,
    ResumeRequirementTrace,
    SelectedResumeEvidence,
    TailoredResumeContent,
)


@pytest.fixture(autouse=True)
def clean_database() -> None:
    """Override the suite-wide database fixture for pure provider tests."""


def _context(*, evidence_type: str = "project") -> ResumeGenerationContext:
    profile_id = uuid4()
    evidence = SelectedResumeEvidence(
        evidence_id=uuid4(),
        profile_id=profile_id,
        evidence_type=evidence_type,
        title="Defect detection",
        content="Built a manufacturing defect-detection platform.",
        source_type="candidate_project",
        source_id=uuid4(),
        selection_score=0.9,
        requirement_traces=[
            ResumeRequirementTrace(
                category="required_experience",
                requirement="Computer vision",
                deterministic_score=1.0,
                semantic_score=0.75,
                hybrid_score=0.9,
                rank=1,
            )
        ],
    )
    return ResumeGenerationContext(
        job_title="Senior Computer Vision Engineer",
        company="Example Manufacturing",
        location="Abu Dhabi",
        job_description="Build production visual-inspection systems.",
        header=ResumeHeader(full_name="Candidate"),
        grounding=ResumeGroundingBundle(
            job_id=uuid4(),
            profile_id=profile_id,
            hybrid_overall_score=90,
            matched_requirements=["Computer vision"],
            selected_evidence=[evidence],
            embedding_provider="fake",
            embedding_model="fake-model",
            minimum_evidence_score=0.6,
            max_evidence_per_requirement=3,
            max_total_evidence=12,
        ),
    )


@pytest.mark.asyncio
async def test_fake_provider_generates_grounded_content() -> None:
    context = _context()
    provider = FakeResumeGenerationProvider()

    result = await provider.generate_resume(context)

    evidence = context.grounding.selected_evidence[0]
    assert result.target_title == context.job_title
    assert result.professional_summary[0].text == evidence.content
    assert result.professional_summary[0].evidence_ids == [evidence.evidence_id]
    assert result.sections[0].heading == "Project"
    assert provider.provider_name == "fake"
    assert provider.model_name == "fake-resume-model"


@pytest.mark.asyncio
async def test_fake_provider_places_skills_in_skills_collection() -> None:
    result = await FakeResumeGenerationProvider().generate_resume(_context(evidence_type="skill"))

    assert len(result.skills) == 1
    assert result.sections == []


@pytest.mark.asyncio
async def test_fake_provider_uses_custom_responder() -> None:
    context = _context()
    evidence_id = context.grounding.selected_evidence[0].evidence_id
    expected = TailoredResumeContent(
        target_title="Custom title",
        professional_summary=[
            GroundedResumeStatement(
                text="Custom grounded statement",
                evidence_ids=[evidence_id],
            )
        ],
    )
    provider = FakeResumeGenerationProvider(lambda received: expected)

    assert await provider.generate_resume(context) == expected
