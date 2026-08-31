from __future__ import annotations

from uuid import uuid4

import pytest
from app.llm.fake_cover_letter_provider import FakeCoverLetterGenerationProvider
from app.schemas.cover_letter import (
    CoverLetterGenerationContext,
    GroundedCoverLetterParagraph,
    TailoredCoverLetterDraft,
)
from app.schemas.tailored_resume import (
    ResumeGroundingBundle,
    ResumeHeader,
    ResumeRequirementTrace,
    SelectedResumeEvidence,
)
from app.services.cover_letter_grounding_validator import (
    CoverLetterGroundingValidationError,
    CoverLetterGroundingValidator,
)
from app.services.cover_letter_markdown_renderer import CoverLetterMarkdownRenderer
from app.services.cover_letter_pdf_renderer import CoverLetterPdfRenderer
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def clean_database() -> None:
    """These unit tests do not require the PostgreSQL integration fixture."""


def _objects() -> tuple[CoverLetterGenerationContext, SelectedResumeEvidence]:
    profile_id = uuid4()
    evidence = SelectedResumeEvidence(
        evidence_id=uuid4(),
        profile_id=profile_id,
        evidence_type="project",
        title="Inspection platform",
        content="Reduced inspection time from 10 minutes to 5 seconds using computer vision.",
        source_type="candidate_document",
        source_id=uuid4(),
        selection_score=0.9,
        requirement_traces=[
            ResumeRequirementTrace(
                category="required_skills",
                requirement="Computer vision",
                deterministic_score=1.0,
                semantic_score=0.75,
                hybrid_score=0.9,
                rank=1,
            )
        ],
    )
    grounding = ResumeGroundingBundle(
        job_id=uuid4(),
        profile_id=profile_id,
        hybrid_overall_score=90,
        matched_requirements=["Computer vision"],
        selected_evidence=[evidence],
        embedding_provider="fake",
        embedding_model="fake",
        minimum_evidence_score=0.6,
        max_evidence_per_requirement=3,
        max_total_evidence=12,
    )
    context = CoverLetterGenerationContext(
        job_title="Computer Vision Engineer",
        company="Example",
        job_description="Build computer-vision inspection systems.",
        header=ResumeHeader(full_name="Candidate Name", email="candidate@example.com"),
        grounding=grounding,
    )
    return context, evidence


async def _draft() -> TailoredCoverLetterDraft:
    context, evidence = _objects()
    content = await FakeCoverLetterGenerationProvider().generate_cover_letter(context)
    return TailoredCoverLetterDraft(
        job_id=context.grounding.job_id,
        profile_id=context.grounding.profile_id,
        header=context.header,
        job_title=context.job_title,
        company=context.company,
        evidence_catalog=[evidence],
        generator_provider="fake",
        generator_model="fake-cover-letter-v1",
        **content.model_dump(),
    )


@pytest.mark.asyncio
async def test_fake_provider_returns_grounded_content() -> None:
    context, evidence = _objects()
    content = await FakeCoverLetterGenerationProvider().generate_cover_letter(context)
    assert content.opening.evidence_ids == [evidence.evidence_id]
    assert content.subject == "Application for Computer Vision Engineer"


@pytest.mark.asyncio
async def test_draft_rejects_unknown_provenance() -> None:
    draft = await _draft()
    payload = draft.model_dump()
    payload["opening"]["evidence_ids"] = [uuid4()]
    with pytest.raises(ValidationError, match="unknown evidence"):
        TailoredCoverLetterDraft.model_validate(payload)


@pytest.mark.asyncio
async def test_numeric_validator_accepts_supported_values() -> None:
    CoverLetterGroundingValidator().validate(await _draft())


@pytest.mark.asyncio
async def test_numeric_validator_rejects_invented_value() -> None:
    draft = await _draft()
    draft.body_paragraphs[0] = GroundedCoverLetterParagraph(
        text="Improved accuracy by 99%.",
        evidence_ids=draft.body_paragraphs[0].evidence_ids,
    )
    with pytest.raises(CoverLetterGroundingValidationError, match="99%"):
        CoverLetterGroundingValidator().validate(draft)


@pytest.mark.asyncio
async def test_markdown_is_readable_and_excludes_provenance() -> None:
    draft = await _draft()
    markdown = CoverLetterMarkdownRenderer().render(draft)
    assert "Dear Hiring Manager" in markdown
    assert "Application for Computer Vision Engineer" in markdown
    assert str(draft.evidence_catalog[0].evidence_id) not in markdown


@pytest.mark.asyncio
async def test_pdf_is_valid_and_excludes_provenance() -> None:
    draft = await _draft()
    pdf = CoverLetterPdfRenderer().render(draft)
    assert pdf.startswith(b"%PDF-")
    assert str(draft.evidence_catalog[0].evidence_id).encode() not in pdf
