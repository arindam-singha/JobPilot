from __future__ import annotations

from io import BytesIO
from uuid import uuid4

import pytest
from app.schemas.tailored_resume import (
    GroundedResumeStatement,
    ResumeHeader,
    ResumeRequirementTrace,
    SelectedResumeEvidence,
    TailoredResumeDraft,
    TailoredResumeSection,
)
from app.services.tailored_resume_pdf_renderer import (
    TailoredResumePdfRenderer,
)
from pypdf import PdfReader


@pytest.fixture(autouse=True)
def clean_database() -> None:
    """Override database fixture for PDF unit tests."""


def _draft(
    *,
    statement_count: int = 1,
) -> TailoredResumeDraft:
    profile_id = uuid4()
    evidence_id = uuid4()

    evidence = SelectedResumeEvidence(
        evidence_id=evidence_id,
        profile_id=profile_id,
        evidence_type="project",
        title="Defect detection",
        content=("Reduced inspection time from " "10 minutes to 5 seconds."),
        source_type="candidate_project",
        source_id=uuid4(),
        selection_score=0.94,
        requirement_traces=[
            ResumeRequirementTrace(
                category="required_experience",
                requirement=("Visual inspection automation"),
                deterministic_score=1.0,
                semantic_score=0.85,
                hybrid_score=0.94,
                rank=1,
            )
        ],
    )

    statements = [
        GroundedResumeStatement(
            text=(f"Delivered manufacturing inspection " f"automation project {index + 1}."),
            evidence_ids=[evidence_id],
        )
        for index in range(statement_count)
    ]

    return TailoredResumeDraft(
        job_id=uuid4(),
        profile_id=profile_id,
        header=ResumeHeader(
            full_name="Test Candidate",
            email="candidate@example.com",
            phone="+971500000000",
            location="Abu Dhabi",
            linkedin_url=("https://linkedin.com/in/candidate"),
            github_url=("https://github.com/candidate"),
        ),
        target_title=("Senior Computer Vision Engineer"),
        professional_summary=[
            GroundedResumeStatement(
                text=("Computer vision engineer with " "manufacturing inspection experience."),
                evidence_ids=[evidence_id],
            )
        ],
        skills=[
            GroundedResumeStatement(
                text="Python, PyTorch, FastAPI",
                evidence_ids=[evidence_id],
            )
        ],
        sections=[
            TailoredResumeSection(
                heading="Professional Experience",
                statements=statements,
            )
        ],
        evidence_catalog=[evidence],
        generator_provider="ollama",
        generator_model="qwen2.5:7b",
    )


def _extract_text(
    pdf_bytes: bytes,
) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))

    return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_renderer_produces_valid_pdf() -> None:
    pdf_bytes = TailoredResumePdfRenderer().render(_draft())

    assert pdf_bytes.startswith(b"%PDF-")


def test_pdf_contains_expected_candidate_text() -> None:
    pdf_bytes = TailoredResumePdfRenderer().render(_draft())

    text = _extract_text(pdf_bytes)

    assert "Test Candidate" in text
    assert "Senior Computer Vision Engineer" in text
    assert "PROFESSIONAL SUMMARY" in text
    assert "SKILLS" in text
    assert "PROFESSIONAL EXPERIENCE" in text


def test_pdf_contains_contact_information() -> None:
    text = _extract_text(TailoredResumePdfRenderer().render(_draft()))

    assert "candidate@example.com" in text
    assert "+971500000000" in text
    assert "Abu Dhabi" in text
    assert "LinkedIn:" in text
    assert "GitHub:" in text


def test_pdf_contains_resume_statements() -> None:
    text = _extract_text(TailoredResumePdfRenderer().render(_draft()))

    assert "Computer vision engineer with " "manufacturing inspection experience." in text

    assert "Delivered manufacturing inspection " "automation project 1." in text


def test_pdf_excludes_provenance() -> None:
    draft = _draft()
    evidence = draft.evidence_catalog[0]

    text = _extract_text(TailoredResumePdfRenderer().render(draft))

    assert str(evidence.evidence_id) not in text
    assert str(evidence.source_id) not in text
    assert "requirement_traces" not in text
    assert "selection_score" not in text


def test_pdf_supports_multiple_pages() -> None:
    pdf_bytes = TailoredResumePdfRenderer().render(_draft(statement_count=100))

    reader = PdfReader(BytesIO(pdf_bytes))

    assert len(reader.pages) > 1

    text = _extract_text(pdf_bytes)

    assert "Delivered manufacturing inspection " "automation project 100." in text


def test_pdf_uses_extractable_text() -> None:
    pdf_bytes = TailoredResumePdfRenderer().render(_draft())

    text = _extract_text(pdf_bytes)

    assert len(text.strip()) > 100
