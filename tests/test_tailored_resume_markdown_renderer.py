from __future__ import annotations

from uuid import uuid4

import pytest
from app.schemas.tailored_resume import (
    GroundedResumeStatement,
    ResumeHeader,
    ResumeRequirementTrace,
    SelectedResumeEvidence,
    TailoredResumeDraft,
    TailoredResumeSection,
    ResumeExperienceEntry,
    ResumeSkillGroup,
)
from app.services.tailored_resume_markdown_renderer import (
    TailoredResumeMarkdownRenderer,
)


@pytest.fixture(autouse=True)
def clean_database() -> None:
    """Override database fixture for renderer tests."""


def _draft() -> TailoredResumeDraft:
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
        verified_summary=(
            "Computer vision engineer with "
            "manufacturing inspection experience."
        ),
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
                statements=[
                    GroundedResumeStatement(
                        text=("Reduced inspection time " "from 10 minutes to 5 seconds."),
                        evidence_ids=[evidence_id],
                    )
                ],
            )
        ],
        skill_groups=[
            ResumeSkillGroup(
                category="Technical Skills",
                skills=["Python", "PyTorch", "FastAPI"],
            )
        ],
        experiences=[
            ResumeExperienceEntry(
                role="Computer Vision Engineer",
                company="Example Company",
                location="Abu Dhabi",
                start_date="2022-01-01",
                end_date=None,
                is_current=True,
                bullets=[
                    "Reduced inspection time from "
                    "10 minutes to 5 seconds."
                ],
            )
        ],
        evidence_catalog=[evidence],
        generator_provider="ollama",
        generator_model="qwen2.5:7b",
    )


def test_renderer_produces_expected_sections() -> None:
    markdown = TailoredResumeMarkdownRenderer().render(_draft())

    assert "# Test Candidate" in markdown
    assert "Senior Computer Vision Engineer" in markdown
    assert "## Professional Summary" in markdown
    assert "## Core Skills" in markdown
    assert "## Professional Experience" in markdown


def test_renderer_includes_contact_information() -> None:
    markdown = TailoredResumeMarkdownRenderer().render(_draft())

    assert "candidate@example.com" in markdown
    assert "+971500000000" in markdown
    assert "Abu Dhabi" in markdown
    assert "LinkedIn:" in markdown
    assert "GitHub:" in markdown


def test_renderer_uses_simple_bullets() -> None:
    markdown = TailoredResumeMarkdownRenderer().render(_draft())

    # assert "- Computer vision engineer with " "manufacturing inspection experience." in markdown
    assert (
        "Computer vision engineer with "
        "manufacturing inspection experience."
        in markdown
    )
    assert "- Reduced inspection time from " "10 minutes to 5 seconds." in markdown


def test_renderer_excludes_internal_provenance() -> None:
    draft = _draft()

    markdown = TailoredResumeMarkdownRenderer().render(draft)

    evidence = draft.evidence_catalog[0]

    assert str(evidence.evidence_id) not in markdown
    assert str(evidence.source_id) not in markdown
    assert "selection_score" not in markdown
    assert "requirement_traces" not in markdown
    assert "generator_provider" not in markdown


def test_renderer_is_deterministic() -> None:
    draft = _draft()
    renderer = TailoredResumeMarkdownRenderer()

    first = renderer.render(draft)
    second = renderer.render(draft)

    assert first == second


def test_renderer_ends_with_single_newline() -> None:
    markdown = TailoredResumeMarkdownRenderer().render(_draft())

    assert markdown.endswith("\n")
    assert not markdown.endswith("\n\n")


def test_renderer_escapes_markdown_characters() -> None:
    draft = _draft()

    # draft.professional_summary[0].text = "Built *production* [AI] systems."
    draft.verified_summary = "Built *production* [AI] systems."

    markdown = TailoredResumeMarkdownRenderer().render(draft)

    assert r"Built \*production\* \[AI\] systems." in markdown


def test_renderer_normalizes_whitespace() -> None:
    draft = _draft()

    # draft.professional_summary[0].text = "Built   production\ninspection\t systems."
    draft.verified_summary = (
    "Built   production\ninspection\t systems."
    )

    markdown = TailoredResumeMarkdownRenderer().render(draft)

    assert "Built production inspection systems." in markdown


def test_renderer_uses_single_column_format() -> None:
    markdown = TailoredResumeMarkdownRenderer().render(_draft())

    assert "<table" not in markdown
    assert "<div" not in markdown
    assert "![" not in markdown
    assert "```" not in markdown
