from __future__ import annotations

from uuid import uuid4

import pytest
from app.llm.fake_skill_gap_provider import FakeSkillGapGenerationProvider
from app.schemas.skill_gap_report import SkillGapGenerationContext, SkillGapReport
from app.services.skill_gap_html_renderer import SkillGapHtmlRenderer
from app.services.skill_gap_markdown_renderer import SkillGapMarkdownRenderer
from app.services.skill_gap_pdf_renderer import SkillGapPdfRenderer


@pytest.fixture(autouse=True)
def clean_database() -> None:
    """These unit tests do not require PostgreSQL."""


def _context() -> SkillGapGenerationContext:
    return SkillGapGenerationContext(
        job_id=uuid4(),
        profile_id=uuid4(),
        job_title="Robotics Engineer",
        company="Example Robotics",
        job_description="Experience with TensorRT and MuJoCo.",
        overall_match_score=72.5,
        matched_skills=["Python", "Computer Vision"],
        preliminary_missing_skills=["TensorRT", "MuJoCo"],
        candidate_evidence=[],
    )


@pytest.mark.asyncio
async def test_fake_provider_creates_preparation_for_each_gap() -> None:
    context = _context()
    content = await FakeSkillGapGenerationProvider().generate_skill_gap_report(context)

    assert [item.skill for item in content.missing_skills] == ["TensorRT", "MuJoCo"]
    assert all(item.preparation_topics for item in content.missing_skills)
    assert all(item.interview_questions for item in content.missing_skills)


@pytest.mark.asyncio
async def test_report_renders_markdown_html_and_pdf() -> None:
    context = _context()
    content = await FakeSkillGapGenerationProvider().generate_skill_gap_report(context)
    report = SkillGapReport(
        job_id=context.job_id,
        profile_id=context.profile_id,
        job_title=context.job_title,
        company=context.company,
        overall_match_score=context.overall_match_score,
        matched_skills=context.matched_skills,
        executive_summary=content.executive_summary,
        resolved_equivalences=content.resolved_equivalences,
        missing_skills=content.missing_skills,
        preparation_strategy=content.preparation_strategy,
        generator_provider="fake",
        generator_model="fake-skill-gap-v1",
    )

    markdown = SkillGapMarkdownRenderer().render(report)
    rendered_html = SkillGapHtmlRenderer().render(report)
    pdf = SkillGapPdfRenderer().render(report)

    assert "# Skill Gap and Preparation Report" in markdown
    assert "TensorRT" in markdown
    assert "<!doctype html>" in rendered_html
    assert "MuJoCo" in rendered_html
    assert pdf.startswith(b"%PDF-")
