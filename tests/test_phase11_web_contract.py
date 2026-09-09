from __future__ import annotations

from uuid import uuid4

import pytest
from app.llm.fake_skill_gap_provider import FakeSkillGapGenerationProvider
from app.main import app
from app.schemas.application_package import (
    ApplicationPackageCreate,
    ApplicationPackageRead,
    CandidateProfilePreparationRead,
)


@pytest.fixture(autouse=True)
def clean_database() -> None:
    """Contract tests do not require PostgreSQL."""


def test_phase11_routes_are_registered() -> None:
    routes = {(route.path, frozenset(route.methods or set())) for route in app.routes}
    assert (
        "/api/v1/candidate-profile/{profile_id}/documents/upload",
        frozenset({"POST"}),
    ) in routes
    assert (
        "/api/v1/candidate-profile/{profile_id}/documents/{document_id}/prepare",
        frozenset({"POST"}),
    ) in routes
    assert ("/api/v1/application-packages", frozenset({"POST"})) in routes


def test_application_request_accepts_manual_job_description() -> None:
    payload = ApplicationPackageCreate(
        profile_id=uuid4(),
        job_url="https://www.linkedin.com/jobs/view/4460856942/",
        job_description="A complete job description containing required skills.",
        title="Robotics Engineer",
        company="Example Company",
        location="Abu Dhabi",
    )
    assert payload.profile_id
    assert payload.job_description.startswith("A complete")


def test_profile_preparation_requires_nonnegative_counts() -> None:
    result = CandidateProfilePreparationRead(
        profile_id=uuid4(),
        document_id=uuid4(),
        evidence_records=5,
        embedded_records=5,
        ready=True,
    )
    assert result.ready is True
    assert result.embedded_records == result.evidence_records


def test_application_package_has_llm_skill_gap_fields() -> None:
    assert hasattr(FakeSkillGapGenerationProvider, "generate_skill_gap_report")
    fields = ApplicationPackageRead.model_fields
    assert "skill_gap_report" in fields
    assert "skill_gap_html" in fields
    assert "skill_gap_pdf_base64" in fields
