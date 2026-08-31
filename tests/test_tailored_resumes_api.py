from __future__ import annotations

from io import BytesIO
from uuid import uuid4

import pytest
from pypdf import PdfReader
from app.embeddings.embedding_provider_factory import (
    EmbeddingProviderConfigurationError,
)
from app.models.candidate_profile import CandidateProfile
from app.models.job import Job
from app.schemas.tailored_resume import (
    GroundedResumeStatement,
    ResumeHeader,
    ResumeRequirementTrace,
    SelectedResumeEvidence,
    TailoredResumeDraft,
)
from app.services.hybrid_job_candidate_matching_service import (
    HybridEmbeddedEvidenceNotFoundError,
)
from app.services.job_candidate_matching_service import (
    MatchingCandidateEvidenceNotFoundError,
    MatchingJobNotFoundError,
    MatchingRequirementsNotFoundError,
)
from app.services.semantic_evidence_retrieval_service import (
    SemanticEvidenceProviderFailureError,
)
from app.services.tailored_resume_generation_service import (
    TailoredResumeProfileNotFoundError,
    TailoredResumeProviderFailureError,
    TailoredResumeValidationError,
)
from app.services.tailored_resume_grounding_service import (
    TailoredResumeEvidenceNotFoundError,
)


class StubGenerationService:
    def __init__(
        self,
        *,
        draft: TailoredResumeDraft | None = None,
        error: Exception | None = None,
    ) -> None:
        self.draft = draft
        self.error = error
        self.calls = []

    async def generate(
        self,
        *,
        job_id,
        profile_id,
    ) -> TailoredResumeDraft:
        self.calls.append((job_id, profile_id))

        if self.error is not None:
            raise self.error

        assert self.draft is not None

        return self.draft


async def _create_job(
    database_session,
) -> Job:
    job = Job(
        title="Senior Computer Vision Engineer",
        company="Example Manufacturing",
        location="Abu Dhabi",
        job_url=f"https://example.com/jobs/{uuid4()}",
        description=("Build production visual-inspection systems."),
        source="manual",
    )

    database_session.add(job)
    await database_session.commit()
    await database_session.refresh(job)

    return job


async def _create_profile(
    database_session,
) -> CandidateProfile:
    profile = CandidateProfile(
        full_name="API Candidate",
        email="candidate@example.com",
        phone="+971500000000",
        location="Abu Dhabi",
        linkedin_url=("https://linkedin.com/in/candidate"),
        total_experience_years=8,
    )

    database_session.add(profile)
    await database_session.commit()
    await database_session.refresh(profile)

    return profile


def _draft(
    *,
    job_id,
    profile_id,
    target_title: str = ("Senior Computer Vision Engineer"),
) -> TailoredResumeDraft:
    evidence = SelectedResumeEvidence(
        evidence_id=uuid4(),
        profile_id=profile_id,
        evidence_type="project",
        title="Defect detection",
        content=("Reduced manufacturing inspection time " "from 10 minutes to 5 seconds."),
        source_type="candidate_project",
        source_id=uuid4(),
        metadata_json=('{"origin":"candidate_profile"}'),
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
        job_id=job_id,
        profile_id=profile_id,
        header=ResumeHeader(
            full_name="API Candidate",
            email="candidate@example.com",
            phone="+971500000000",
            location="Abu Dhabi",
            linkedin_url=("https://linkedin.com/in/candidate"),
        ),
        target_title=target_title,
        professional_summary=[
            GroundedResumeStatement(
                text=("Computer vision engineer experienced " "in manufacturing inspection."),
                evidence_ids=[evidence.evidence_id],
            )
        ],
        sections=[],
        skills=[],
        evidence_catalog=[evidence],
        generator_provider="fake",
        generator_model="fake-resume-model",
    )


def _configure_generation(
    monkeypatch,
    *,
    draft: TailoredResumeDraft | None = None,
    error: Exception | None = None,
) -> StubGenerationService:
    generation_service = StubGenerationService(
        draft=draft,
        error=error,
    )

    monkeypatch.setattr(
        "app.api.routes.tailored_resumes." "create_embedding_provider",
        lambda: object(),
    )

    monkeypatch.setattr(
        "app.api.routes.tailored_resumes." "create_resume_generation_provider",
        lambda: object(),
    )

    monkeypatch.setattr(
        "app.api.routes.tailored_resumes." "TailoredResumeGenerationService",
        lambda *args, **kwargs: generation_service,
    )

    return generation_service


def _collection_url(
    job_id,
    profile_id,
) -> str:
    return f"/api/v1/jobs/{job_id}" f"/tailored-resumes/{profile_id}"


@pytest.mark.asyncio
async def test_generate_tailored_resume_returns_201(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    draft = _draft(
        job_id=job.id,
        profile_id=profile.id,
    )

    generation_service = _configure_generation(
        monkeypatch,
        draft=draft,
    )

    response = await async_client.post(
        _collection_url(
            job.id,
            profile.id,
        )
    )

    assert response.status_code == 201

    body = response.json()

    assert body["job_id"] == str(job.id)
    assert body["profile_id"] == str(profile.id)
    assert body["status"] == "generated"
    assert body["generator_provider"] == "fake"
    assert body["generator_model"] == ("fake-resume-model")

    assert generation_service.calls == [(job.id, profile.id)]


@pytest.mark.asyncio
async def test_generated_response_contains_provenance(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    draft = _draft(
        job_id=job.id,
        profile_id=profile.id,
    )

    _configure_generation(
        monkeypatch,
        draft=draft,
    )

    response = await async_client.post(
        _collection_url(
            job.id,
            profile.id,
        )
    )

    assert response.status_code == 201

    content = response.json()["structured_content"]

    evidence = content["evidence_catalog"][0]

    assert evidence["evidence_id"] == str(draft.evidence_catalog[0].evidence_id)

    assert evidence["source_id"] == str(draft.evidence_catalog[0].source_id)

    assert evidence["requirement_traces"][0]["requirement"] == "Visual inspection automation"

    assert content["professional_summary"][0]["evidence_ids"][0] == evidence["evidence_id"]


@pytest.mark.asyncio
async def test_get_tailored_resume_returns_record(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    _configure_generation(
        monkeypatch,
        draft=_draft(
            job_id=job.id,
            profile_id=profile.id,
        ),
    )

    created_response = await async_client.post(
        _collection_url(
            job.id,
            profile.id,
        )
    )

    resume_id = created_response.json()["id"]

    response = await async_client.get(f"{_collection_url(job.id, profile.id)}" f"/{resume_id}")

    assert response.status_code == 200
    assert response.json()["id"] == resume_id


@pytest.mark.asyncio
async def test_list_tailored_resumes(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    for index in range(3):
        _configure_generation(
            monkeypatch,
            draft=_draft(
                job_id=job.id,
                profile_id=profile.id,
                target_title=f"Resume {index}",
            ),
        )

        response = await async_client.post(
            _collection_url(
                job.id,
                profile.id,
            )
        )

        assert response.status_code == 201

    response = await async_client.get(
        _collection_url(
            job.id,
            profile.id,
        ),
        params={
            "limit": 2,
            "offset": 0,
        },
    )

    # assert response.status_code == 200
    assert response.status_code == 200, response.json()
    body = response.json()

    assert body["total"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 0
    assert len(body["items"]) == 2
    assert body["items"][0]["evidence_count"] == 1


@pytest.mark.asyncio
async def test_unknown_or_cross_scoped_resume_returns_404(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    _configure_generation(
        monkeypatch,
        draft=_draft(
            job_id=job.id,
            profile_id=profile.id,
        ),
    )

    created_response = await async_client.post(
        _collection_url(
            job.id,
            profile.id,
        )
    )

    resume_id = created_response.json()["id"]

    response = await async_client.get(f"{_collection_url(job.id, uuid4())}" f"/{resume_id}")

    assert response.status_code == 404
    assert response.json()["detail"] == ("Tailored resume not found")


@pytest.mark.asyncio
async def test_invalid_pagination_returns_422(
    async_client,
) -> None:
    response = await async_client.get(
        _collection_url(
            uuid4(),
            uuid4(),
        ),
        params={
            "limit": 0,
            "offset": -1,
        },
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_uuid_returns_422(
    async_client,
) -> None:
    response = await async_client.post("/api/v1/jobs/not-a-uuid/" f"tailored-resumes/{uuid4()}")

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_provider_configuration_error_returns_503(
    async_client,
    monkeypatch,
) -> None:
    def raise_configuration_error():
        raise EmbeddingProviderConfigurationError("Invalid embedding configuration")

    monkeypatch.setattr(
        "app.api.routes.tailored_resumes." "create_embedding_provider",
        raise_configuration_error,
    )

    response = await async_client.post(
        _collection_url(
            uuid4(),
            uuid4(),
        )
    )

    assert response.status_code == 503
    assert response.json()["detail"] == ("Invalid embedding configuration")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "error",
        "expected_status",
        "expected_detail",
    ),
    [
        (
            MatchingJobNotFoundError(),
            404,
            "Job not found",
        ),
        (
            MatchingRequirementsNotFoundError(),
            404,
            ("Structured job requirements " "not found"),
        ),
        (
            TailoredResumeProfileNotFoundError(),
            404,
            "Candidate profile not found",
        ),
        (
            MatchingCandidateEvidenceNotFoundError(),
            404,
            "Candidate evidence not found",
        ),
        (
            HybridEmbeddedEvidenceNotFoundError(),
            409,
            ("Compatible candidate evidence " "embeddings not found"),
        ),
        (
            TailoredResumeEvidenceNotFoundError(),
            422,
            ("No sufficiently relevant evidence " "was found for resume generation"),
        ),
        (
            TailoredResumeValidationError(),
            422,
            ("Generated resume failed " "provenance validation"),
        ),
        (
            SemanticEvidenceProviderFailureError(),
            502,
            "Resume generation provider failed",
        ),
        (
            TailoredResumeProviderFailureError(),
            502,
            "Resume generation provider failed",
        ),
    ],
)
async def test_generation_errors_are_mapped(
    async_client,
    monkeypatch,
    error,
    expected_status,
    expected_detail,
) -> None:
    _configure_generation(
        monkeypatch,
        error=error,
    )

    response = await async_client.post(
        _collection_url(
            uuid4(),
            uuid4(),
        )
    )

    assert response.status_code == (expected_status)

    assert response.json()["detail"] == (expected_detail)


@pytest.mark.asyncio
async def test_health_endpoint_still_works(
    async_client,
) -> None:
    response = await async_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


@pytest.mark.asyncio
async def test_get_tailored_resume_markdown(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    draft = _draft(
        job_id=job.id,
        profile_id=profile.id,
    )

    _configure_generation(
        monkeypatch,
        draft=draft,
    )

    created_response = await async_client.post(
        _collection_url(
            job.id,
            profile.id,
        )
    )

    assert created_response.status_code == 201

    resume_id = created_response.json()["id"]

    response = await async_client.get(
        f"{_collection_url(job.id, profile.id)}" f"/{resume_id}/markdown"
    )

    # assert response.status_code == 200
    assert response.status_code == 200, response.json()

    assert response.headers["content-type"].startswith("text/markdown")

    assert "# API Candidate" in response.text

    assert "## Professional Summary" in response.text

    assert "Senior Computer Vision Engineer" in response.text


@pytest.mark.asyncio
async def test_markdown_endpoint_excludes_provenance(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    draft = _draft(
        job_id=job.id,
        profile_id=profile.id,
    )

    evidence = draft.evidence_catalog[0]

    _configure_generation(
        monkeypatch,
        draft=draft,
    )

    created_response = await async_client.post(
        _collection_url(
            job.id,
            profile.id,
        )
    )

    resume_id = created_response.json()["id"]

    response = await async_client.get(
        f"{_collection_url(job.id, profile.id)}" f"/{resume_id}/markdown"
    )

    # assert response.status_code == 200
    assert response.status_code == 200, response.json()
    assert str(evidence.evidence_id) not in (response.text)
    assert str(evidence.source_id) not in (response.text)
    assert "requirement_traces" not in (response.text)


@pytest.mark.asyncio
async def test_unknown_markdown_resume_returns_404(
    async_client,
) -> None:
    response = await async_client.get(f"{_collection_url(uuid4(), uuid4())}" f"/{uuid4()}/markdown")

    assert response.status_code == 404
    assert response.json()["detail"] == ("Tailored resume not found")


@pytest.mark.asyncio
async def test_get_tailored_resume_pdf(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)
    draft = _draft(job_id=job.id, profile_id=profile.id)
    _configure_generation(monkeypatch, draft=draft)

    created_response = await async_client.post(
        _collection_url(job.id, profile.id)
    )
    assert created_response.status_code == 201
    resume_id = created_response.json()["id"]

    response = await async_client.get(
        f"{_collection_url(job.id, profile.id)}/{resume_id}/pdf"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF-")
    assert (
        f"tailored-resume-{resume_id}.pdf"
        in response.headers["content-disposition"]
    )
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.asyncio
async def test_pdf_endpoint_contains_extractable_text(
    async_client,
    database_session,
    monkeypatch,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)
    draft = _draft(job_id=job.id, profile_id=profile.id)
    evidence = draft.evidence_catalog[0]
    _configure_generation(monkeypatch, draft=draft)

    created_response = await async_client.post(
        _collection_url(job.id, profile.id)
    )
    assert created_response.status_code == 201
    resume_id = created_response.json()["id"]

    response = await async_client.get(
        f"{_collection_url(job.id, profile.id)}/{resume_id}/pdf"
    )
    assert response.status_code == 200

    reader = PdfReader(BytesIO(response.content))
    extracted_text = "\n".join(
        page.extract_text() or "" for page in reader.pages
    )

    assert "API Candidate" in extracted_text
    assert "Senior Computer Vision Engineer" in extracted_text
    assert (
        "Computer vision engineer experienced in manufacturing inspection."
        in extracted_text
    )
    assert str(evidence.evidence_id) not in extracted_text
    assert str(evidence.source_id) not in extracted_text
    assert "requirement_traces" not in extracted_text


@pytest.mark.asyncio
async def test_unknown_pdf_resume_returns_404(
    async_client,
) -> None:
    response = await async_client.get(
        f"{_collection_url(uuid4(), uuid4())}/{uuid4()}/pdf"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Tailored resume not found"
