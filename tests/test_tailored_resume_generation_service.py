from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.llm.resume_provider import (
    ResumeGenerationProviderError,
)
from app.models.candidate_profile import CandidateProfile
from app.models.job import Job
from app.schemas.tailored_resume import (
    GroundedResumeStatement,
    ResumeGroundingBundle,
    ResumeRequirementTrace,
    SelectedResumeEvidence,
    TailoredResumeContent,
)
from app.services.tailored_resume_generation_service import (
    TailoredResumeGenerationService,
    TailoredResumeJobNotFoundError,
    TailoredResumeProfileNotFoundError,
    TailoredResumeProviderFailureError,
    TailoredResumeValidationError,
)


@pytest.fixture(autouse=True)
def clean_database() -> None:
    """Override the global database fixture for unit tests."""


class StubScalarResult:
    def __init__(self, value) -> None:
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class StubSession:
    def __init__(self, values: list[object | None]) -> None:
        self.values = list(values)
        self.statements: list[object] = []

    async def execute(self, statement):
        self.statements.append(statement)
        return StubScalarResult(self.values.pop(0))


class StubEmbeddingProvider:
    provider_name = "fake"
    model_name = "fake-embedding-model"
    dimensions = 768

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:
        raise AssertionError("Embedding should not be called directly")

    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        raise AssertionError("Embedding should not be called directly")


class StubGroundingService:
    def __init__(
        self,
        bundle: ResumeGroundingBundle,
    ) -> None:
        self.bundle = bundle
        self.calls: list[tuple[UUID, UUID]] = []

    async def build_bundle(
        self,
        *,
        job_id: UUID,
        profile_id: UUID,
    ) -> ResumeGroundingBundle:
        self.calls.append((job_id, profile_id))
        return self.bundle


class StubResumeProvider:
    provider_name = "fake"
    model_name = "fake-resume-model"

    def __init__(
        self,
        content: TailoredResumeContent,
    ) -> None:
        self.content = content
        self.contexts = []

    async def generate_resume(self, context):
        self.contexts.append(context)
        return self.content


class FailingResumeProvider:
    provider_name = "fake"
    model_name = "failing-model"

    async def generate_resume(self, context):
        raise ResumeGenerationProviderError(
            "Provider unavailable"
        )


def _job() -> Job:
    return Job(
        id=uuid4(),
        title="Senior Computer Vision Engineer",
        company="Example Manufacturing",
        location="Abu Dhabi",
        job_url=f"https://example.com/jobs/{uuid4()}",
        description=(
            "Build production manufacturing inspection systems."
        ),
        source="manual",
    )


def _profile() -> CandidateProfile:
    return CandidateProfile(
        id=uuid4(),
        full_name="Test Candidate",
        email="candidate@example.com",
        phone="+971500000000",
        location="Abu Dhabi",
        linkedin_url="https://linkedin.com/in/candidate",
        github_url="https://github.com/candidate",
        portfolio_url="https://candidate.example.com",
        total_experience_years=8,
    )


def _grounding(
    *,
    job_id: UUID,
    profile_id: UUID,
) -> ResumeGroundingBundle:
    evidence = SelectedResumeEvidence(
        evidence_id=uuid4(),
        profile_id=profile_id,
        evidence_type="project",
        title="Defect detection",
        content=(
            "Reduced manufacturing inspection time "
            "from 10 minutes to 5 seconds."
        ),
        source_type="candidate_project",
        source_id=uuid4(),
        metadata_json='{"origin":"candidate_profile"}',
        selection_score=0.94,
        requirement_traces=[
            ResumeRequirementTrace(
                category="required_experience",
                requirement="Visual inspection automation",
                deterministic_score=1.0,
                semantic_score=0.85,
                hybrid_score=0.94,
                rank=1,
            )
        ],
    )

    return ResumeGroundingBundle(
        job_id=job_id,
        profile_id=profile_id,
        hybrid_overall_score=92,
        matched_requirements=[
            "Visual inspection automation",
        ],
        missing_requirements=[
            "Kubernetes",
        ],
        selected_evidence=[evidence],
        embedding_provider="fake",
        embedding_model="fake-embedding-model",
        minimum_evidence_score=0.6,
        max_evidence_per_requirement=3,
        max_total_evidence=12,
    )


def _content(
    evidence_id: UUID,
) -> TailoredResumeContent:
    return TailoredResumeContent(
        target_title="Senior Computer Vision Engineer",
        professional_summary=[
            GroundedResumeStatement(
                text=(
                    "Computer vision engineer experienced "
                    "in manufacturing inspection automation."
                ),
                evidence_ids=[evidence_id],
            )
        ],
    )


def _service(
    *,
    job: Job,
    profile: CandidateProfile,
    grounding: ResumeGroundingBundle,
    provider,
):
    session = StubSession([job, profile])
    grounding_service = StubGroundingService(grounding)

    service = TailoredResumeGenerationService(
        session,  # type: ignore[arg-type]
        StubEmbeddingProvider(),
        provider,
        grounding_service=grounding_service,  # type: ignore[arg-type]
    )

    return service, session, grounding_service


@pytest.mark.asyncio
async def test_generate_returns_valid_grounded_draft() -> None:
    job = _job()
    profile = _profile()
    grounding = _grounding(
        job_id=job.id,
        profile_id=profile.id,
    )
    evidence = grounding.selected_evidence[0]

    provider = StubResumeProvider(
        _content(evidence.evidence_id)
    )

    service, _, grounding_service = _service(
        job=job,
        profile=profile,
        grounding=grounding,
        provider=provider,
    )

    result = await service.generate(
        job_id=job.id,
        profile_id=profile.id,
    )

    assert grounding_service.calls == [
        (job.id, profile.id)
    ]

    assert result.job_id == job.id
    assert result.profile_id == profile.id
    assert result.target_title == job.title

    assert result.header.full_name == profile.full_name
    assert result.header.email == profile.email
    assert result.header.phone == profile.phone

    assert result.evidence_catalog == (
        grounding.selected_evidence
    )

    assert result.generator_provider == "fake"
    assert result.generator_model == "fake-resume-model"


@pytest.mark.asyncio
async def test_generation_context_contains_job_and_profile_facts() -> None:
    job = _job()
    profile = _profile()
    grounding = _grounding(
        job_id=job.id,
        profile_id=profile.id,
    )

    provider = StubResumeProvider(
        _content(
            grounding.selected_evidence[0].evidence_id
        )
    )

    service, _, _ = _service(
        job=job,
        profile=profile,
        grounding=grounding,
        provider=provider,
    )

    await service.generate(
        job_id=job.id,
        profile_id=profile.id,
    )

    context = provider.contexts[0]

    assert context.job_title == job.title
    assert context.company == job.company
    assert context.location == job.location
    assert context.job_description == job.description
    assert context.header.full_name == profile.full_name
    assert context.grounding == grounding


@pytest.mark.asyncio
async def test_unknown_job_is_rejected() -> None:
    profile = _profile()
    job_id = uuid4()

    grounding = _grounding(
        job_id=job_id,
        profile_id=profile.id,
    )

    service = TailoredResumeGenerationService(
        StubSession([None]),  # type: ignore[arg-type]
        StubEmbeddingProvider(),
        StubResumeProvider(
            _content(
                grounding.selected_evidence[0].evidence_id
            )
        ),
        grounding_service=StubGroundingService(
            grounding
        ),  # type: ignore[arg-type]
    )

    with pytest.raises(
        TailoredResumeJobNotFoundError,
        match="Job not found",
    ):
        await service.generate(
            job_id=job_id,
            profile_id=profile.id,
        )


@pytest.mark.asyncio
async def test_unknown_profile_is_rejected() -> None:
    job = _job()
    profile_id = uuid4()

    grounding = _grounding(
        job_id=job.id,
        profile_id=profile_id,
    )

    service = TailoredResumeGenerationService(
        StubSession([job, None]),  # type: ignore[arg-type]
        StubEmbeddingProvider(),
        StubResumeProvider(
            _content(
                grounding.selected_evidence[0].evidence_id
            )
        ),
        grounding_service=StubGroundingService(
            grounding
        ),  # type: ignore[arg-type]
    )

    with pytest.raises(
        TailoredResumeProfileNotFoundError,
        match="Candidate profile not found",
    ):
        await service.generate(
            job_id=job.id,
            profile_id=profile_id,
        )


@pytest.mark.asyncio
async def test_provider_failure_is_wrapped() -> None:
    job = _job()
    profile = _profile()

    grounding = _grounding(
        job_id=job.id,
        profile_id=profile.id,
    )

    service, _, _ = _service(
        job=job,
        profile=profile,
        grounding=grounding,
        provider=FailingResumeProvider(),
    )

    with pytest.raises(
        TailoredResumeProviderFailureError,
        match="provider failed",
    ):
        await service.generate(
            job_id=job.id,
            profile_id=profile.id,
        )


@pytest.mark.asyncio
async def test_unknown_evidence_reference_is_rejected() -> None:
    job = _job()
    profile = _profile()

    grounding = _grounding(
        job_id=job.id,
        profile_id=profile.id,
    )

    provider = StubResumeProvider(
        _content(uuid4())
    )

    service, _, _ = _service(
        job=job,
        profile=profile,
        grounding=grounding,
        provider=provider,
    )

    with pytest.raises(
        TailoredResumeValidationError,
        match="provenance validation",
    ):
        await service.generate(
            job_id=job.id,
            profile_id=profile.id,
        )


@pytest.mark.asyncio
async def test_contact_facts_come_from_profile_not_provider() -> None:
    job = _job()
    profile = _profile()

    grounding = _grounding(
        job_id=job.id,
        profile_id=profile.id,
    )

    provider = StubResumeProvider(
        _content(
            grounding.selected_evidence[0].evidence_id
        )
    )

    service, _, _ = _service(
        job=job,
        profile=profile,
        grounding=grounding,
        provider=provider,
    )

    result = await service.generate(
        job_id=job.id,
        profile_id=profile.id,
    )

    assert result.header.full_name == profile.full_name
    assert result.header.email == profile.email
    assert result.header.linkedin_url == (
        profile.linkedin_url
    )