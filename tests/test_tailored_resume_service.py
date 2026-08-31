from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import func, select

from app.models.candidate_profile import CandidateProfile
from app.models.job import Job
from app.models.tailored_resume import TailoredResume
from app.schemas.tailored_resume import (
    GroundedResumeStatement,
    ResumeHeader,
    ResumeRequirementTrace,
    SelectedResumeEvidence,
    TailoredResumeDraft,
)
from app.services.tailored_resume_service import (
    TailoredResumeNotFoundError,
    TailoredResumeService,
)


class StubGenerationService:
    def __init__(
        self,
        draft: TailoredResumeDraft,
    ) -> None:
        self.draft = draft
        self.calls: list[tuple] = []

    async def generate(
        self,
        *,
        job_id,
        profile_id,
    ) -> TailoredResumeDraft:
        self.calls.append((job_id, profile_id))
        return self.draft


class FailingGenerationService:
    async def generate(
        self,
        *,
        job_id,
        profile_id,
    ) -> TailoredResumeDraft:
        raise RuntimeError("Generation failed")


class CommitFailureSession:
    def __init__(self) -> None:
        self.added = None
        self.rollback_called = False

    def add(self, record) -> None:
        self.added = record

    async def commit(self) -> None:
        raise RuntimeError("Commit failed")

    async def refresh(self, record) -> None:
        raise AssertionError(
            "refresh must not run after commit failure"
        )

    async def rollback(self) -> None:
        self.rollback_called = True


async def _create_job(
    database_session,
) -> Job:
    job = Job(
        title="Senior Computer Vision Engineer",
        company="Example Manufacturing",
        location="Abu Dhabi",
        job_url=f"https://example.com/jobs/{uuid4()}",
        description=(
            "Build production visual-inspection systems."
        ),
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
        full_name="Test Candidate",
        email="candidate@example.com",
        phone="+971500000000",
        location="Abu Dhabi",
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
    target_title: str = (
        "Senior Computer Vision Engineer"
    ),
) -> TailoredResumeDraft:
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
        metadata_json=(
            '{"origin":"candidate_profile"}'
        ),
        selection_score=0.94,
        requirement_traces=[
            ResumeRequirementTrace(
                category="required_experience",
                requirement=(
                    "Visual inspection automation"
                ),
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
            full_name="Test Candidate",
            email="candidate@example.com",
            location="Abu Dhabi",
        ),
        target_title=target_title,
        professional_summary=[
            GroundedResumeStatement(
                text=(
                    "Computer vision engineer experienced "
                    "in manufacturing inspection."
                ),
                evidence_ids=[
                    evidence.evidence_id
                ],
            )
        ],
        sections=[],
        skills=[],
        evidence_catalog=[evidence],
        generator_provider="fake",
        generator_model="fake-resume-model",
    )


@pytest.mark.asyncio
async def test_generate_and_save_persists_resume(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    draft = _draft(
        job_id=job.id,
        profile_id=profile.id,
    )

    generation_service = StubGenerationService(
        draft
    )

    service = TailoredResumeService(
        database_session,
        generation_service,  # type: ignore[arg-type]
    )

    result = await service.generate_and_save(
        job_id=job.id,
        profile_id=profile.id,
    )

    assert generation_service.calls == [
        (job.id, profile.id)
    ]

    assert result.job_id == job.id
    assert result.profile_id == profile.id
    assert result.status == "generated"
    assert result.structured_content == draft
    assert result.generator_provider == "fake"
    assert result.generator_model == (
        "fake-resume-model"
    )

    stored = await database_session.get(
        TailoredResume,
        result.id,
    )

    assert stored is not None
    assert stored.structured_content == (
        draft.model_dump(mode="json")
    )


@pytest.mark.asyncio
async def test_persisted_resume_preserves_provenance(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    draft = _draft(
        job_id=job.id,
        profile_id=profile.id,
    )

    service = TailoredResumeService(
        database_session,
        StubGenerationService(
            draft
        ),  # type: ignore[arg-type]
    )

    result = await service.generate_and_save(
        job_id=job.id,
        profile_id=profile.id,
    )

    original_evidence = draft.evidence_catalog[0]
    persisted_evidence = (
        result.structured_content
        .evidence_catalog[0]
    )

    assert persisted_evidence.evidence_id == (
        original_evidence.evidence_id
    )

    assert persisted_evidence.source_id == (
        original_evidence.source_id
    )

    assert (
        persisted_evidence
        .requirement_traces[0]
        .requirement
        == "Visual inspection automation"
    )


@pytest.mark.asyncio
async def test_generation_metadata_is_stored(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    draft = _draft(
        job_id=job.id,
        profile_id=profile.id,
    )

    service = TailoredResumeService(
        database_session,
        StubGenerationService(
            draft
        ),  # type: ignore[arg-type]
    )

    result = await service.generate_and_save(
        job_id=job.id,
        profile_id=profile.id,
    )

    assert result.generation_metadata == {
        "schema_version": 1,
        "provenance_version": 1,
        "evidence_count": 1,
        "statement_count": 1,
    }


@pytest.mark.asyncio
async def test_multiple_generations_create_immutable_records(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    first_service = TailoredResumeService(
        database_session,
        StubGenerationService(
            _draft(
                job_id=job.id,
                profile_id=profile.id,
                target_title="First Resume",
            )
        ),  # type: ignore[arg-type]
    )

    first = await first_service.generate_and_save(
        job_id=job.id,
        profile_id=profile.id,
    )

    second_service = TailoredResumeService(
        database_session,
        StubGenerationService(
            _draft(
                job_id=job.id,
                profile_id=profile.id,
                target_title="Second Resume",
            )
        ),  # type: ignore[arg-type]
    )

    second = await second_service.generate_and_save(
        job_id=job.id,
        profile_id=profile.id,
    )

    assert first.id != second.id

    count = await database_session.scalar(
        select(
            func.count(TailoredResume.id)
        )
    )

    assert count == 2


@pytest.mark.asyncio
async def test_get_resume_is_scoped(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    draft = _draft(
        job_id=job.id,
        profile_id=profile.id,
    )

    service = TailoredResumeService(
        database_session,
        StubGenerationService(
            draft
        ),  # type: ignore[arg-type]
    )

    created = await service.generate_and_save(
        job_id=job.id,
        profile_id=profile.id,
    )

    retrieved = await service.get_resume(
        resume_id=created.id,
        job_id=job.id,
        profile_id=profile.id,
    )

    assert retrieved == created

    with pytest.raises(
        TailoredResumeNotFoundError
    ):
        await service.get_resume(
            resume_id=created.id,
            job_id=job.id,
            profile_id=uuid4(),
        )


@pytest.mark.asyncio
async def test_list_resumes_supports_pagination(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    for index in range(3):
        service = TailoredResumeService(
            database_session,
            StubGenerationService(
                _draft(
                    job_id=job.id,
                    profile_id=profile.id,
                    target_title=(
                        f"Resume {index}"
                    ),
                )
            ),  # type: ignore[arg-type]
        )

        await service.generate_and_save(
            job_id=job.id,
            profile_id=profile.id,
        )

    service = TailoredResumeService(
        database_session,
        StubGenerationService(
            _draft(
                job_id=job.id,
                profile_id=profile.id,
            )
        ),  # type: ignore[arg-type]
    )

    first_page = await service.list_resumes(
        job_id=job.id,
        profile_id=profile.id,
        limit=2,
        offset=0,
    )

    second_page = await service.list_resumes(
        job_id=job.id,
        profile_id=profile.id,
        limit=2,
        offset=2,
    )

    assert first_page.total == 3
    assert len(first_page.items) == 2
    assert len(second_page.items) == 1

    all_ids = {
        item.id
        for item in [
            *first_page.items,
            *second_page.items,
        ]
    }

    assert len(all_ids) == 3


@pytest.mark.asyncio
async def test_invalid_pagination_is_rejected(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    service = TailoredResumeService(
        database_session,
        StubGenerationService(
            _draft(
                job_id=job.id,
                profile_id=profile.id,
            )
        ),  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match="limit",
    ):
        await service.list_resumes(
            job_id=job.id,
            profile_id=profile.id,
            limit=0,
        )

    with pytest.raises(
        ValueError,
        match="offset",
    ):
        await service.list_resumes(
            job_id=job.id,
            profile_id=profile.id,
            offset=-1,
        )


@pytest.mark.asyncio
async def test_job_delete_cascades_to_resumes(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    service = TailoredResumeService(
        database_session,
        StubGenerationService(
            _draft(
                job_id=job.id,
                profile_id=profile.id,
            )
        ),  # type: ignore[arg-type]
    )

    created = await service.generate_and_save(
        job_id=job.id,
        profile_id=profile.id,
    )

    await database_session.delete(job)
    await database_session.commit()

    stored = await database_session.get(
        TailoredResume,
        created.id,
    )

    assert stored is None


@pytest.mark.asyncio
async def test_generation_failure_does_not_create_record(
    database_session,
) -> None:
    job = await _create_job(database_session)
    profile = await _create_profile(database_session)

    service = TailoredResumeService(
        database_session,
        FailingGenerationService(),  # type: ignore[arg-type]
    )

    with pytest.raises(
        RuntimeError,
        match="Generation failed",
    ):
        await service.generate_and_save(
            job_id=job.id,
            profile_id=profile.id,
        )

    count = await database_session.scalar(
        select(
            func.count(TailoredResume.id)
        )
    )

    assert count == 0


@pytest.mark.asyncio
async def test_commit_failure_rolls_back() -> None:
    job_id = uuid4()
    profile_id = uuid4()

    draft = _draft(
        job_id=job_id,
        profile_id=profile_id,
    )

    session = CommitFailureSession()

    service = TailoredResumeService(
        session,  # type: ignore[arg-type]
        StubGenerationService(
            draft
        ),  # type: ignore[arg-type]
    )

    with pytest.raises(
        RuntimeError,
        match="Commit failed",
    ):
        await service.generate_and_save(
            job_id=job_id,
            profile_id=profile_id,
        )

    assert session.added is not None
    assert session.rollback_called is True