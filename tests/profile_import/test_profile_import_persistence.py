"""SQLite transaction tests; PostgreSQL integration is additionally required in deployment."""

import json
from uuid import uuid4

import pytest
import pytest_asyncio
from app.models.candidate_document import CandidateDocument
from app.models.candidate_evidence import CandidateEvidence
from app.models.candidate_profile import CandidateProfile
from app.schemas.profile_import import ProfileImportConfirm, ProfileImportData
from app.services import profile_import_service as service
from app.services.candidate_profile_service import CandidateProfileService
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest_asyncio.fixture
async def database():
    pytest.importorskip("aiosqlite")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    tables = [
        CandidateProfile.__table__,
        CandidateDocument.__table__,
        CandidateEvidence.__table__,
    ]
    tables += [model.__table__ for _, model, _ in service.COLLECTIONS.values()]
    async with engine.begin() as connection:
        await connection.run_sync(
            lambda sync: CandidateProfile.metadata.create_all(sync, tables=tables)
        )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        profile = CandidateProfile(id=uuid4(), full_name="Candidate", email="saved@example.com")
        document = CandidateDocument(
            id=uuid4(),
            profile_id=profile.id,
            filename="resume.pdf",
            content_type="application/pdf",
            storage_path="unused.pdf",
            file_size=1,
            extracted_text="Lead Employer",
            extraction_status="extracted",
        )
        session.add_all([profile, document])
        await session.commit()
        yield session, profile.id, document.id
    await engine.dispose()


async def request_for(session, profile_id, document_id):
    current = service.snapshot(await CandidateProfileService(session).get_profile(profile_id))
    return ProfileImportConfirm(
        profile_id=profile_id,
        document_id=document_id,
        base_revision=service.revision(current),
        data=current,
        confirmed=True,
    )


@pytest.mark.asyncio
async def test_confirm_roundtrip_and_repeat_without_duplicates(database):
    session, profile_id, document_id = database
    request = await request_for(session, profile_id, document_id)
    request.data = ProfileImportData(
        profile={"full_name": "Candidate", "email": "saved@example.com"},
        experiences=[{"company": "Employer", "role": "Lead", "achievements": "Built RAG"}],
        education=[{"institution": "University", "degree": "PhD", "description": "2016 - 2021"}],
        skills=[{"name": "Python"}],
        projects=[{"name": "RAG", "project_url": "https://example.com"}],
    )
    result = await service.confirm_review(session, request)
    assert result["saved"] and result["evidence_records"] == 4
    current = service.snapshot(await CandidateProfileService(session).get_profile(profile_id))
    assert current.experiences[0].company == "Employer"
    assert current.education[0].degree == "PhD"
    assert str(current.projects[0].project_url) == "https://example.com/"
    records = (await session.scalars(select(CandidateEvidence))).all()
    assert all(
        json.loads(r.metadata_json)["profile_revision"] == service.revision(current)
        for r in records
    )
    first_id = current.experiences[0].id
    repeated = await request_for(session, profile_id, document_id)
    await service.confirm_review(session, repeated)
    current = service.snapshot(await CandidateProfileService(session).get_profile(profile_id))
    assert len(current.experiences) == 1 and current.experiences[0].id == first_id
    assert len((await session.scalars(select(CandidateEvidence))).all()) == 4


@pytest.mark.asyncio
async def test_deletion_is_explicit_and_manual_evidence_survives(database):
    session, profile_id, document_id = database
    manual_id, old_id = uuid4(), uuid4()
    session.add_all(
        [
            CandidateEvidence(
                id=manual_id,
                profile_id=profile_id,
                source_type="manual",
                evidence_type="skill",
                title="Manual",
                content="Manual fact",
            ),
            CandidateEvidence(
                id=old_id,
                profile_id=profile_id,
                source_type="candidate_document_llm",
                evidence_type="skill",
                title="Old",
                content="Old automatic fact",
            ),
        ]
    )
    await session.commit()
    request = await request_for(session, profile_id, document_id)
    request.data.skills = ProfileImportData(
        profile={"full_name": "Candidate"}, skills=[{"name": "Python"}]
    ).skills
    await service.confirm_review(session, request)
    request = await request_for(session, profile_id, document_id)
    request.data.skills = []
    await service.confirm_review(session, request)
    remaining = (await session.scalars(select(CandidateEvidence))).all()
    assert [r.id for r in remaining] == [manual_id]
    assert not service.snapshot(
        await CandidateProfileService(session).get_profile(profile_id)
    ).skills


@pytest.mark.asyncio
async def test_foreign_entity_id_is_rejected(database):
    session, profile_id, document_id = database
    request = await request_for(session, profile_id, document_id)
    request.data.skills = ProfileImportData(
        profile={"full_name": "Candidate"}, skills=[{"id": uuid4(), "name": "Foreign"}]
    ).skills
    with pytest.raises(service.ProfileImportError) as error:
        await service.confirm_review(session, request)
    assert error.value.status_code == 422
    assert not (await session.scalars(select(CandidateEvidence))).all()


@pytest.mark.asyncio
async def test_failed_commit_rolls_back_profile_and_evidence(database, monkeypatch):
    session, profile_id, document_id = database
    request = await request_for(session, profile_id, document_id)
    request.data.profile.full_name = "Must roll back"

    async def fail():
        raise RuntimeError("Simulated database failure")

    monkeypatch.setattr(session, "commit", fail)
    with pytest.raises(RuntimeError):
        await service.confirm_review(session, request)
    profile = await CandidateProfileService(session).get_profile(profile_id)
    assert profile.full_name == "Candidate"
    assert not (await session.scalars(select(CandidateEvidence))).all()
