from uuid import uuid4

import pytest

from app.models.candidate_document import CandidateDocument
from app.models.candidate_profile import CandidateProfile
from app.schemas.candidate_document import CandidateDocumentCreate, CandidateDocumentRead, CandidateDocumentUpdate
from app.services.candidate_document_service import CandidateDocumentNotFoundError, CandidateDocumentService
from app.services.candidate_profile_service import CandidateProfileNotFoundError, CandidateProfileService
from app.schemas.candidate_profile import CandidateProfileCreate


@pytest.mark.asyncio
async def test_candidate_document_model_can_be_created(database_session) -> None:
    profile = CandidateProfile(full_name="Jane Doe")
    document = CandidateDocument(
        profile_id=profile.id,
        filename="resume.pdf",
        content_type="application/pdf",
        storage_path="/tmp/resume.pdf",
        file_size=1234,
    )

    assert document.filename == "resume.pdf"
    assert document.content_type == "application/pdf"
    assert document.extraction_status == "uploaded"
    assert document.profile_id == profile.id


@pytest.mark.asyncio
async def test_create_document_success(database_session) -> None:
    profile_service = CandidateProfileService(database_session)
    profile = await profile_service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    service = CandidateDocumentService(database_session)

    document = await service.create_document(
        profile.id,
        CandidateDocumentCreate(
            filename="resume.pdf",
            content_type="application/pdf",
            storage_path="/tmp/resume.pdf",
            file_size=1234,
        ),
    )

    assert document.id is not None
    assert document.profile_id == profile.id
    assert document.filename == "resume.pdf"
    assert document.extraction_status == "uploaded"


@pytest.mark.asyncio
async def test_create_document_has_correct_profile_id(database_session) -> None:
    profile_service = CandidateProfileService(database_session)
    profile = await profile_service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    service = CandidateDocumentService(database_session)

    document = await service.create_document(
        profile.id,
        CandidateDocumentCreate(
            filename="cv.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            storage_path="/tmp/cv.docx",
            file_size=988,
        ),
    )

    assert document.profile_id == profile.id


@pytest.mark.asyncio
async def test_get_document_retrieves_document(database_session) -> None:
    profile_service = CandidateProfileService(database_session)
    profile = await profile_service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    service = CandidateDocumentService(database_session)
    created = await service.create_document(
        profile.id,
        CandidateDocumentCreate(
            filename="resume.pdf",
            content_type="application/pdf",
            storage_path="/tmp/resume.pdf",
            file_size=1234,
        ),
    )

    retrieved = await service.get_document(profile.id, created.id)

    assert retrieved.id == created.id
    assert retrieved.filename == "resume.pdf"
    assert retrieved.profile_id == profile.id


@pytest.mark.asyncio
async def test_get_document_for_unknown_uuid_raises(database_session) -> None:
    profile_service = CandidateProfileService(database_session)
    profile = await profile_service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    service = CandidateDocumentService(database_session)

    with pytest.raises(CandidateDocumentNotFoundError):
        await service.get_document(profile.id, uuid4())


@pytest.mark.asyncio
async def test_get_document_cannot_access_another_profiles_document(database_session) -> None:
    profile_service = CandidateProfileService(database_session)
    profile_a = await profile_service.create_profile(CandidateProfileCreate(full_name="A"))
    profile_b = await profile_service.create_profile(CandidateProfileCreate(full_name="B"))
    service = CandidateDocumentService(database_session)
    document = await service.create_document(
        profile_b.id,
        CandidateDocumentCreate(
            filename="other.pdf",
            content_type="application/pdf",
            storage_path="/tmp/other.pdf",
            file_size=200,
        ),
    )

    with pytest.raises(CandidateDocumentNotFoundError):
        await service.get_document(profile_a.id, document.id)


@pytest.mark.asyncio
async def test_update_document_updates_extracted_text(database_session) -> None:
    profile_service = CandidateProfileService(database_session)
    profile = await profile_service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    service = CandidateDocumentService(database_session)
    document = await service.create_document(
        profile.id,
        CandidateDocumentCreate(
            filename="resume.pdf",
            content_type="application/pdf",
            storage_path="/tmp/resume.pdf",
            file_size=1234,
        ),
    )

    updated = await service.update_document(
        profile.id,
        document.id,
        CandidateDocumentUpdate(extracted_text="Candidate has Python experience"),
    )

    assert updated.extracted_text == "Candidate has Python experience"


@pytest.mark.asyncio
async def test_update_document_updates_extraction_status(database_session) -> None:
    profile_service = CandidateProfileService(database_session)
    profile = await profile_service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    service = CandidateDocumentService(database_session)
    document = await service.create_document(
        profile.id,
        CandidateDocumentCreate(
            filename="resume.pdf",
            content_type="application/pdf",
            storage_path="/tmp/resume.pdf",
            file_size=1234,
        ),
    )

    updated = await service.update_document(
        profile.id,
        document.id,
        CandidateDocumentUpdate(extraction_status="extracted"),
    )

    assert updated.extraction_status == "extracted"


@pytest.mark.asyncio
async def test_update_document_updates_extraction_error(database_session) -> None:
    profile_service = CandidateProfileService(database_session)
    profile = await profile_service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    service = CandidateDocumentService(database_session)
    document = await service.create_document(
        profile.id,
        CandidateDocumentCreate(
            filename="resume.pdf",
            content_type="application/pdf",
            storage_path="/tmp/resume.pdf",
            file_size=1234,
        ),
    )

    updated = await service.update_document(
        profile.id,
        document.id,
        CandidateDocumentUpdate(extraction_error="Parsing failed"),
    )

    assert updated.extraction_error == "Parsing failed"


@pytest.mark.asyncio
async def test_update_document_cannot_modify_another_profiles_document(database_session) -> None:
    profile_service = CandidateProfileService(database_session)
    profile_a = await profile_service.create_profile(CandidateProfileCreate(full_name="A"))
    profile_b = await profile_service.create_profile(CandidateProfileCreate(full_name="B"))
    service = CandidateDocumentService(database_session)
    document = await service.create_document(
        profile_b.id,
        CandidateDocumentCreate(
            filename="other.pdf",
            content_type="application/pdf",
            storage_path="/tmp/other.pdf",
            file_size=200,
        ),
    )

    with pytest.raises(CandidateDocumentNotFoundError):
        await service.update_document(
            profile_a.id,
            document.id,
            CandidateDocumentUpdate(extracted_text="should-not-work"),
        )


@pytest.mark.asyncio
async def test_delete_document_removes_document(database_session) -> None:
    profile_service = CandidateProfileService(database_session)
    profile = await profile_service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    service = CandidateDocumentService(database_session)
    document = await service.create_document(
        profile.id,
        CandidateDocumentCreate(
            filename="resume.pdf",
            content_type="application/pdf",
            storage_path="/tmp/resume.pdf",
            file_size=1234,
        ),
    )

    await service.delete_document(profile.id, document.id)

    with pytest.raises(CandidateDocumentNotFoundError):
        await service.get_document(profile.id, document.id)


@pytest.mark.asyncio
async def test_delete_document_cannot_delete_another_profiles_document(database_session) -> None:
    profile_service = CandidateProfileService(database_session)
    profile_a = await profile_service.create_profile(CandidateProfileCreate(full_name="A"))
    profile_b = await profile_service.create_profile(CandidateProfileCreate(full_name="B"))
    service = CandidateDocumentService(database_session)
    document = await service.create_document(
        profile_b.id,
        CandidateDocumentCreate(
            filename="other.pdf",
            content_type="application/pdf",
            storage_path="/tmp/other.pdf",
            file_size=200,
        ),
    )

    with pytest.raises(CandidateDocumentNotFoundError):
        await service.delete_document(profile_a.id, document.id)


@pytest.mark.asyncio
async def test_create_document_for_nonexistent_profile_fails(database_session) -> None:
    service = CandidateDocumentService(database_session)

    with pytest.raises(CandidateProfileNotFoundError):
        await service.create_document(
            uuid4(),
            CandidateDocumentCreate(
                filename="resume.pdf",
                content_type="application/pdf",
                storage_path="/tmp/resume.pdf",
                file_size=1234,
            ),
        )


@pytest.mark.asyncio
async def test_delete_candidate_profile_cascades_to_candidate_documents(database_session) -> None:
    profile_service = CandidateProfileService(database_session)
    profile = await profile_service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    service = CandidateDocumentService(database_session)
    document = await service.create_document(
        profile.id,
        CandidateDocumentCreate(
            filename="resume.pdf",
            content_type="application/pdf",
            storage_path="/tmp/resume.pdf",
            file_size=1234,
        ),
    )

    await database_session.delete(profile)
    await database_session.commit()

    remaining = await database_session.get(CandidateDocument, document.id)
    assert remaining is None


@pytest.mark.asyncio
async def test_candidate_document_read_serializes_orm_model(database_session) -> None:
    profile_service = CandidateProfileService(database_session)
    profile = await profile_service.create_profile(CandidateProfileCreate(full_name="Jane Doe"))
    service = CandidateDocumentService(database_session)
    document = await service.create_document(
        profile.id,
        CandidateDocumentCreate(
            filename="resume.pdf",
            content_type="application/pdf",
            storage_path="/tmp/resume.pdf",
            file_size=1234,
            extracted_text="Python, SQL",
            extraction_status="extracted",
            extraction_error=None,
        ),
    )

    payload = CandidateDocumentRead.model_validate(document)

    assert payload.id == document.id
    assert payload.profile_id == profile.id
    assert payload.filename == "resume.pdf"
    assert payload.content_type == "application/pdf"
    assert payload.storage_path == "/tmp/resume.pdf"
    assert payload.file_size == 1234
    assert payload.extracted_text == "Python, SQL"
    assert payload.extraction_status == "extracted"
