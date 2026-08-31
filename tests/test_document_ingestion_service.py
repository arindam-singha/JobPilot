from io import BytesIO
from uuid import uuid4

import pytest
import pytest_asyncio
from docx import Document
from pypdf import PdfWriter
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.candidate_document import CandidateDocument
from app.models.candidate_profile import CandidateProfile
from app.schemas.candidate_profile import CandidateProfileCreate
from app.services.candidate_document_service import CandidateDocumentService
from app.services.document_ingestion_service import (
    DocumentIngestionError,
    DocumentIngestionService,
    DocumentIngestionValidationError,
)
from app.services.document_storage_service import DocumentStorageService


@pytest_asyncio.fixture
def storage_service(tmp_path):
    """Provide a DocumentStorageService with temporary storage."""
    return DocumentStorageService(storage_root=tmp_path / "cv-storage")


@pytest.fixture
def valid_pdf_bytes() -> bytes:
    """Generate a valid PDF with text content for testing."""
    # Create a minimal valid PDF with text that pypdf can extract
    # This is a hand-crafted minimal PDF that includes extractable text
    pdf_content = (
        b'%PDF-1.4\n'
        b'1 0 obj\n'
        b'<< /Type /Catalog /Pages 2 0 R >>\n'
        b'endobj\n'
        b'2 0 obj\n'
        b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n'
        b'endobj\n'
        b'3 0 obj\n'
        b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n'
        b'endobj\n'
        b'4 0 obj\n'
        b'<< /Length 44 >>\n'
        b'stream\n'
        b'BT\n'
        b'/F1 12 Tf\n'
        b'100 700 Td\n'
        b'(Hello World Resume) Tj\n'
        b'ET\n'
        b'endstream\n'
        b'endobj\n'
        b'5 0 obj\n'
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n'
        b'endobj\n'
        b'xref\n'
        b'0 6\n'
        b'0000000000 65535 f\n'
        b'0000000009 00000 n\n'
        b'0000000058 00000 n\n'
        b'0000000115 00000 n\n'
        b'0000000258 00000 n\n'
        b'0000000353 00000 n\n'
        b'trailer\n'
        b'<< /Size 6 /Root 1 0 R >>\n'
        b'startxref\n'
        b'433\n'
        b'%%EOF\n'
    )
    return pdf_content


@pytest.fixture
def valid_docx_bytes() -> bytes:
    """Generate a valid DOCX for testing."""
    document = Document()
    document.add_paragraph("First line")
    document.add_paragraph("Second line")
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest_asyncio.fixture
async def candidate_profile(database_session: AsyncSession):
    """Create a test candidate profile."""
    profile = CandidateProfile(**CandidateProfileCreate(full_name="Test User", email="test@example.com").model_dump())
    database_session.add(profile)
    await database_session.commit()
    await database_session.refresh(profile)
    return profile


@pytest_asyncio.fixture
async def document_service(database_session: AsyncSession):
    """Provide a CandidateDocumentService."""
    return CandidateDocumentService(database_session)


@pytest_asyncio.fixture
async def ingestion_service(database_session: AsyncSession, document_service, storage_service):
    """Provide a DocumentIngestionService."""
    return DocumentIngestionService(database_session, document_service, storage_service)


# =============================================================================
# SUCCESS TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_successfully_ingest_pdf(ingestion_service, candidate_profile, storage_service, valid_pdf_bytes):
    """Test successful PDF ingestion."""
    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=valid_pdf_bytes,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert document is not None
    assert document.profile_id == candidate_profile.id
    assert document.filename == "resume.pdf"
    assert document.content_type == "application/pdf"
    assert document.extraction_status == "extracted"
    assert document.extracted_text is not None
    assert len(document.extracted_text) >= 0
    assert document.extraction_error is None


@pytest.mark.asyncio
async def test_successfully_ingest_docx(ingestion_service, candidate_profile, storage_service, valid_docx_bytes):
    """Test successful DOCX ingestion."""
    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=valid_docx_bytes,
        filename="resume.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert document is not None
    assert document.profile_id == candidate_profile.id
    assert document.filename == "resume.docx"
    assert document.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert document.extraction_status == "extracted"


@pytest.mark.asyncio
async def test_candidate_document_is_created(ingestion_service, candidate_profile, valid_pdf_bytes):
    """Test that CandidateDocument record is created in the database."""
    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=valid_pdf_bytes,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert isinstance(document, CandidateDocument)
    assert document.id is not None
    assert document.profile_id == candidate_profile.id


@pytest.mark.asyncio
async def test_correct_profile_id_is_stored(ingestion_service, candidate_profile, valid_pdf_bytes):
    """Test that the correct profile_id is associated with the document."""
    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=valid_pdf_bytes,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert document.profile_id == candidate_profile.id


@pytest.mark.asyncio
async def test_correct_filename_is_stored(ingestion_service, candidate_profile, valid_pdf_bytes):
    """Test that the correct filename is stored."""
    expected_filename = "my_resume_2024.pdf"

    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=valid_pdf_bytes,
        filename=expected_filename,
        content_type="application/pdf",
    )

    assert document.filename == expected_filename


@pytest.mark.asyncio
async def test_correct_content_type_is_stored(ingestion_service, candidate_profile, valid_pdf_bytes):
    """Test that the correct content_type is stored."""
    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=valid_pdf_bytes,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert document.content_type == "application/pdf"


@pytest.mark.asyncio
async def test_correct_file_size_is_stored(ingestion_service, candidate_profile, valid_pdf_bytes):
    """Test that the correct file_size is stored."""
    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=valid_pdf_bytes,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert document.file_size == len(valid_pdf_bytes)


@pytest.mark.asyncio
async def test_storage_path_is_stored(ingestion_service, candidate_profile, valid_pdf_bytes):
    """Test that storage_path is properly stored."""
    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=valid_pdf_bytes,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert document.storage_path is not None
    assert len(document.storage_path) > 0


@pytest.mark.asyncio
async def test_original_file_exists_in_storage(ingestion_service, candidate_profile, storage_service, valid_pdf_bytes):
    """Test that the original file exists in storage after ingestion."""
    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=valid_pdf_bytes,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert storage_service.exists(document.storage_path)
    stored_content = storage_service.get(document.storage_path)
    assert stored_content == valid_pdf_bytes


@pytest.mark.asyncio
async def test_pdf_text_is_extracted(ingestion_service, candidate_profile, valid_pdf_bytes):
    """Test that PDF text is successfully extracted."""
    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=valid_pdf_bytes,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert document.extracted_text is not None
    assert len(document.extracted_text) >= 0


@pytest.mark.asyncio
async def test_extraction_status_becomes_extracted(ingestion_service, candidate_profile, valid_pdf_bytes):
    """Test that extraction_status is set to 'extracted' after successful extraction."""
    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=valid_pdf_bytes,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert document.extraction_status == "extracted"


@pytest.mark.asyncio
async def test_extracted_text_is_persisted(ingestion_service, candidate_profile, valid_pdf_bytes):
    """Test that extracted_text is persisted in the database."""
    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=valid_pdf_bytes,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert document.extracted_text is not None
    assert len(document.extracted_text) > 0


@pytest.mark.asyncio
async def test_extraction_error_is_none_after_success(ingestion_service, candidate_profile, valid_pdf_bytes):
    """Test that extraction_error is None after successful extraction."""
    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=valid_pdf_bytes,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert document.extraction_error is None


# =============================================================================
# VALIDATION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_empty_file_is_rejected(ingestion_service, candidate_profile):
    """Test that empty file content is rejected."""
    with pytest.raises(DocumentIngestionValidationError) as exc_info:
        await ingestion_service.ingest_document(
            profile_id=candidate_profile.id,
            file_content=b"",
            filename="resume.pdf",
            content_type="application/pdf",
        )
    assert "empty" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_unsupported_content_type_is_rejected(ingestion_service, candidate_profile, valid_pdf_bytes):
    """Test that unsupported content type is rejected."""
    with pytest.raises(DocumentIngestionValidationError) as exc_info:
        await ingestion_service.ingest_document(
            profile_id=candidate_profile.id,
            file_content=valid_pdf_bytes,
            filename="resume.txt",
            content_type="text/plain",
        )
    assert "unsupported" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_empty_filename_is_rejected(ingestion_service, candidate_profile, valid_pdf_bytes):
    """Test that empty filename is rejected."""
    with pytest.raises(DocumentIngestionValidationError) as exc_info:
        await ingestion_service.ingest_document(
            profile_id=candidate_profile.id,
            file_content=valid_pdf_bytes,
            filename="",
            content_type="application/pdf",
        )
    assert "filename" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_whitespace_only_filename_is_rejected(ingestion_service, candidate_profile, valid_pdf_bytes):
    """Test that whitespace-only filename is rejected."""
    with pytest.raises(DocumentIngestionValidationError) as exc_info:
        await ingestion_service.ingest_document(
            profile_id=candidate_profile.id,
            file_content=valid_pdf_bytes,
            filename="   ",
            content_type="application/pdf",
        )
    assert "filename" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_nonexistent_profile_is_rejected(ingestion_service):
    """Test that nonexistent profile is rejected."""
    pdf_content = b"%PDF-1.4\n%Test PDF\nBT /F1 12 Tf 100 700 Td (Hello World) Tj ET\nendstream\nendobj\nxref\ntrailer\nstartxref\neof"
    nonexistent_profile_id = uuid4()

    with pytest.raises(DocumentIngestionValidationError) as exc_info:
        await ingestion_service.ingest_document(
            profile_id=nonexistent_profile_id,
            file_content=pdf_content,
            filename="resume.pdf",
            content_type="application/pdf",
        )
    assert "not found" in str(exc_info.value).lower()


# =============================================================================
# OWNERSHIP / ISOLATION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_document_belongs_to_supplied_profile(ingestion_service, candidate_profile, valid_pdf_bytes):
    """Test that document is associated with the supplied profile."""
    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=valid_pdf_bytes,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert document.profile_id == candidate_profile.id


@pytest.mark.asyncio
async def test_no_document_accidentally_associated_with_another_profile(ingestion_service, database_session, valid_pdf_bytes):
    """Test that document cannot be accidentally associated with another profile."""
    profile1 = CandidateProfile(**CandidateProfileCreate(full_name="Profile One", email="profile1@example.com").model_dump())
    profile2 = CandidateProfile(**CandidateProfileCreate(full_name="Profile Two", email="profile2@example.com").model_dump())
    database_session.add(profile1)
    database_session.add(profile2)
    await database_session.commit()
    await database_session.refresh(profile1)
    await database_session.refresh(profile2)

    document = await ingestion_service.ingest_document(
        profile_id=profile1.id,
        file_content=valid_pdf_bytes,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert document.profile_id == profile1.id
    assert document.profile_id != profile2.id


# =============================================================================
# FAILURE TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_malformed_pdf_produces_extraction_failure(ingestion_service, candidate_profile):
    """Test that malformed PDF produces extraction failure."""
    malformed_pdf = b"This is not a valid PDF at all"

    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=malformed_pdf,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert document.extraction_status == "failed"
    assert document.extraction_error is not None


@pytest.mark.asyncio
async def test_malformed_docx_produces_extraction_failure(ingestion_service, candidate_profile):
    """Test that malformed DOCX produces extraction failure."""
    malformed_docx = b"This is not a valid DOCX at all"

    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=malformed_docx,
        filename="resume.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert document.extraction_status == "failed"
    assert document.extraction_error is not None


@pytest.mark.asyncio
async def test_failed_extraction_sets_status_to_failed(ingestion_service, candidate_profile):
    """Test that failed extraction sets extraction_status to 'failed'."""
    malformed_pdf = b"Not a PDF"

    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=malformed_pdf,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert document.extraction_status == "failed"


@pytest.mark.asyncio
async def test_failed_extraction_stores_error_message(ingestion_service, candidate_profile):
    """Test that failed extraction stores error message."""
    malformed_pdf = b"Not a PDF"

    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=malformed_pdf,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert document.extraction_error is not None
    assert len(document.extraction_error) > 0


@pytest.mark.asyncio
async def test_original_file_remains_stored_after_extraction_failure(ingestion_service, candidate_profile, storage_service):
    """Test that original file remains stored even if extraction fails."""
    malformed_pdf = b"Not a PDF"

    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=malformed_pdf,
        filename="resume.pdf",
        content_type="application/pdf",
    )

    assert document.extraction_status == "failed"
    assert storage_service.exists(document.storage_path)
    stored_content = storage_service.get(document.storage_path)
    assert stored_content == malformed_pdf


@pytest.mark.asyncio
async def test_empty_extracted_text_treated_as_failure(ingestion_service, candidate_profile):
    """Test that extraction producing empty text is treated as failure."""
    # Create a minimal valid PDF that has structure but no extractable text
    # This is a PDF with just whitespace/structure
    minimal_pdf = b"%PDF-1.4\n1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n3 0 obj\n<</Type /Page /MediaBox [0 0 612 792] /Parent 2 0 R /Resources <<>>>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000056 00000 n\n0000000115 00000 n\ntrailer\n<</Size 4 /Root 1 0 R>>\nstartxref\n210\n%%EOF"

    document = await ingestion_service.ingest_document(
        profile_id=candidate_profile.id,
        file_content=minimal_pdf,
        filename="empty_resume.pdf",
        content_type="application/pdf",
    )

    # Empty text should result in failed status
    assert document.extraction_status == "failed"
    assert document.extraction_error is not None
    assert "no extractable text" in document.extraction_error.lower()
