from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.cv.extraction import ExtractedText, extract_document_text
from app.models.candidate_document import CandidateDocument
from app.models.candidate_profile import CandidateProfile
from app.schemas.candidate_document import CandidateDocumentCreate, CandidateDocumentUpdate
from app.services.candidate_document_service import (
    CandidateDocumentError,
    CandidateDocumentService,
)
from app.services.document_storage_service import (
    DocumentStorageError,
    DocumentStorageService,
    DocumentStorageTypeError,
)


class DocumentIngestionError(Exception):
    """Base exception for document ingestion errors."""


class DocumentIngestionValidationError(DocumentIngestionError):
    """Raised when document ingestion validation fails."""


class DocumentIngestionExtractionError(DocumentIngestionError):
    """Raised when document text extraction fails."""


class DocumentIngestionStorageError(DocumentIngestionError):
    """Raised when document storage operations fail."""


class DocumentIngestionService:
    """Orchestration layer for document ingestion.
    
    Coordinates:
    - CandidateDocumentService (database operations)
    - DocumentStorageService (file storage)
    - Document text extraction (via extract_document_text)
    
    Does NOT contain low-level filesystem or parsing logic.
    """

    def __init__(
        self,
        session: AsyncSession,
        document_service: CandidateDocumentService,
        storage_service: DocumentStorageService,
    ) -> None:
        self.session = session
        self.document_service = document_service
        self.storage_service = storage_service

    async def ingest_document(
        self,
        profile_id: UUID,
        file_content: bytes,
        filename: str,
        content_type: str,
    ) -> CandidateDocument:
        """Ingest a document for a candidate profile.
        
        Flow:
        1. Validate the candidate profile exists.
        2. Validate that the document type is supported.
        3. Generate a new document UUID.
        4. Store the original document using DocumentStorageService.
        5. Create the CandidateDocument database record.
        6. Set extraction_status = "extracting".
        7. Retrieve/use the stored file bytes.
        8. Select the appropriate extractor using extract_document_text.
        9. Extract text.
        10. Update CandidateDocument with extracted_text.
        11. Set extraction_status = "extracted".
        12. Set extraction_error = None.
        13. Commit the transaction.
        14. Return CandidateDocument.
        
        Failure handling:
        - If storage succeeds but database creation fails: attempt to delete the stored file.
        - If extraction fails: persist the document with extraction_status = "failed" and error message.
        - If database transaction fails after storage: attempt to delete the stored file.
        """

        # Step 1: Validate the candidate profile exists
        profile = await self.session.get(CandidateProfile, profile_id)
        if profile is None:
            raise DocumentIngestionValidationError(f"Candidate profile {profile_id} not found")

        # Step 2: Validate that the document type is supported
        try:
            self.storage_service._validate_content_type(content_type)
        except DocumentStorageTypeError as exc:
            raise DocumentIngestionValidationError(f"Unsupported document content type: {content_type}") from exc

        # Validate file content is not empty
        if not file_content or len(file_content) == 0:
            raise DocumentIngestionValidationError("File content cannot be empty")

        # Validate filename is not empty
        if not filename or not filename.strip():
            raise DocumentIngestionValidationError("Filename cannot be empty")

        # Step 3: Generate a new document UUID
        document_id = uuid4()

        # Step 4: Store the original document using DocumentStorageService
        try:
            stored_document = self.storage_service.save(
                file_content=file_content,
                document_id=document_id,
                original_filename=filename,
                content_type=content_type,
            )
        except DocumentStorageError as exc:
            raise DocumentIngestionStorageError(f"Failed to store document: {exc}") from exc

        # Step 5-6: Create the CandidateDocument database record with extraction_status = "extracting"
        try:
            create_data = CandidateDocumentCreate(
                filename=filename,
                content_type=content_type,
                storage_path=stored_document.storage_path,
                file_size=stored_document.file_size,
                extracted_text=None,
                extraction_status="extracting",
                extraction_error=None,
            )
            document = CandidateDocument(
                id=document_id,
                profile_id=profile_id,
                **create_data.model_dump(),
            )
            self.session.add(document)
            await self.session.flush()  # Flush to get any validation errors
        except Exception as exc:
            # If database creation fails, attempt cleanup of stored file
            try:
                self.storage_service.delete(stored_document.storage_path)
            except DocumentStorageError:
                pass  # Log but don't raise - we want to report the original database error
            raise DocumentIngestionStorageError(
                f"Failed to create document record: {exc}"
            ) from exc

        # Step 7-9: Retrieve the stored file and extract text
        try:
            extracted = extract_document_text(file_content, content_type)
        except Exception as exc:
            # Extraction failed, but keep the document and file
            error_message = f"Text extraction failed: {exc}"
            document.extraction_status = "failed"
            document.extraction_error = error_message
            await self.session.commit()
            await self.session.refresh(document)
            return document

        # Step 10-12: Update with extracted text and mark as extracted
        # Handle case of empty extracted text
        if not extracted.text or not extracted.text.strip():
            document.extraction_status = "failed"
            document.extraction_error = "No extractable text found in document"
            await self.session.commit()
            await self.session.refresh(document)
            return document

        document.extracted_text = extracted.text
        document.extraction_status = "extracted"
        document.extraction_error = None

        # Step 13: Commit the transaction
        await self.session.commit()

        # Step 14: Return CandidateDocument
        await self.session.refresh(document)
        return document
