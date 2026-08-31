from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.cv.models import TextChunk
from app.cv.text_processing import (
    DEFAULT_CHUNK_OVERLAP_CHARACTERS,
    DEFAULT_MAX_CHUNK_CHARACTERS,
    build_chunks,
)
from app.services.candidate_document_service import (
    CandidateDocumentNotFoundError,
    CandidateDocumentService,
)


class EvidenceExtractionError(Exception):
    """Base exception for candidate evidence extraction errors."""


class EvidenceExtractionDocumentNotFoundError(EvidenceExtractionError):
    """Raised when a document is unavailable for the supplied profile."""


class EvidenceExtractionDocumentNotReadyError(EvidenceExtractionError):
    """Raised when document text extraction has not completed successfully."""


class EvidenceExtractionEmptyTextError(EvidenceExtractionError):
    """Raised when a document has no usable extracted text."""


class EvidenceExtractionService:
    """Build deterministic evidence text chunks from candidate documents.

    This service is read-only. It does not persist text chunks or candidate
    evidence records.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        max_chunk_characters: int = DEFAULT_MAX_CHUNK_CHARACTERS,
        chunk_overlap_characters: int = DEFAULT_CHUNK_OVERLAP_CHARACTERS,
    ) -> None:
        self.session = session
        self.document_service = CandidateDocumentService(session)
        self.max_chunk_characters = max_chunk_characters
        self.chunk_overlap_characters = chunk_overlap_characters

        self._validate_chunk_configuration()

    async def build_document_chunks(
        self,
        profile_id: UUID,
        document_id: UUID,
    ) -> list[TextChunk]:
        """Load a candidate document and convert its extracted text to chunks."""

        try:
            document = await self.document_service.get_document(
                profile_id,
                document_id,
            )
        except CandidateDocumentNotFoundError as exc:
            raise EvidenceExtractionDocumentNotFoundError(
                f"Candidate document {document_id} not found "
                f"for candidate profile {profile_id}"
            ) from exc

        if document.extraction_status != "extracted":
            raise EvidenceExtractionDocumentNotReadyError(
                f"Candidate document {document_id} is not ready for evidence "
                f"extraction; current status is "
                f"{document.extraction_status!r}"
            )

        extracted_text = document.extracted_text

        if extracted_text is None or not extracted_text.strip():
            raise EvidenceExtractionEmptyTextError(
                f"Candidate document {document_id} has no usable extracted text"
            )

        chunks = build_chunks(
            extracted_text,
            max_chunk_characters=self.max_chunk_characters,
            chunk_overlap_characters=self.chunk_overlap_characters,
        )

        if not chunks:
            raise EvidenceExtractionEmptyTextError(
                f"Candidate document {document_id} produced no text chunks"
            )

        return chunks

    def _validate_chunk_configuration(self) -> None:
        if self.max_chunk_characters <= 0:
            raise ValueError(
                "max_chunk_characters must be greater than zero"
            )

        if self.chunk_overlap_characters < 0:
            raise ValueError(
                "chunk_overlap_characters must be greater than or equal to zero"
            )

        if (
            self.chunk_overlap_characters
            >= self.max_chunk_characters
        ):
            raise ValueError(
                "chunk_overlap_characters must be smaller than "
                "max_chunk_characters"
            )