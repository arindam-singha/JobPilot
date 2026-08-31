from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.candidate_document import CandidateDocument
from app.models.candidate_profile import CandidateProfile
from app.schemas.candidate_document import CandidateDocumentCreate, CandidateDocumentUpdate


class CandidateDocumentError(Exception):
    """Base exception for candidate document service errors."""


class CandidateDocumentNotFoundError(CandidateDocumentError):
    """Raised when a candidate document cannot be found for the given profile."""


class CandidateDocumentService:
    """Service layer for candidate document metadata operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_document(
        self,
        profile_id: UUID,
        data: CandidateDocumentCreate,
    ) -> CandidateDocument:
        profile = await self.session.get(CandidateProfile, profile_id)
        if profile is None:
            from app.services.candidate_profile_service import CandidateProfileNotFoundError

            raise CandidateProfileNotFoundError(f"Candidate profile {profile_id} not found")

        document = CandidateDocument(profile_id=profile_id, **data.model_dump())
        self.session.add(document)
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def get_document(
        self,
        profile_id: UUID,
        document_id: UUID,
    ) -> CandidateDocument:
        result = await self.session.execute(
            select(CandidateDocument).where(
                CandidateDocument.id == document_id,
                CandidateDocument.profile_id == profile_id,
            )
        )
        document = result.scalar_one_or_none()

        if document is None:
            raise CandidateDocumentNotFoundError(
                f"Candidate document {document_id} not found for candidate profile {profile_id}"
            )

        return document

    async def update_document(
        self,
        profile_id: UUID,
        document_id: UUID,
        data: CandidateDocumentUpdate,
    ) -> CandidateDocument:
        document = await self.get_document(profile_id, document_id)
        payload = data.model_dump(exclude_unset=True)

        for field_name, value in payload.items():
            setattr(document, field_name, value)

        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def delete_document(
        self,
        profile_id: UUID,
        document_id: UUID,
    ) -> None:
        document = await self.get_document(profile_id, document_id)
        await self.session.delete(document)
        await self.session.commit()
