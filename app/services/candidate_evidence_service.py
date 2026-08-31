from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.candidate_evidence import CandidateEvidence
from app.models.candidate_profile import CandidateProfile
from app.schemas.candidate_evidence import (
    CandidateEvidenceCreate,
    CandidateEvidenceUpdate,
)


class CandidateEvidenceError(Exception):
    """Base exception for candidate evidence service errors."""


class CandidateEvidenceNotFoundError(CandidateEvidenceError):
    """Raised when evidence cannot be found for the given profile."""


class CandidateEvidenceService:
    """Service layer for candidate evidence operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_evidence(
        self,
        profile_id: UUID,
        data: CandidateEvidenceCreate,
    ) -> CandidateEvidence:
        profile = await self.session.get(
            CandidateProfile,
            profile_id,
        )

        if profile is None:
            from app.services.candidate_profile_service import (
                CandidateProfileNotFoundError,
            )

            raise CandidateProfileNotFoundError(
                f"Candidate profile {profile_id} not found"
            )

        evidence = CandidateEvidence(
            profile_id=profile_id,
            **data.model_dump(),
        )

        self.session.add(evidence)
        await self.session.commit()
        await self.session.refresh(evidence)

        return evidence

    async def list_evidence(
        self,
        profile_id: UUID,
    ) -> list[CandidateEvidence]:
        profile = await self.session.get(
            CandidateProfile,
            profile_id,
        )

        if profile is None:
            from app.services.candidate_profile_service import (
                CandidateProfileNotFoundError,
            )

            raise CandidateProfileNotFoundError(
                f"Candidate profile {profile_id} not found"
            )

        result = await self.session.execute(
            select(CandidateEvidence)
            .where(CandidateEvidence.profile_id == profile_id)
            .order_by(CandidateEvidence.id)
        )

        return list(result.scalars().all())

    async def get_evidence(
        self,
        profile_id: UUID,
        evidence_id: UUID,
    ) -> CandidateEvidence:
        result = await self.session.execute(
            select(CandidateEvidence).where(
                CandidateEvidence.id == evidence_id,
                CandidateEvidence.profile_id == profile_id,
            )
        )

        evidence = result.scalar_one_or_none()

        if evidence is None:
            raise CandidateEvidenceNotFoundError(
                f"Candidate evidence {evidence_id} not found "
                f"for candidate profile {profile_id}"
            )

        return evidence

    async def update_evidence(
        self,
        profile_id: UUID,
        evidence_id: UUID,
        data: CandidateEvidenceUpdate,
    ) -> CandidateEvidence:
        evidence = await self.get_evidence(
            profile_id,
            evidence_id,
        )

        payload = data.model_dump(exclude_unset=True)

        for field_name, value in payload.items():
            setattr(evidence, field_name, value)

        await self.session.commit()
        await self.session.refresh(evidence)

        return evidence

    async def delete_evidence(
        self,
        profile_id: UUID,
        evidence_id: UUID,
    ) -> None:
        evidence = await self.get_evidence(
            profile_id,
            evidence_id,
        )

        await self.session.delete(evidence)
        await self.session.commit()