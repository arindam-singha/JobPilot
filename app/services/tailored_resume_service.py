from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tailored_resume import TailoredResume
from app.schemas.tailored_resume import TailoredResumeDraft
from app.schemas.tailored_resume_record import (
    TailoredResumeRecordList,
    TailoredResumeRecordRead,
    TailoredResumeRecordSummary,
)
from app.services.tailored_resume_generation_service import (
    TailoredResumeGenerationService,
)


class TailoredResumeServiceError(Exception):
    """Base exception for tailored-resume persistence failures."""


class TailoredResumeNotFoundError(TailoredResumeServiceError):
    """Raised when a scoped tailored resume is not found."""


class TailoredResumeService:
    """Generate, persist, and retrieve tailored resumes."""

    DEFAULT_LIMIT = 20
    MAX_LIMIT = 100

    def __init__(
        self,
        session: AsyncSession,
        generation_service: (
            TailoredResumeGenerationService | None
        ) = None,
    ) -> None:
        self.session = session
        self.generation_service = generation_service


    # def __init__(
    #     self,
    #     session: AsyncSession,
    #     generation_service: TailoredResumeGenerationService,
    # ) -> None:
    #     self.session = session
    #     self.generation_service = generation_service

    async def generate_and_save(
        self,
        *,
        job_id: UUID,
        profile_id: UUID,
    ) -> TailoredResumeRecordRead:
        if self.generation_service is None:
            raise RuntimeError(
                "A generation service is required "
                "to generate a tailored resume"
            )
        draft = await self.generation_service.generate(
            job_id=job_id,
            profile_id=profile_id,
        )
        record = TailoredResume(
            job_id=job_id,
            profile_id=profile_id,
            status="generated",
            structured_content=draft.model_dump(mode="json"),
            generator_provider=draft.generator_provider,
            generator_model=draft.generator_model,
            generation_metadata=(self._build_generation_metadata(draft)),
        )

        self.session.add(record)

        try:
            await self.session.commit()
            await self.session.refresh(record)
        except Exception:
            await self.session.rollback()
            raise

        return self._to_read(record)

    async def get_resume(
        self,
        *,
        resume_id: UUID,
        job_id: UUID,
        profile_id: UUID,
    ) -> TailoredResumeRecordRead:
        statement = select(TailoredResume).where(
            TailoredResume.id == resume_id,
            TailoredResume.job_id == job_id,
            TailoredResume.profile_id == profile_id,
        )

        result = await self.session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            raise TailoredResumeNotFoundError("Tailored resume not found")

        return self._to_read(record)

    async def list_resumes(
        self,
        *,
        job_id: UUID,
        profile_id: UUID,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> TailoredResumeRecordList:
        self._validate_pagination(
            limit=limit,
            offset=offset,
        )

        filters = (
            TailoredResume.job_id == job_id,
            TailoredResume.profile_id == profile_id,
        )

        count_result = await self.session.execute(
            select(func.count(TailoredResume.id)).where(*filters)
        )

        total = int(count_result.scalar_one())

        result = await self.session.execute(
            select(TailoredResume)
            .where(*filters)
            .order_by(
                TailoredResume.created_at.desc(),
                TailoredResume.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )

        records = result.scalars().all()

        return TailoredResumeRecordList(
            items=[self._to_summary(record) for record in records],
            total=total,
            limit=limit,
            offset=offset,
        )

    @staticmethod
    def _to_read(
        record: TailoredResume,
    ) -> TailoredResumeRecordRead:
        return TailoredResumeRecordRead(
            id=record.id,
            job_id=record.job_id,
            profile_id=record.profile_id,
            status=record.status,
            structured_content=(TailoredResumeDraft.model_validate(record.structured_content)),
            generator_provider=(record.generator_provider),
            generator_model=record.generator_model,
            generation_metadata=(record.generation_metadata),
            created_at=record.created_at,
        )

    @staticmethod
    def _to_summary(
        record: TailoredResume,
    ) -> TailoredResumeRecordSummary:
        content = TailoredResumeDraft.model_validate(record.structured_content)

        metadata = record.generation_metadata or {}

        raw_evidence_count = metadata.get(
            "evidence_count",
            len(content.evidence_catalog),
        )

        evidence_count = (
            raw_evidence_count
            if isinstance(raw_evidence_count, int)
            else len(content.evidence_catalog)
        )

        return TailoredResumeRecordSummary(
            id=record.id,
            job_id=record.job_id,
            profile_id=record.profile_id,
            status=record.status,
            target_title=content.target_title,
            generator_provider=(record.generator_provider),
            generator_model=record.generator_model,
            evidence_count=evidence_count,
            created_at=record.created_at,
        )

    @staticmethod
    def _build_generation_metadata(
        draft: TailoredResumeDraft,
    ) -> dict[str, Any]:
        statement_count = (
            len(draft.professional_summary)
            + len(draft.skills)
            + sum(len(section.statements) for section in draft.sections)
        )

        return {
            "schema_version": 1,
            "provenance_version": 1,
            "evidence_count": len(draft.evidence_catalog),
            "statement_count": statement_count,
        }

    @classmethod
    def _validate_pagination(
        cls,
        *,
        limit: int,
        offset: int,
    ) -> None:
        if not 1 <= limit <= cls.MAX_LIMIT:
            raise ValueError("limit must be between 1 and 100")

        if offset < 0:
            raise ValueError("offset must be greater than or equal to zero")
