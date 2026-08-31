from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.embedding_provider import (
    EmbeddingProvider,
    EmbeddingProviderError,
)
from app.matching.text_matching import combine_evidence_text
from app.models.candidate_evidence import CandidateEvidence
from app.models.candidate_profile import CandidateProfile


class CandidateEvidenceEmbeddingError(Exception):
    """Base exception for candidate-evidence embedding failures."""


class CandidateEvidenceEmbeddingProfileNotFoundError(
    CandidateEvidenceEmbeddingError
):
    """Raised when the supplied candidate profile does not exist."""


class CandidateEvidenceEmbeddingNotFoundError(
    CandidateEvidenceEmbeddingError
):
    """Raised when the candidate has no evidence to embed."""


class CandidateEvidenceEmbeddingProviderFailureError(
    CandidateEvidenceEmbeddingError
):
    """Raised when the embedding provider fails."""


class CandidateEvidenceEmbeddingService:
    """Generate and persist embeddings for candidate evidence."""

    def __init__(
        self,
        session: AsyncSession,
        provider: EmbeddingProvider,
    ) -> None:
        self.session = session
        self.provider = provider

    async def embed_profile_evidence(
        self,
        profile_id: UUID,
        *,
        force: bool = False,
    ) -> list[CandidateEvidence]:
        await self._ensure_profile_exists(profile_id)

        evidence = await self._get_profile_evidence(profile_id)

        if not evidence:
            raise CandidateEvidenceEmbeddingNotFoundError(
                f"Candidate profile {profile_id} has no evidence"
            )

        pending = [
            item
            for item in evidence
            if force or self._needs_embedding(item)
        ]

        if not pending:
            return evidence

        texts = [
            self._build_embedding_text(item)
            for item in pending
        ]

        try:
            vectors = await self.provider.embed_batch(texts)
        except EmbeddingProviderError as exc:
            raise CandidateEvidenceEmbeddingProviderFailureError(
                f"Embedding provider {self.provider.provider_name!r} "
                f"failed for candidate profile {profile_id}"
            ) from exc
        except Exception as exc:
            raise CandidateEvidenceEmbeddingProviderFailureError(
                f"Embedding provider {self.provider.provider_name!r} "
                f"failed for candidate profile {profile_id}"
            ) from exc

        if len(vectors) != len(pending):
            raise CandidateEvidenceEmbeddingProviderFailureError(
                "Embedding provider returned an unexpected number of vectors"
            )

        embedded_at = datetime.now(UTC)

        try:
            for item, vector in zip(
                pending,
                vectors,
                strict=True,
            ):
                self._validate_vector(vector)

                item.embedding = vector
                item.embedding_provider = self.provider.provider_name
                item.embedding_model = self.provider.model_name
                item.embedded_at = embedded_at

            await self.session.commit()

            for item in pending:
                await self.session.refresh(item)

        except Exception:
            await self.session.rollback()
            raise

        return evidence

    async def embed_evidence(
        self,
        profile_id: UUID,
        evidence_id: UUID,
        *,
        force: bool = False,
    ) -> CandidateEvidence:
        await self._ensure_profile_exists(profile_id)

        result = await self.session.execute(
            select(CandidateEvidence).where(
                CandidateEvidence.id == evidence_id,
                CandidateEvidence.profile_id == profile_id,
            )
        )

        evidence = result.scalar_one_or_none()

        if evidence is None:
            raise CandidateEvidenceEmbeddingNotFoundError(
                f"Candidate evidence {evidence_id} not found "
                f"for candidate profile {profile_id}"
            )

        if not force and not self._needs_embedding(evidence):
            return evidence

        text = self._build_embedding_text(evidence)

        try:
            vector = await self.provider.embed_text(text)
        except EmbeddingProviderError as exc:
            raise CandidateEvidenceEmbeddingProviderFailureError(
                f"Embedding provider {self.provider.provider_name!r} "
                f"failed for candidate evidence {evidence_id}"
            ) from exc
        except Exception as exc:
            raise CandidateEvidenceEmbeddingProviderFailureError(
                f"Embedding provider {self.provider.provider_name!r} "
                f"failed for candidate evidence {evidence_id}"
            ) from exc

        self._validate_vector(vector)

        try:
            evidence.embedding = vector
            evidence.embedding_provider = self.provider.provider_name
            evidence.embedding_model = self.provider.model_name
            evidence.embedded_at = datetime.now(UTC)

            await self.session.commit()
            await self.session.refresh(evidence)

        except Exception:
            await self.session.rollback()
            raise

        return evidence

    def _needs_embedding(
        self,
        evidence: CandidateEvidence,
    ) -> bool:
        return (
            evidence.embedding is None
            or evidence.embedding_provider != self.provider.provider_name
            or evidence.embedding_model != self.provider.model_name
        )

    def _build_embedding_text(
        self,
        evidence: CandidateEvidence,
    ) -> str:
        text = combine_evidence_text(
            [
                evidence.evidence_type,
                evidence.title,
                evidence.content,
            ]
        )

        if not text:
            raise CandidateEvidenceEmbeddingError(
                f"Candidate evidence {evidence.id} has no embeddable text"
            )

        return text

    def _validate_vector(
        self,
        vector: list[float],
    ) -> None:
        if len(vector) != self.provider.dimensions:
            raise CandidateEvidenceEmbeddingProviderFailureError(
                "Embedding dimensions do not match provider configuration: "
                f"expected {self.provider.dimensions}, got {len(vector)}"
            )

    async def _ensure_profile_exists(
        self,
        profile_id: UUID,
    ) -> None:
        profile = await self.session.get(
            CandidateProfile,
            profile_id,
        )

        if profile is None:
            raise CandidateEvidenceEmbeddingProfileNotFoundError(
                f"Candidate profile {profile_id} not found"
            )

    async def _get_profile_evidence(
        self,
        profile_id: UUID,
    ) -> list[CandidateEvidence]:
        result = await self.session.execute(
            select(CandidateEvidence)
            .where(
                CandidateEvidence.profile_id == profile_id
            )
            .order_by(
                CandidateEvidence.created_at,
                CandidateEvidence.id,
            )
        )

        return list(result.scalars().all())