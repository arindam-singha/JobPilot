from __future__ import annotations

from collections.abc import Collection
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.embedding_provider import (
    EmbeddingProvider,
    EmbeddingProviderError,
)
from app.models.candidate_evidence import CandidateEvidence
from app.models.candidate_profile import CandidateProfile
from app.schemas.semantic_evidence import (
    SemanticEvidenceMatch,
    SemanticEvidenceSearchResult,
)


class SemanticEvidenceRetrievalError(Exception):
    """Base exception for semantic evidence retrieval."""


class SemanticEvidenceProfileNotFoundError(
    SemanticEvidenceRetrievalError
):
    """Raised when the supplied candidate profile does not exist."""


class SemanticEvidenceNotFoundError(
    SemanticEvidenceRetrievalError
):
    """Raised when a profile has no compatible embedded evidence."""


class SemanticEvidenceProviderFailureError(
    SemanticEvidenceRetrievalError
):
    """Raised when query embedding generation fails."""


class SemanticEvidenceRetrievalService:
    """Retrieve candidate evidence using pgvector cosine similarity."""

    DEFAULT_TOP_K = 10
    MAX_TOP_K = 100

    def __init__(
        self,
        session: AsyncSession,
        provider: EmbeddingProvider,
    ) -> None:
        self.session = session
        self.provider = provider

    async def search(
        self,
        *,
        profile_id: UUID,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        minimum_similarity: float = 0.0,
        evidence_types: Collection[str] | None = None,
    ) -> SemanticEvidenceSearchResult:
        cleaned_query = self._validate_search_arguments(
            query=query,
            top_k=top_k,
            minimum_similarity=minimum_similarity,
        )

        await self._ensure_profile_exists(profile_id)

        query_vector = await self._embed_query(cleaned_query)

        distance_expression = (
            CandidateEvidence.embedding.cosine_distance(
                query_vector
            ).label("cosine_distance")
        )

        statement = (
            select(
                CandidateEvidence,
                distance_expression,
            )
            .where(
                CandidateEvidence.profile_id == profile_id,
                CandidateEvidence.embedding.is_not(None),
                CandidateEvidence.embedding_provider
                == self.provider.provider_name,
                CandidateEvidence.embedding_model
                == self.provider.model_name,
            )
            .order_by(
                distance_expression.asc(),
                CandidateEvidence.id.asc(),
            )
            .limit(top_k)
        )

        normalized_types = self._normalize_evidence_types(
            evidence_types
        )

        if normalized_types:
            statement = statement.where(
                CandidateEvidence.evidence_type.in_(
                    normalized_types
                )
            )

        result = await self.session.execute(statement)
        rows = result.all()

        if not rows:
            raise SemanticEvidenceNotFoundError(
                f"Candidate profile {profile_id} has no compatible "
                "embedded evidence"
            )

        matches: list[SemanticEvidenceMatch] = []

        for evidence, cosine_distance in rows:
            similarity = self._distance_to_similarity(
                cosine_distance
            )

            if similarity < minimum_similarity:
                continue

            matches.append(
                SemanticEvidenceMatch(
                    evidence_id=evidence.id,
                    profile_id=evidence.profile_id,
                    evidence_type=evidence.evidence_type,
                    title=evidence.title,
                    content=evidence.content,
                    source_type=evidence.source_type,
                    source_id=evidence.source_id,
                    similarity=round(similarity, 6),
                    embedding_provider=(
                        evidence.embedding_provider
                        or self.provider.provider_name
                    ),
                    embedding_model=(
                        evidence.embedding_model
                        or self.provider.model_name
                    ),
                )
            )

        return SemanticEvidenceSearchResult(
            profile_id=profile_id,
            query=cleaned_query,
            provider=self.provider.provider_name,
            model=self.provider.model_name,
            matches=matches,
        )

    async def _embed_query(
        self,
        query: str,
    ) -> list[float]:
        try:
            vector = await self.provider.embed_text(query)
        except EmbeddingProviderError as exc:
            raise SemanticEvidenceProviderFailureError(
                f"Embedding provider "
                f"{self.provider.provider_name!r} failed "
                "while embedding the search query"
            ) from exc
        except Exception as exc:
            raise SemanticEvidenceProviderFailureError(
                f"Embedding provider "
                f"{self.provider.provider_name!r} failed "
                "while embedding the search query"
            ) from exc

        if len(vector) != self.provider.dimensions:
            raise SemanticEvidenceProviderFailureError(
                "Query embedding dimensions do not match provider "
                f"configuration: expected "
                f"{self.provider.dimensions}, got {len(vector)}"
            )

        return vector

    async def _ensure_profile_exists(
        self,
        profile_id: UUID,
    ) -> None:
        profile = await self.session.get(
            CandidateProfile,
            profile_id,
        )

        if profile is None:
            raise SemanticEvidenceProfileNotFoundError(
                f"Candidate profile {profile_id} not found"
            )

    @classmethod
    def _validate_search_arguments(
        cls,
        *,
        query: str,
        top_k: int,
        minimum_similarity: float,
    ) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "query must not be empty or whitespace"
            )

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        if top_k > cls.MAX_TOP_K:
            raise ValueError(
                f"top_k must not exceed {cls.MAX_TOP_K}"
            )

        if not isinstance(
            minimum_similarity,
            int | float,
        ):
            raise TypeError(
                "minimum_similarity must be numeric"
            )

        if not 0.0 <= float(minimum_similarity) <= 1.0:
            raise ValueError(
                "minimum_similarity must be between 0 and 1"
            )

        return cleaned_query

    @staticmethod
    def _normalize_evidence_types(
        evidence_types: Collection[str] | None,
    ) -> list[str]:
        if evidence_types is None:
            return []

        normalized: list[str] = []
        seen: set[str] = set()

        for evidence_type in evidence_types:
            if not isinstance(evidence_type, str):
                raise TypeError(
                    "each evidence type must be a string"
                )

            cleaned = evidence_type.strip()

            if not cleaned:
                continue

            key = cleaned.casefold()

            if key in seen:
                continue

            seen.add(key)
            normalized.append(cleaned)

        return normalized

    @staticmethod
    def _distance_to_similarity(
        distance: float | None,
    ) -> float:
        if distance is None:
            return 0.0

        similarity = 1.0 - float(distance)

        # Cosine similarity may technically be negative. The public
        # retrieval contract uses the simpler 0–1 range.
        return max(0.0, min(1.0, similarity))