from __future__ import annotations

from uuid import uuid4

import pytest

from app.models.candidate_evidence import CandidateEvidence
from app.models.candidate_profile import CandidateProfile
from app.services.semantic_evidence_retrieval_service import (
    SemanticEvidenceNotFoundError,
    SemanticEvidenceProfileNotFoundError,
    SemanticEvidenceProviderFailureError,
    SemanticEvidenceRetrievalService,
)


DIMENSIONS = 768


def _vector(
    first: float,
    second: float = 0.0,
) -> list[float]:
    vector = [0.0] * DIMENSIONS
    vector[0] = first
    vector[1] = second
    return vector


class FixedEmbeddingProvider:
    def __init__(
        self,
        vector: list[float],
        *,
        provider_name: str = "fake",
        model_name: str = "semantic-test-model",
    ) -> None:
        self._vector = vector
        self._provider_name = provider_name
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:
        return list(self._vector)

    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            list(self._vector)
            for _ in texts
        ]


async def _create_profile(
    database_session,
    *,
    name: str = "Semantic Search Candidate",
) -> CandidateProfile:
    profile = CandidateProfile(full_name=name)

    database_session.add(profile)
    await database_session.commit()
    await database_session.refresh(profile)

    return profile


async def _create_evidence(
    database_session,
    *,
    profile_id,
    title: str,
    content: str,
    embedding: list[float] | None,
    evidence_type: str = "skill",
    provider: str | None = "fake",
    model: str | None = "semantic-test-model",
) -> CandidateEvidence:
    evidence = CandidateEvidence(
        profile_id=profile_id,
        evidence_type=evidence_type,
        title=title,
        content=content,
        source_type="manual",
        source_id=None,
        metadata_json=None,
        embedding=embedding,
        embedding_provider=provider,
        embedding_model=model,
    )

    database_session.add(evidence)
    await database_session.commit()
    await database_session.refresh(evidence)

    return evidence


@pytest.mark.asyncio
async def test_search_returns_most_similar_evidence_first(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    exact = await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Python experience",
        content="Developed Python systems.",
        embedding=_vector(1.0),
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Partial match",
        content="Related engineering evidence.",
        embedding=_vector(0.8, 0.6),
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Orthogonal evidence",
        content="Unrelated evidence.",
        embedding=_vector(0.0, 1.0),
    )

    service = SemanticEvidenceRetrievalService(
        database_session,
        FixedEmbeddingProvider(_vector(1.0)),
    )

    result = await service.search(
        profile_id=profile.id,
        query="Python engineering",
    )

    assert len(result.matches) == 3
    assert result.matches[0].evidence_id == exact.id
    assert result.matches[0].similarity == pytest.approx(1.0)
    assert result.matches[1].similarity == pytest.approx(0.8)
    assert result.matches[2].similarity == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_search_respects_top_k(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    for index in range(4):
        await _create_evidence(
            database_session,
            profile_id=profile.id,
            title=f"Evidence {index}",
            content="Python",
            embedding=_vector(1.0),
        )

    result = await SemanticEvidenceRetrievalService(
        database_session,
        FixedEmbeddingProvider(_vector(1.0)),
    ).search(
        profile_id=profile.id,
        query="Python",
        top_k=2,
    )

    assert len(result.matches) == 2


@pytest.mark.asyncio
async def test_search_filters_by_minimum_similarity(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Strong",
        content="Strong evidence",
        embedding=_vector(1.0),
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Weak",
        content="Weak evidence",
        embedding=_vector(0.0, 1.0),
    )

    result = await SemanticEvidenceRetrievalService(
        database_session,
        FixedEmbeddingProvider(_vector(1.0)),
    ).search(
        profile_id=profile.id,
        query="Python",
        minimum_similarity=0.5,
    )

    assert len(result.matches) == 1
    assert result.matches[0].title == "Strong"


@pytest.mark.asyncio
async def test_search_filters_by_evidence_type(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    skill = await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Python skill",
        content="Python",
        embedding=_vector(1.0),
        evidence_type="skill",
    )

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Python project",
        content="Python project",
        embedding=_vector(1.0),
        evidence_type="project",
    )

    result = await SemanticEvidenceRetrievalService(
        database_session,
        FixedEmbeddingProvider(_vector(1.0)),
    ).search(
        profile_id=profile.id,
        query="Python",
        evidence_types=["skill"],
    )

    assert len(result.matches) == 1
    assert result.matches[0].evidence_id == skill.id


@pytest.mark.asyncio
async def test_search_is_isolated_to_profile(
    database_session,
) -> None:
    first_profile = await _create_profile(
        database_session,
        name="First",
    )
    second_profile = await _create_profile(
        database_session,
        name="Second",
    )

    first_evidence = await _create_evidence(
        database_session,
        profile_id=first_profile.id,
        title="First evidence",
        content="Python",
        embedding=_vector(1.0),
    )

    await _create_evidence(
        database_session,
        profile_id=second_profile.id,
        title="Second evidence",
        content="Python",
        embedding=_vector(1.0),
    )

    result = await SemanticEvidenceRetrievalService(
        database_session,
        FixedEmbeddingProvider(_vector(1.0)),
    ).search(
        profile_id=first_profile.id,
        query="Python",
    )

    assert len(result.matches) == 1
    assert result.matches[0].evidence_id == first_evidence.id


@pytest.mark.asyncio
async def test_search_ignores_unembedded_evidence(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Unembedded",
        content="Python",
        embedding=None,
        provider=None,
        model=None,
    )

    service = SemanticEvidenceRetrievalService(
        database_session,
        FixedEmbeddingProvider(_vector(1.0)),
    )

    with pytest.raises(SemanticEvidenceNotFoundError):
        await service.search(
            profile_id=profile.id,
            query="Python",
        )


@pytest.mark.asyncio
async def test_search_ignores_different_embedding_model(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Old model evidence",
        content="Python",
        embedding=_vector(1.0),
        model="old-model",
    )

    service = SemanticEvidenceRetrievalService(
        database_session,
        FixedEmbeddingProvider(_vector(1.0)),
    )

    with pytest.raises(SemanticEvidenceNotFoundError):
        await service.search(
            profile_id=profile.id,
            query="Python",
        )


@pytest.mark.asyncio
async def test_unknown_profile_is_rejected(
    database_session,
) -> None:
    service = SemanticEvidenceRetrievalService(
        database_session,
        FixedEmbeddingProvider(_vector(1.0)),
    )

    with pytest.raises(
        SemanticEvidenceProfileNotFoundError
    ):
        await service.search(
            profile_id=uuid4(),
            query="Python",
        )


@pytest.mark.asyncio
async def test_provider_failure_is_wrapped(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Python",
        content="Python",
        embedding=_vector(1.0),
    )

    class FailingProvider(FixedEmbeddingProvider):
        async def embed_text(
            self,
            text: str,
        ) -> list[float]:
            raise RuntimeError("provider unavailable")

    service = SemanticEvidenceRetrievalService(
        database_session,
        FailingProvider(_vector(1.0)),
    )

    with pytest.raises(
        SemanticEvidenceProviderFailureError,
        match="failed while embedding",
    ):
        await service.search(
            profile_id=profile.id,
            query="Python",
        )


@pytest.mark.asyncio
async def test_wrong_query_vector_dimensions_are_rejected(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    await _create_evidence(
        database_session,
        profile_id=profile.id,
        title="Python",
        content="Python",
        embedding=_vector(1.0),
    )

    class WrongDimensionsProvider(
        FixedEmbeddingProvider
    ):
        async def embed_text(
            self,
            text: str,
        ) -> list[float]:
            return [1.0] * 10

    service = SemanticEvidenceRetrievalService(
        database_session,
        WrongDimensionsProvider(_vector(1.0)),
    )

    with pytest.raises(
        SemanticEvidenceProviderFailureError,
        match="dimensions do not match",
    ):
        await service.search(
            profile_id=profile.id,
            query="Python",
        )


@pytest.mark.parametrize(
    ("top_k", "exception_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (101, ValueError),
        (True, TypeError),
    ],
)
def test_invalid_top_k_is_rejected(
    top_k,
    exception_type,
) -> None:
    with pytest.raises(exception_type):
        SemanticEvidenceRetrievalService._validate_search_arguments(
            query="Python",
            top_k=top_k,
            minimum_similarity=0.0,
        )


@pytest.mark.parametrize(
    "minimum_similarity",
    [
        -0.1,
        1.1,
    ],
)
def test_invalid_minimum_similarity_is_rejected(
    minimum_similarity: float,
) -> None:
    with pytest.raises(ValueError):
        SemanticEvidenceRetrievalService._validate_search_arguments(
            query="Python",
            top_k=10,
            minimum_similarity=minimum_similarity,
        )


def test_empty_query_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        SemanticEvidenceRetrievalService._validate_search_arguments(
            query="   ",
            top_k=10,
            minimum_similarity=0.0,
        )