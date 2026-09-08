from __future__ import annotations

from uuid import uuid4

import pytest
from app.embeddings.fake_embedding_provider import (
    FakeEmbeddingProvider,
)
from app.models.candidate_evidence import CandidateEvidence
from app.models.candidate_profile import CandidateProfile
from app.services.candidate_evidence_embedding_service import (
    CandidateEvidenceEmbeddingNotFoundError,
    CandidateEvidenceEmbeddingProfileNotFoundError,
    CandidateEvidenceEmbeddingProviderFailureError,
    CandidateEvidenceEmbeddingService,
)


async def _create_profile(
    database_session,
) -> CandidateProfile:
    profile = CandidateProfile(
        full_name="Embedding Test Candidate",
    )

    database_session.add(profile)
    await database_session.commit()
    await database_session.refresh(profile)

    return profile


async def _create_evidence(
    database_session,
    *,
    profile_id,
    evidence_type: str = "skill",
    title: str = "Technical skills",
    content: str = "Python and PyTorch",
) -> CandidateEvidence:
    evidence = CandidateEvidence(
        profile_id=profile_id,
        evidence_type=evidence_type,
        title=title,
        content=content,
        source_type="manual",
        source_id=None,
        metadata_json=None,
    )

    database_session.add(evidence)
    await database_session.commit()
    await database_session.refresh(evidence)

    return evidence


@pytest.mark.asyncio
async def test_embed_profile_evidence_persists_vectors(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    first = await _create_evidence(
        database_session,
        profile_id=profile.id,
        content="Python and PyTorch",
    )

    second = await _create_evidence(
        database_session,
        profile_id=profile.id,
        evidence_type="project",
        title="Defect detection",
        content="Built a computer vision system",
    )

    provider = FakeEmbeddingProvider(dimensions=768)

    service = CandidateEvidenceEmbeddingService(
        database_session,
        provider,
    )

    result = await service.embed_profile_evidence(
        profile.id,
    )

    assert len(result) == 2

    refreshed_first = await database_session.get(
        CandidateEvidence,
        first.id,
    )
    refreshed_second = await database_session.get(
        CandidateEvidence,
        second.id,
    )

    assert refreshed_first is not None
    assert refreshed_second is not None

    assert refreshed_first.embedding is not None
    assert refreshed_second.embedding is not None
    assert len(refreshed_first.embedding) == 768
    assert len(refreshed_second.embedding) == 768

    assert refreshed_first.embedding_provider == "fake"
    assert refreshed_first.embedding_model == "fake-embedding-model"
    assert refreshed_first.embedded_at is not None


@pytest.mark.asyncio
async def test_embedding_is_idempotent_without_force(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    evidence = await _create_evidence(
        database_session,
        profile_id=profile.id,
    )

    provider = FakeEmbeddingProvider(dimensions=768)

    service = CandidateEvidenceEmbeddingService(
        database_session,
        provider,
    )

    first = await service.embed_profile_evidence(
        profile.id,
    )

    first_vector = list(first[0].embedding)
    first_embedded_at = first[0].embedded_at

    second = await service.embed_profile_evidence(
        profile.id,
    )

    assert list(second[0].embedding) == first_vector
    assert second[0].embedded_at == first_embedded_at
    assert evidence.id == second[0].id


@pytest.mark.asyncio
async def test_force_reembedding_updates_timestamp(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    await _create_evidence(
        database_session,
        profile_id=profile.id,
    )

    provider = FakeEmbeddingProvider(dimensions=768)

    service = CandidateEvidenceEmbeddingService(
        database_session,
        provider,
    )

    first = await service.embed_profile_evidence(
        profile.id,
    )

    first_timestamp = first[0].embedded_at

    second = await service.embed_profile_evidence(
        profile.id,
        force=True,
    )

    assert second[0].embedded_at is not None
    assert first_timestamp is not None
    assert second[0].embedded_at >= first_timestamp


@pytest.mark.asyncio
async def test_provider_change_triggers_reembedding(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    evidence = await _create_evidence(
        database_session,
        profile_id=profile.id,
    )

    first_provider = FakeEmbeddingProvider(dimensions=768)

    first_service = CandidateEvidenceEmbeddingService(
        database_session,
        first_provider,
    )

    await first_service.embed_profile_evidence(
        profile.id,
    )

    evidence.embedding_model = "old-model"
    await database_session.commit()

    second_service = CandidateEvidenceEmbeddingService(
        database_session,
        first_provider,
    )

    await second_service.embed_profile_evidence(
        profile.id,
    )

    refreshed = await database_session.get(
        CandidateEvidence,
        evidence.id,
    )

    assert refreshed is not None
    assert refreshed.embedding_model == "fake-embedding-model"


@pytest.mark.asyncio
async def test_embed_single_evidence(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    evidence = await _create_evidence(
        database_session,
        profile_id=profile.id,
    )

    service = CandidateEvidenceEmbeddingService(
        database_session,
        FakeEmbeddingProvider(dimensions=768),
    )

    result = await service.embed_evidence(
        profile.id,
        evidence.id,
    )

    assert result.embedding is not None
    assert len(result.embedding) == 768
    assert result.embedding_provider == "fake"


@pytest.mark.asyncio
async def test_unknown_profile_is_rejected(
    database_session,
) -> None:
    service = CandidateEvidenceEmbeddingService(
        database_session,
        FakeEmbeddingProvider(dimensions=768),
    )

    with pytest.raises(
        CandidateEvidenceEmbeddingProfileNotFoundError,
    ):
        await service.embed_profile_evidence(uuid4())


@pytest.mark.asyncio
async def test_profile_without_evidence_is_rejected(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    service = CandidateEvidenceEmbeddingService(
        database_session,
        FakeEmbeddingProvider(dimensions=768),
    )

    with pytest.raises(
        CandidateEvidenceEmbeddingNotFoundError,
    ):
        await service.embed_profile_evidence(profile.id)


@pytest.mark.asyncio
async def test_unknown_evidence_is_rejected(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    service = CandidateEvidenceEmbeddingService(
        database_session,
        FakeEmbeddingProvider(dimensions=768),
    )

    with pytest.raises(
        CandidateEvidenceEmbeddingNotFoundError,
    ):
        await service.embed_evidence(
            profile.id,
            uuid4(),
        )


@pytest.mark.asyncio
async def test_wrong_vector_dimensions_are_rejected(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    await _create_evidence(
        database_session,
        profile_id=profile.id,
    )

    class WrongDimensionProvider:
        @property
        def provider_name(self) -> str:
            return "wrong"

        @property
        def model_name(self) -> str:
            return "wrong-model"

        @property
        def dimensions(self) -> int:
            return 768

        async def embed_text(self, text: str) -> list[float]:
            return [0.1] * 10

        async def embed_batch(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            return [
                [0.1] * 10
                for _ in texts
            ]

    service = CandidateEvidenceEmbeddingService(
        database_session,
        WrongDimensionProvider(),
    )

    with pytest.raises(
        CandidateEvidenceEmbeddingProviderFailureError,
        match="dimensions do not match",
    ):
        await service.embed_profile_evidence(
            profile.id,
        )


@pytest.mark.asyncio
async def test_provider_failure_is_wrapped(
    database_session,
) -> None:
    profile = await _create_profile(database_session)

    await _create_evidence(
        database_session,
        profile_id=profile.id,
    )

    class FailingProvider:
        @property
        def provider_name(self) -> str:
            return "failing"

        @property
        def model_name(self) -> str:
            return "failing-model"

        @property
        def dimensions(self) -> int:
            return 768

        async def embed_text(self, text: str) -> list[float]:
            raise RuntimeError("provider unavailable")

        async def embed_batch(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            raise RuntimeError("provider unavailable")

    service = CandidateEvidenceEmbeddingService(
        database_session,
        FailingProvider(),
    )

    with pytest.raises(
        CandidateEvidenceEmbeddingProviderFailureError,
        match="failed for candidate profile",
    ):
        await service.embed_profile_evidence(
            profile.id,
        )