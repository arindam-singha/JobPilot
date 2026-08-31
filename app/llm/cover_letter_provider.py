from __future__ import annotations

from typing import Protocol

from app.schemas.cover_letter import (
    CoverLetterGenerationContext,
    TailoredCoverLetterContent,
)


class CoverLetterGenerationProviderError(Exception):
    """Raised when structured cover-letter generation fails."""


class CoverLetterGenerationProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def generate_cover_letter(
        self,
        context: CoverLetterGenerationContext,
    ) -> TailoredCoverLetterContent: ...
