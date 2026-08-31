from __future__ import annotations

from typing import Protocol

from app.schemas.tailored_resume import ResumeGenerationContext, TailoredResumeContent


class ResumeGenerationProviderError(Exception):
    """Base exception for grounded resume-generation provider failures."""


class ResumeGenerationProvider(Protocol):
    """Interface implemented by structured resume-generation providers."""

    @property
    def provider_name(self) -> str:
        """Stable provider identifier stored with generated resumes."""

    @property
    def model_name(self) -> str:
        """Model identifier stored with generated resumes."""

    async def generate_resume(
        self,
        context: ResumeGenerationContext,
    ) -> TailoredResumeContent:
        """Generate structured resume content from a trusted grounding context."""
