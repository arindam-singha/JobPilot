from __future__ import annotations

from typing import Protocol

from app.schemas.job_requirements import JobRequirementsData


class JobRequirementsProviderError(Exception):
    """Base exception for job-requirement provider failures."""


class JobRequirementsProvider(Protocol):
    """Interface implemented by structured job-requirement providers."""

    @property
    def provider_name(self) -> str:
        """Stable provider identifier stored with extracted requirements."""

    @property
    def model_name(self) -> str:
        """Model identifier stored with extracted requirements."""

    async def extract_requirements(
        self,
        *,
        title: str,
        company: str,
        location: str | None,
        description: str,
    ) -> JobRequirementsData:
        """Extract structured requirements from a job description."""