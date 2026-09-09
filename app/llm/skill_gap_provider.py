from __future__ import annotations

from typing import Protocol

from app.schemas.skill_gap_report import SkillGapGenerationContext, SkillGapReportContent


class SkillGapGenerationProviderError(Exception):
    """Raised when skill-gap report generation fails."""


class SkillGapGenerationProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def generate_skill_gap_report(
        self, context: SkillGapGenerationContext
    ) -> SkillGapReportContent: ...
