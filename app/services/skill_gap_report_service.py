from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.skill_gap_provider import (
    SkillGapGenerationProvider,
    SkillGapGenerationProviderError,
)
from app.models.candidate_evidence import CandidateEvidence
from app.schemas.hybrid_job_candidate_match import HybridJobCandidateMatchRead
from app.schemas.job import JobRead
from app.schemas.skill_gap_report import (
    SkillGapEvidence,
    SkillGapGenerationContext,
    SkillGapReport,
    SkillGapReportContent,
)


class SkillGapReportGenerationError(Exception):
    pass


class SkillGapReportService:
    def __init__(self, session: AsyncSession, provider: SkillGapGenerationProvider) -> None:
        self.session = session
        self.provider = provider

    async def generate(self, *, job: JobRead, match: HybridJobCandidateMatchRead) -> SkillGapReport:
        evidence_rows = (
            await self.session.scalars(
                select(CandidateEvidence)
                .where(CandidateEvidence.profile_id == match.profile_id)
                .order_by(CandidateEvidence.created_at.asc())
            )
        ).all()
        skill_categories = [match.required_skills, match.preferred_skills]
        matched_skills = [
            item.requirement
            for category in skill_categories
            for item in category.requirements
            if item.matched
        ]
        preliminary_missing = [
            item.requirement
            for category in skill_categories
            for item in category.requirements
            if not item.matched
        ]
        context = SkillGapGenerationContext(
            job_id=job.id,
            profile_id=match.profile_id,
            job_title=job.title,
            company=job.company,
            job_description=job.description,
            overall_match_score=match.overall_score,
            matched_skills=list(dict.fromkeys(matched_skills)),
            preliminary_missing_skills=list(dict.fromkeys(preliminary_missing)),
            candidate_evidence=[
                SkillGapEvidence(
                    evidence_id=item.id,
                    evidence_type=item.evidence_type,
                    title=item.title,
                    content=item.content,
                )
                for item in evidence_rows
            ],
        )
        try:
            content = await self.provider.generate_skill_gap_report(context)
        except SkillGapGenerationProviderError as exc:
            raise SkillGapReportGenerationError("Skill-gap LLM provider failed") from exc
        self._validate_coverage(content, context.preliminary_missing_skills)
        return SkillGapReport(
            job_id=job.id,
            profile_id=match.profile_id,
            job_title=job.title,
            company=job.company,
            overall_match_score=match.overall_score,
            matched_skills=context.matched_skills,
            executive_summary=content.executive_summary,
            resolved_equivalences=content.resolved_equivalences,
            missing_skills=content.missing_skills,
            preparation_strategy=content.preparation_strategy,
            generator_provider=self.provider.provider_name,
            generator_model=self.provider.model_name,
        )

    @staticmethod
    def _validate_coverage(
        content: SkillGapReportContent,
        preliminary_missing: list[str],
    ) -> None:
        expected = set(preliminary_missing)
        reported = {item.skill for item in content.missing_skills}
        resolved = {item.job_requirement for item in content.resolved_equivalences}
        if reported & resolved or reported | resolved != expected:
            raise SkillGapReportGenerationError(
                "Skill-gap LLM did not classify every preliminary missing skill exactly once"
            )
