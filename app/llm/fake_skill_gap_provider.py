from __future__ import annotations

from app.schemas.skill_gap_report import (
    MissingSkillPreparation,
    SkillGapGenerationContext,
    SkillGapReportContent,
)


class FakeSkillGapGenerationProvider:
    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-skill-gap-v1"

    async def generate_skill_gap_report(
        self, context: SkillGapGenerationContext
    ) -> SkillGapReportContent:
        gaps = [
            MissingSkillPreparation(
                skill=skill,
                priority="high",
                why_it_matters=f"{skill} is requested for this role.",
                preparation_topics=[f"Core concepts of {skill}"],
                practical_exercise=f"Build a small demonstrator using {skill}.",
                interview_questions=[f"How would you apply {skill} in this role?"],
            )
            for skill in context.preliminary_missing_skills
        ]
        return SkillGapReportContent(
            executive_summary=(
                f"Preparation plan for {len(gaps)} skill gap(s) identified for "
                f"the {context.job_title} role."
            ),
            missing_skills=gaps,
            preparation_strategy=["Study high-priority gaps before medium-priority gaps."],
        )
