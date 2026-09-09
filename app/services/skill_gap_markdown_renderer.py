from __future__ import annotations

from app.schemas.skill_gap_report import SkillGapReport


class SkillGapMarkdownRenderer:
    def render(self, report: SkillGapReport) -> str:
        lines = [
            "# Skill Gap and Preparation Report",
            "",
            f"**Role:** {report.job_title}",
            f"**Company:** {report.company}",
            f"**Overall match:** {report.overall_match_score:.1f}%",
            "",
            "## Summary",
            "",
            report.executive_summary,
        ]
        if report.resolved_equivalences:
            lines.extend(["", "## Resolved Equivalences", ""])
            for item in report.resolved_equivalences:
                lines.append(
                    f"- **{item.job_requirement} → {item.candidate_term}:** " f"{item.explanation}"
                )
        lines.extend(["", "## Skills to Prepare", ""])
        if not report.missing_skills:
            lines.append("No genuine skill gaps were identified after LLM reassessment.")
        for gap in report.missing_skills:
            lines.extend(
                [
                    f"### {gap.skill} ({gap.priority.title()} priority)",
                    "",
                    gap.why_it_matters,
                ]
            )
            if gap.candidate_overlap:
                lines.extend(["", f"**Existing overlap:** {gap.candidate_overlap}"])
            lines.extend(["", "**Preparation topics:**", ""])
            lines.extend(f"- {topic}" for topic in gap.preparation_topics)
            lines.extend(
                [
                    "",
                    f"**Practical exercise:** {gap.practical_exercise}",
                    "",
                    "**Practice questions:**",
                    "",
                ]
            )
            lines.extend(f"- {question}" for question in gap.interview_questions)
            lines.append("")
        lines.extend(["## Preparation Strategy", ""])
        lines.extend(
            f"{index}. {item}" for index, item in enumerate(report.preparation_strategy, 1)
        )
        return "\n".join(lines).strip() + "\n"
