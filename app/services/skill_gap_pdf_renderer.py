from __future__ import annotations

import html
from io import BytesIO

from reportlab.lib.pagesizes import A4  # type: ignore[import-untyped]
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore[import-untyped]
from reportlab.lib.units import inch  # type: ignore[import-untyped]
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer  # type: ignore[import-untyped]

from app.schemas.skill_gap_report import SkillGapReport


class SkillGapPdfRenderer:
    def render(self, report: SkillGapReport) -> bytes:
        output = BytesIO()
        styles = getSampleStyleSheet()
        title = ParagraphStyle("GapTitle", parent=styles["Title"], textColor="#123f59")
        heading = ParagraphStyle(
            "GapHeading", parent=styles["Heading2"], textColor="#176b8a", spaceBefore=10
        )
        subheading = ParagraphStyle(
            "GapSubheading", parent=styles["Heading3"], textColor="#123f59", spaceBefore=8
        )
        body = ParagraphStyle("GapBody", parent=styles["BodyText"], leading=14, spaceAfter=5)
        document = SimpleDocTemplate(
            output,
            pagesize=A4,
            rightMargin=0.65 * inch,
            leftMargin=0.65 * inch,
            topMargin=0.6 * inch,
            bottomMargin=0.6 * inch,
            title=f"Skill Gap Report - {report.job_title}",
        )

        def safe(value: object) -> str:
            return html.escape(str(value))

        story: list[object] = [
            Paragraph("Skill Gap and Preparation Report", title),
            Paragraph(f"{safe(report.job_title)} at {safe(report.company)}", body),
            Paragraph(f"<b>Overall match: {report.overall_match_score:.1f}%</b>", body),
            Paragraph("Summary", heading),
            Paragraph(safe(report.executive_summary), body),
        ]
        if report.resolved_equivalences:
            story.append(Paragraph("Resolved Equivalences", heading))
            for item in report.resolved_equivalences:
                equivalence = (
                    f"• <b>{safe(item.job_requirement)} → "
                    f"{safe(item.candidate_term)}</b>: {safe(item.explanation)}"
                )
                story.append(
                    Paragraph(
                        equivalence,
                        body,
                    )
                )
        story.append(Paragraph("Skills to Prepare", heading))
        if not report.missing_skills:
            story.append(
                Paragraph("No genuine skill gaps were identified after LLM reassessment.", body)
            )
        for gap in report.missing_skills:
            story.extend(
                [
                    Paragraph(f"{safe(gap.skill)} ({gap.priority.title()} priority)", subheading),
                    Paragraph(safe(gap.why_it_matters), body),
                ]
            )
            if gap.candidate_overlap:
                story.append(
                    Paragraph(f"<b>Existing overlap:</b> {safe(gap.candidate_overlap)}", body)
                )
            story.append(Paragraph("<b>Preparation topics</b>", body))
            story.extend(Paragraph(f"• {safe(item)}", body) for item in gap.preparation_topics)
            story.append(
                Paragraph(f"<b>Practical exercise:</b> {safe(gap.practical_exercise)}", body)
            )
            story.append(Paragraph("<b>Practice questions</b>", body))
            story.extend(Paragraph(f"• {safe(item)}", body) for item in gap.interview_questions)
            story.append(Spacer(1, 6))
        story.append(Paragraph("Preparation Strategy", heading))
        story.extend(
            Paragraph(f"{index}. {safe(item)}", body)
            for index, item in enumerate(report.preparation_strategy, 1)
        )
        document.build(story)
        result = output.getvalue()
        if not result.startswith(b"%PDF-"):
            raise ValueError("Skill-gap renderer did not produce a valid PDF")
        return result
