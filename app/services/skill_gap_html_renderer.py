from __future__ import annotations

import html

from app.schemas.skill_gap_report import SkillGapReport


class SkillGapHtmlRenderer:
    def render(self, report: SkillGapReport) -> str:
        def esc(value: object) -> str:
            return html.escape(str(value))

        equivalences = "".join(
            f"<li><strong>{esc(item.job_requirement)} → {esc(item.candidate_term)}</strong>: "
            f"{esc(item.explanation)}</li>"
            for item in report.resolved_equivalences
        )
        gaps = "".join(
            f"<section class='gap {esc(gap.priority)}'>"
            f"<h2>{esc(gap.skill)} <span>{esc(gap.priority.title())} priority</span></h2>"
            f"<p>{esc(gap.why_it_matters)}</p>"
            + (
                f"<p><strong>Existing overlap:</strong> {esc(gap.candidate_overlap)}</p>"
                if gap.candidate_overlap
                else ""
            )
            + "<h3>Preparation topics</h3><ul>"
            + "".join(f"<li>{esc(item)}</li>" for item in gap.preparation_topics)
            + f"</ul><p><strong>Practical exercise:</strong> {esc(gap.practical_exercise)}</p>"
            + "<h3>Practice questions</h3><ul>"
            + "".join(f"<li>{esc(item)}</li>" for item in gap.interview_questions)
            + "</ul></section>"
            for gap in report.missing_skills
        )
        if not gaps:
            gaps = "<p>No genuine skill gaps were identified after LLM reassessment.</p>"
        strategy = "".join(f"<li>{esc(item)}</li>" for item in report.preparation_strategy)
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skill Gap Report | {esc(report.job_title)}</title>
<style>
body{{margin:0;background:#eef3f6;color:#172632;font:15px/1.5 Arial,sans-serif}}
main{{max-width:900px;margin:24px auto;padding:32px;background:white}}
h1{{color:#123f59;margin-bottom:4px}}
h2{{color:#176b8a;border-bottom:1px solid #c9dce5;padding-bottom:5px}}
h2 span{{float:right;font-size:13px;color:#6b4f00}}
h3{{margin-bottom:4px}}
.meta{{color:#536575}}
.score{{font-size:22px;font-weight:bold;color:#176b8a}}
.gap{{padding:8px 0;break-inside:avoid}}
li{{margin:4px 0}}
@media print{{body{{background:white}}main{{margin:0;max-width:none;padding:0}}}}
</style></head><body><main>
<h1>Skill Gap and Preparation Report</h1>
<p class="meta">{esc(report.job_title)} at {esc(report.company)}</p>
<p class="score">Overall match: {report.overall_match_score:.1f}%</p>
<h2>Summary</h2><p>{esc(report.executive_summary)}</p>
{f'<h2>Resolved Equivalences</h2><ul>{equivalences}</ul>' if equivalences else ''}
<h2>Skills to Prepare</h2>{gaps}
<h2>Preparation Strategy</h2><ol>{strategy}</ol>
</main></body></html>"""
