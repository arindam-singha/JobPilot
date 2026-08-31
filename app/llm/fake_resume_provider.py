from __future__ import annotations

from collections.abc import Callable

from app.schemas.tailored_resume import (
    GroundedResumeStatement,
    ResumeGenerationContext,
    TailoredResumeContent,
    TailoredResumeSection,
)

FakeResumeResponder = Callable[[ResumeGenerationContext], TailoredResumeContent]


class FakeResumeGenerationProvider:
    """Deterministic grounded provider used in tests and local development."""

    def __init__(self, responder: FakeResumeResponder | None = None) -> None:
        self._responder = responder

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-resume-model"

    async def generate_resume(
        self,
        context: ResumeGenerationContext,
    ) -> TailoredResumeContent:
        if self._responder is not None:
            return self._responder(context)

        selected = context.grounding.selected_evidence
        primary = selected[0]
        summary = GroundedResumeStatement(
            text=primary.content,
            evidence_ids=[primary.evidence_id],
        )

        grouped: dict[str, list[GroundedResumeStatement]] = {}
        for evidence in selected:
            grouped.setdefault(evidence.evidence_type, []).append(
                GroundedResumeStatement(
                    text=evidence.content,
                    evidence_ids=[evidence.evidence_id],
                )
            )

        sections = [
            TailoredResumeSection(
                heading=evidence_type.replace("_", " ").title(),
                statements=statements,
            )
            for evidence_type, statements in sorted(grouped.items())
            if evidence_type != "skill"
        ]
        skills = grouped.get("skill", [])

        return TailoredResumeContent(
            target_title=context.job_title,
            professional_summary=[summary],
            sections=sections,
            skills=skills,
        )
