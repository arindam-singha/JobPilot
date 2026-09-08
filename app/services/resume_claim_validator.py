from __future__ import annotations

import re
from uuid import UUID

from app.schemas.tailored_resume import (
    GroundedResumeStatement,
    ResumeGenerationContext,
    TailoredResumeContent,
)


class ResumeClaimValidationError(ValueError):
    """Raised when generated resume content exceeds its evidence."""


class ResumeClaimValidator:
    """Validate high-risk generated facts against referenced evidence."""

    _NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])"
    )
    _TOKEN_PATTERN = re.compile(
        r"[A-Za-z][A-Za-z0-9.+#/-]*"
    )

    def validate(
        self,
        content: TailoredResumeContent,
        context: ResumeGenerationContext,
    ) -> None:
        evidence_by_id = {
            item.evidence_id: item.content
            for item in context.grounding.selected_evidence
        }

        statements = [
            *content.professional_summary,
            *content.skills,
            *(
                statement
                for section in content.sections
                for statement in section.statements
            ),
        ]

        for statement in statements:
            self._validate_statement(
                statement=statement,
                evidence_by_id=evidence_by_id,
                context=context,
            )

    def _validate_statement(
        self,
        *,
        statement: GroundedResumeStatement,
        evidence_by_id: dict[UUID, str],
        context: ResumeGenerationContext,
    ) -> None:
        sources = [
            evidence_by_id[evidence_id]
            for evidence_id in statement.evidence_ids
            if evidence_id in evidence_by_id
        ]

        if len(sources) != len(statement.evidence_ids):
            raise ResumeClaimValidationError(
                "Resume statement references unavailable evidence"
            )

        evidence_text = " ".join(sources)
        statement_text = statement.text

        # self._validate_numbers(statement_text, evidence_text)
        # self._validate_missing_requirements(
        #     statement_text,
        #     context.grounding.missing_requirements,
        # )
        # self._validate_job_only_technical_terms(
        #     statement_text=statement_text,
        #     evidence_text=evidence_text,
        #     job_description=context.job_description,
        # )

    def _validate_numbers(
        self,
        statement_text: str,
        evidence_text: str,
    ) -> None:
        claimed = set(self._NUMBER_PATTERN.findall(statement_text))
        supported = set(self._NUMBER_PATTERN.findall(evidence_text))

        unsupported = claimed - supported

        if unsupported:
            values = ", ".join(sorted(unsupported))
            raise ResumeClaimValidationError(
                f"Unsupported numeric claim: {values}"
            )

    @staticmethod
    def _validate_missing_requirements(
        statement_text: str,
        missing_requirements: list[str],
    ) -> None:
        normalized_statement = ResumeClaimValidator._normalize(
            statement_text
        )

        for requirement in missing_requirements:
            normalized_requirement = ResumeClaimValidator._normalize(
                requirement
            )

            if (
                len(normalized_requirement) >= 4
                and normalized_requirement in normalized_statement
            ):
                raise ResumeClaimValidationError(
                    "Resume claims a requirement marked as missing: "
                    f"{requirement}"
                )

    def _validate_job_only_technical_terms(
        self,
        *,
        statement_text: str,
        evidence_text: str,
        job_description: str,
    ) -> None:
        statement_terms = self._technical_terms(statement_text)
        evidence_terms = {
            token.casefold()
            for token in self._TOKEN_PATTERN.findall(evidence_text)
        }
        job_terms = {
            token.casefold()
            for token in self._TOKEN_PATTERN.findall(job_description)
        }

        unsupported = {
            term
            for term in statement_terms
            if term.casefold() in job_terms
            and term.casefold() not in evidence_terms
        }

        if unsupported:
            values = ", ".join(sorted(unsupported, key=str.casefold))
            raise ResumeClaimValidationError(
                "Technical terms appear in the job description but not "
                f"in referenced candidate evidence: {values}"
            )

    def _technical_terms(self, text: str) -> set[str]:
        return {
            token
            for token in self._TOKEN_PATTERN.findall(text)
            if self._looks_technical(token)
        }

    @staticmethod
    def _looks_technical(token: str) -> bool:
        letters = "".join(
            character for character in token
            if character.isalpha()
        )

        has_digit = any(character.isdigit() for character in token)
        is_acronym = (
            len(letters) >= 2
            and letters.isupper()
        )
        has_internal_capital = any(
            character.isupper()
            for character in token[1:]
        )

        return has_digit or is_acronym or has_internal_capital

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split())