from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from uuid import UUID

from app.db.session import AsyncSessionLocal, engine
from app.embeddings.embedding_provider_factory import create_embedding_provider
from app.llm.cover_letter_provider_factory import create_cover_letter_generation_provider
from app.llm.job_requirements_provider_factory import create_job_requirements_provider
from app.llm.resume_provider_factory import create_resume_generation_provider
from app.schemas.job import JobIngestRequest
from app.services.candidate_evidence_embedding_service import CandidateEvidenceEmbeddingService
from app.services.cover_letter_generation_service import CoverLetterGenerationService
from app.services.cover_letter_markdown_renderer import CoverLetterMarkdownRenderer
from app.services.cover_letter_pdf_renderer import CoverLetterPdfRenderer
from app.services.job_ingestion_service import JobIngestionService
from app.services.job_requirements_extraction_service import JobRequirementsExtractionService
from app.services.tailored_resume_generation_service import TailoredResumeGenerationService
from app.services.tailored_resume_markdown_renderer import TailoredResumeMarkdownRenderer
from app.services.tailored_resume_pdf_renderer import TailoredResumePdfRenderer
from sqlalchemy.ext.asyncio import AsyncSession


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a grounded resume and cover letter for one job."
    )
    parser.add_argument("--profile-id", required=True, type=UUID)
    job = parser.add_mutually_exclusive_group(required=True)
    job.add_argument("--job-id", type=UUID)
    job.add_argument("--job-file", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--company")
    parser.add_argument("--location")
    parser.add_argument("--job-url")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "application"


def _require_job_metadata(args: argparse.Namespace) -> None:
    missing = [name for name in ("title", "company", "job_url") if not getattr(args, name)]
    if missing:
        names = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise SystemExit(f"{names} required when --job-file is used")


async def _resolve_job_id(args: argparse.Namespace, session: AsyncSession) -> UUID:
    if args.job_id is not None:
        return args.job_id

    _require_job_metadata(args)
    description = args.job_file.read_text(encoding="utf-8").strip()
    if len(description) < 20:
        raise SystemExit("Job-description file must contain at least 20 characters")

    result = await JobIngestionService().ingest_job(
        session,
        JobIngestRequest(
            title=args.title,
            company=args.company,
            location=args.location,
            job_url=args.job_url,
            job_description=description,
        ),
    )
    print(f"Job {'created' if result.created else 'reused'}: {result.job.id}")
    return result.job.id


async def _run(args: argparse.Namespace) -> Path:
    embedding_provider = create_embedding_provider()
    requirements_provider = create_job_requirements_provider()
    resume_provider = create_resume_generation_provider()
    cover_letter_provider = create_cover_letter_generation_provider()

    async with AsyncSessionLocal() as session:
        job_id = await _resolve_job_id(args, session)

        evidence = await CandidateEvidenceEmbeddingService(
            session=session,
            provider=embedding_provider,
        ).embed_profile_evidence(
            args.profile_id,
            force=args.force_embeddings,
        )
        print(f"Candidate evidence ready: {len(evidence)} record(s)")

        requirements = await JobRequirementsExtractionService(
            session,
            requirements_provider,
        ).extract_and_persist(job_id)
        requirement_count = sum(
            len(items)
            for items in (
                requirements.required_skills,
                requirements.preferred_skills,
                requirements.required_experience,
                requirements.education_requirements,
                requirements.certifications,
            )
        )
        if requirement_count == 0:
            raise SystemExit(
                "Requirement extraction returned no matchable requirements. "
                "Check that the job file contains the complete description."
            )
        print(f"Structured requirements ready: {requirement_count}")

        resume = await TailoredResumeGenerationService(
            session,
            embedding_provider,
            resume_provider,
        ).generate(job_id=job_id, profile_id=args.profile_id)

        cover_letter = await CoverLetterGenerationService(
            session,
            embedding_provider,
            cover_letter_provider,
        ).generate(job_id=job_id, profile_id=args.profile_id)

    directory = args.output_dir / f"{_slug(cover_letter.company)}-{_slug(cover_letter.job_title)}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "resume.json").write_text(resume.model_dump_json(indent=2), encoding="utf-8")
    (directory / "resume.md").write_text(
        TailoredResumeMarkdownRenderer().render(resume), encoding="utf-8"
    )
    (directory / "resume.pdf").write_bytes(TailoredResumePdfRenderer().render(resume))
    (directory / "cover-letter.json").write_text(
        cover_letter.model_dump_json(indent=2), encoding="utf-8"
    )
    (directory / "cover-letter.md").write_text(
        CoverLetterMarkdownRenderer().render(cover_letter), encoding="utf-8"
    )
    (directory / "cover-letter.pdf").write_bytes(CoverLetterPdfRenderer().render(cover_letter))
    manifest = {
        "job_id": str(job_id),
        "profile_id": str(args.profile_id),
        "resume_evidence_ids": [str(item.evidence_id) for item in resume.evidence_catalog],
        "cover_letter_evidence_ids": [
            str(item.evidence_id) for item in cover_letter.evidence_catalog
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


async def _main() -> None:
    try:
        directory = await _run(_parse_args())
        print(f"Application package generated: {directory.resolve()}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
