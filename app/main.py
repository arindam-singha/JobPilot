import logging

from fastapi import FastAPI

from app.api.routes.application_packages import (
    router as application_packages_router,
)
from app.api.routes.candidate_documents import (
    router as candidate_documents_router,
)
from app.api.routes.candidate_evidence import (
    router as candidate_evidence_router,
)
from app.api.routes.candidate_evidence_generation import (
    router as candidate_evidence_generation_router,
)
from app.api.routes.candidate_profile import (
    router as candidate_profile_router,
)
from app.api.routes.health import (
    router as health_router,
)
from app.api.routes.hybrid_job_candidate_matching import (
    router as hybrid_job_candidate_matching_router,
)
from app.api.routes.job_candidate_matching import (
    router as job_candidate_matching_router,
)
from app.api.routes.job_requirements import (
    router as job_requirements_router,
)
from app.api.routes.jobs import (
    router as jobs_router,
)
from app.api.routes.llm_evidence_generation import (
    router as llm_evidence_generation_router,
)
from app.api.routes.tailored_resumes import (
    router as tailored_resumes_router,
)
from app.api.routes import job_applications

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | %(levelname)s | " "%(name)s | %(message)s"),
)


app = FastAPI(
    title="JobPilot API",
    version="0.1.0",
    description=("Human-in-the-loop Agentic RAG " "job application assistant foundation."),
)


app.include_router(health_router)
app.include_router(jobs_router)
app.include_router(candidate_profile_router)
app.include_router(candidate_documents_router)
app.include_router(candidate_evidence_router)
app.include_router(candidate_evidence_generation_router)
app.include_router(llm_evidence_generation_router)
app.include_router(job_requirements_router)
app.include_router(job_candidate_matching_router)
app.include_router(hybrid_job_candidate_matching_router)
app.include_router(tailored_resumes_router)
app.include_router(application_packages_router)
app.include_router(job_applications.router)

@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "JobPilot API",
    }
