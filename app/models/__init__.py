"""ORM model package."""

from app.models.candidate_document import CandidateDocument
from app.models.candidate_evidence import CandidateEvidence
from app.models.candidate_profile import (
    CandidateAchievement,
    CandidateCertification,
    CandidateEducation,
    CandidateExperience,
    CandidateProfile,
    CandidateProject,
    CandidatePublication,
    CandidateSkill,
)
from app.models.job import Job
from app.models.job_requirements import JobRequirements
from app.models.tailored_resume import TailoredResume

__all__ = [
    "CandidateAchievement",
    "CandidateCertification",
    "CandidateDocument",
    "CandidateEducation",
    "CandidateExperience",
    "CandidateProfile",
    "CandidateProject",
    "CandidatePublication",
    "CandidateSkill",
    "Job",
    "CandidateEvidence",
    "JobRequirements",
    "TailoredResume",
]
