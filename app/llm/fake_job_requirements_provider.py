from __future__ import annotations

from collections.abc import Callable

from app.schemas.job_requirements import JobRequirementsData

FakeJobRequirementsResponder = Callable[
    [str, str, str | None, str],
    JobRequirementsData,
]


class FakeJobRequirementsProvider:
    """Deterministic provider used in tests and local development."""

    def __init__(
        self,
        responder: FakeJobRequirementsResponder | None = None,
    ) -> None:
        self._responder = responder

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-job-requirements-model"

    async def extract_requirements(
        self,
        *,
        title: str,
        company: str,
        location: str | None,
        description: str,
    ) -> JobRequirementsData:
        if self._responder is not None:
            return self._responder(
                title,
                company,
                location,
                description,
            )

        return self._extract_deterministically(
            title=title,
            description=description,
        )

    @staticmethod
    def _extract_deterministically(
        *,
        title: str,
        description: str,
    ) -> JobRequirementsData:
        combined_text = f"{title}\n{description}".casefold()

        required_skills: list[str] = []
        preferred_skills: list[str] = []
        required_experience: list[str] = []
        domain_keywords: list[str] = []

        skill_patterns = {
            "python": "Python",
            "pytorch": "PyTorch",
            "tensorflow": "TensorFlow",
            "fastapi": "FastAPI",
            "docker": "Docker",
            "kubernetes": "Kubernetes",
            "postgresql": "PostgreSQL",
            "sql": "SQL",
            "ros2": "ROS2",
            "ros 2": "ROS2",
            "computer vision": "Computer Vision",
            "deep learning": "Deep Learning",
            "machine learning": "Machine Learning",
            "llm": "LLM",
            "rag": "RAG",
            "isaac sim": "Isaac Sim",
            "isaac lab": "Isaac Lab",
        }

        for pattern, normalized_name in skill_patterns.items():
            if pattern in combined_text:
                required_skills.append(normalized_name)

        experience_patterns = {
            "computer vision": "Computer vision systems",
            "deep learning": "Deep-learning model development",
            "machine learning": "Machine-learning system development",
            "production ai": "Production AI systems",
            "autonomous systems": "Autonomous systems",
            "manufacturing": "Manufacturing applications",
            "robotics": "Robotics engineering",
        }

        for pattern, normalized_name in experience_patterns.items():
            if pattern in combined_text:
                required_experience.append(normalized_name)

        domain_patterns = {
            "robotics": "robotics",
            "automotive": "automotive",
            "manufacturing": "manufacturing",
            "autonomous": "autonomous systems",
            "artificial intelligence": "artificial intelligence",
            "computer vision": "computer vision",
        }

        for pattern, normalized_name in domain_patterns.items():
            if pattern in combined_text:
                domain_keywords.append(normalized_name)

        minimum_experience_years = (
            FakeJobRequirementsProvider._extract_minimum_experience_years(
                combined_text
            )
        )

        return JobRequirementsData(
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            required_experience=required_experience,
            responsibilities=[],
            education_requirements=[],
            certifications=[],
            domain_keywords=domain_keywords,
            minimum_experience_years=minimum_experience_years,
        )

    @staticmethod
    def _extract_minimum_experience_years(
        text: str,
    ) -> float | None:
        import re

        patterns = [
            r"(\d+(?:\.\d+)?)\+?\s+years?\s+of\s+experience",
            r"minimum\s+of\s+(\d+(?:\.\d+)?)\s+years?",
            r"at\s+least\s+(\d+(?:\.\d+)?)\s+years?",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)

            if match is not None:
                return float(match.group(1))

        return None