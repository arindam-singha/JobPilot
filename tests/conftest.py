import os

import httpx
import pytest_asyncio
from app.db.session import get_db
from app.main import app
from app.models import (
    CandidateAchievement,
    CandidateCertification,
    CandidateDocument,
    CandidateEducation,
    CandidateEvidence,
    CandidateExperience,
    CandidateProfile,
    CandidateProject,
    CandidatePublication,
    CandidateSkill,
    TailoredResume,
)
from app.models.job import Job
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://jobpilot:jobpilot@postgres:55432/jobpilot",
)


@pytest_asyncio.fixture
async def async_engine():
    engine = create_async_engine(
        os.environ["DATABASE_URL"],
        future=True,
        echo=False,
        pool_pre_ping=True,
    )

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def clean_database(async_engine):
    async with async_engine.begin() as connection:
        await connection.execute(delete(CandidateAchievement))
        await connection.execute(delete(CandidateCertification))
        await connection.execute(delete(CandidatePublication))
        await connection.execute(delete(CandidateProject))
        await connection.execute(delete(CandidateEducation))
        await connection.execute(delete(CandidateSkill))
        await connection.execute(delete(CandidateExperience))
        await connection.execute(delete(CandidateDocument))
        await connection.execute(delete(CandidateEvidence))
        await connection.execute(delete(TailoredResume))
        await connection.execute(delete(CandidateProfile))
        await connection.execute(delete(Job))


@pytest_asyncio.fixture
async def async_session_maker(async_engine):
    return async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@pytest_asyncio.fixture
async def database_session(async_session_maker):
    async with async_session_maker() as session:
        yield session


@pytest_asyncio.fixture
async def override_get_db(async_session_maker):
    async def _get_test_db():
        async with async_session_maker() as session:
            yield session

    app.dependency_overrides[get_db] = _get_test_db

    try:
        yield
    finally:
        app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client(override_get_db):
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        yield client
