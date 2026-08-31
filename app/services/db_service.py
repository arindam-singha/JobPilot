from sqlalchemy.ext.asyncio import AsyncSession

from app.db.health import database_is_healthy


class DatabaseService:
    """Application-level database service boundary."""

    async def health_check(self, session: AsyncSession) -> bool:
        return await database_is_healthy(session)
