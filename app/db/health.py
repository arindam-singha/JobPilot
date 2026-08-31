from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def database_is_healthy(session: AsyncSession) -> bool:
    try:
        await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
