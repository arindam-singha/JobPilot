import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
async def test_database_connection_smoke(database_session: AsyncSession) -> None:
    result = await database_session.execute(text("SELECT 1"))
    value = result.scalar_one()
    assert value == 1
