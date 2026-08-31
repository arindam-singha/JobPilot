class HealthService:
    """Simple health service placeholder for the API boundary."""

    async def get_health_status(self) -> dict[str, str]:
        return {"status": "ok"}
