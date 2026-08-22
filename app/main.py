import logging

from fastapi import FastAPI
from sqlalchemy import text

from app.config import get_settings
from app.db import engine
from app.webhooks.router import router as webhooks_router

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="Dunning Agent", version="0.1.0")
app.include_router(webhooks_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness plus a real database round-trip, so Cloud Run does not route to
    an instance that cannot reach Cloud SQL."""
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return {"status": "ok"}
