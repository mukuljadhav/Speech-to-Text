"""
Campaign Management AI Service -- entrypoint.

Run with:
    uvicorn app.main:app --reload --port 8000

Docs available at:
    http://localhost:8000/docs
"""

import logging

from fastapi import FastAPI

from app.routers import campaign

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Campaign Management AI Service",
    description="Generates marketing banner campaigns from text (Phase 1), voice, and image input using Gemma 4.",
    version="0.1.0",
)

app.include_router(campaign.router)


@app.get("/health")
def health_check():
    """Simple liveness check -- useful once this is deployed behind another backend."""
    return {"status": "ok", "service": "campaign-management-ai-service", "phase": "1 - text only"}
