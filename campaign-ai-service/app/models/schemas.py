"""
Pydantic schemas: define the exact shape of every request and response
the API accepts/returns. FastAPI uses these for validation + auto docs.
"""

from typing import Optional
from pydantic import BaseModel, Field


class CampaignContent(BaseModel):
    """
    The structured creative brief that Gemma 4 produces from the input.
    This is intentionally simple in Phase 1 -- just enough fields to draw
    a clean banner. We can extend it later without breaking the API.
    """

    headline: str
    subheadline: str
    cta: str  # call-to-action, e.g. "Shop Now"
    background_color: str  # hex code, e.g. "#FF5733"
    accent_color: str      # hex code, used for CTA button / highlights
    text_color: str        # hex code, used for text (must contrast background)


class CampaignResponse(BaseModel):
    """What the API returns to the caller (another backend)."""

    status: str  # "success" or "error"
    banner_path: Optional[str] = None
    campaign_content: Optional[CampaignContent] = None
    source_text: Optional[str] = Field(
        default=None,
        description="The resolved text Gemma 4 actually used (shows which inputs contributed and how they were combined).",
    )
    message: Optional[str] = None