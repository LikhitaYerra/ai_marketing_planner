"""Content brief composition utilities."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)


def build_content_brief(keyword_data: Optional[Dict[str, List[str]]]) -> Optional[Dict[str, object]]:
    """Create a structured brief from keyword data.

    Returns None when no keyword data is available.
    """

    if not keyword_data:
        LOGGER.info("Content brief skipped due to missing keyword data")
        return None

    primary_pool = keyword_data.get("top_keywords") or keyword_data.get("primary_keywords") or []
    secondary_pool = keyword_data.get("keyword_gaps") or keyword_data.get("secondary_keywords") or []
    gaps = keyword_data.get("keyword_gaps") or keyword_data.get("gaps") or []

    def _pad(pool: List[str], minimum: int, maximum: int, filler_prefix: str) -> List[str]:
        items = list(dict.fromkeys(pool))  # preserve order while deduplicating
        while len(items) < minimum:
            items.append(f"{filler_prefix}-{len(items)+1}")
        return items[:maximum]

    brief = {
        "target_audience": "Growth-focused marketing teams seeking predictable pipeline.",
        "primary_keywords": _pad(primary_pool, 10, 12, "primary"),
        "secondary_keywords": _pad(secondary_pool or primary_pool, 10, 12, "secondary"),
        "brand_voice": "professional, friendly",
        "goals": ["traffic growth", "education", "lead generation"],
        "tone_constraints": ["concise", "avoid jargon"],
        "gaps": gaps,
    }
    LOGGER.info("Built content brief with %d primary keywords", len(brief["primary_keywords"]))
    return brief


__all__ = ["build_content_brief"]
