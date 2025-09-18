"""Fallback keyword discovery utilities."""

from __future__ import annotations

import logging
import re
from typing import Dict, List

LOGGER = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _derive_keywords(site_url: str) -> List[str]:
    tokens = _WORD_RE.findall(site_url.lower())
    if not tokens:
        tokens = ["marketing", "growth", "strategy"]
    unique = []
    for token in tokens:
        if token not in unique:
            unique.append(token)
    seed = unique[:10]
    while len(seed) < 10:
        seed.append(f"insight-{len(seed)+1}")
    return seed


def keyword_backfill(site_url: str) -> Dict[str, List[str]]:
    """Generate fallback keywords when no SEO report is available."""

    LOGGER.info("Running keyword backfill for site %s", site_url)
    seed = _derive_keywords(site_url)
    primary = [f"{token} strategy" for token in seed[:5]]
    secondary = [f"{token} tips" for token in seed[5:10]]
    gaps = [f"Need more coverage on {token}" for token in seed[:5]]
    return {
        "primary_keywords": primary,
        "secondary_keywords": secondary,
        "gaps": gaps,
    }


__all__ = ["keyword_backfill"]
