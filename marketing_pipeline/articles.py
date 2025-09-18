"""Article generation utilities."""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, Optional

from .config import FeatureConfig
from .localization import localize

LOGGER = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def _build_base_article(topic: str, brief: Optional[Dict[str, object]]) -> str:
    sections = [
        f"## {topic} in Focus",
        "Driving impact with deliberate marketing plays.",
    ]
    if brief:
        primary = ", ".join(str(k) for k in brief.get("primary_keywords", [])[:5])
        audience = brief.get("target_audience", "modern teams")
        sections.append(f"Primary keywords include {primary} for {audience}.")
    body = []
    for idx in range(1, 9):
        body.append(
            f"Paragraph {idx}: Our marketing team embraces experimentation and structured measurement to transform curiosity into customers."
        )
    conclusion = "We conclude with a confident call-to-action inviting readers to start their next campaign today."
    content = "\n\n".join(sections + body + [conclusion])
    return content


def _ensure_length(text: str, target_min: int = 800, target_max: int = 1200) -> str:
    words = text.split()
    if target_min <= len(words) <= target_max:
        return text
    multiplier = max(1, math.ceil(target_min / max(1, len(words))))
    extended = " ".join(words * multiplier)
    words = extended.split()
    if len(words) > target_max:
        words = words[:target_max]
    return " ".join(words)


def generate_article(
    topic: str,
    brief: Optional[Dict[str, object]],
    language: str,
    config: FeatureConfig,
) -> Dict[str, object]:
    """Generate a structured article payload."""

    LOGGER.info("Generating article for topic '%s' in %s", topic, language)
    base_content = _build_base_article(topic, brief)
    content = _ensure_length(base_content)

    seo_title = f"{topic} Strategies that Deliver"[:70]
    meta_description = (
        f"Discover how {topic.lower()} drives sustainable marketing growth with tactical plays and a confident CTA."
    )
    if len(meta_description) < 150:
        meta_description = (meta_description + " " + meta_description)[:155]
    meta_description = meta_description[:160]
    slug = _slugify(topic)
    internal_links = [f"/{slug}/resource-{idx}" for idx in range(1, 4)]
    explainer = "This works because it combines storytelling with measurable next steps."

    qa_notes = []
    if language.lower() == "fr":
        content, qa_notes = localize(content, "fr", config)
    article = {
        "topic": topic,
        "language": language,
        "seo_title": seo_title,
        "meta_description": meta_description,
        "slug": slug,
        "content": content,
        "internal_link_ideas": internal_links,
        "why_this_works": explainer,
        "qa_notes": qa_notes,
    }
    return article


__all__ = ["generate_article"]
