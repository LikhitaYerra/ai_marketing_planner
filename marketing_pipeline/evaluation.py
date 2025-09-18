"""Lightweight evaluation for pipeline runs."""

from __future__ import annotations

import logging
from typing import Dict, List

from .config import FeatureConfig

LOGGER = logging.getLogger(__name__)


def _keyword_coverage(content: str, keywords: List[str]) -> str:
    hits = sum(1 for keyword in keywords if keyword.lower() in content.lower())
    if hits >= max(1, len(keywords) * 0.6):
        return "high"
    if hits >= max(1, len(keywords) * 0.3):
        return "medium"
    return "low"


def evaluate_run(result: Dict[str, object], brief: Dict[str, object], config: FeatureConfig) -> Dict[str, object]:
    """Evaluate the run and surface actionable insights."""

    if not config.FEATURE_RUN_REPORT:
        LOGGER.info("Evaluation skipped - reporting disabled")
        return {}

    articles = result.get("articles") or []
    primary = brief.get("primary_keywords", []) if brief else []
    secondary = brief.get("secondary_keywords", []) if brief else []

    coverage_primary = "low"
    coverage_secondary = "low"
    if articles:
        content = " ".join(article.get("content", "") for article in articles)
        coverage_primary = _keyword_coverage(content, primary)
        coverage_secondary = _keyword_coverage(content, secondary)

    def _flatten_posts(field: str) -> bool:
        if field == "utm" and not config.FEATURE_UTM_LINKS:
            return True
        posts = result.get("posts") or {}
        for language_posts in posts.values():
            for variant_list in language_posts.values():
                if not variant_list:
                    return False
                if field == "utm" and not any(item.get("utm_url") for item in variant_list):
                    return False
                if field == "hashtags" and not any(item.get("hashtags") for item in variant_list):
                    return False
        return True if posts else False

    evaluation = {
        "coverage_score": {
            "primary": coverage_primary,
            "secondary": coverage_secondary,
        },
        "checks": {
            "cta_present": any("call-to-action" in (article.get("content", "").lower()) for article in articles),
            "length_within_target": all(
                800 <= len(article.get("content", "").split()) <= 1200 for article in articles
            ),
            "utm_present": _flatten_posts("utm"),
            "hashtags_present": _flatten_posts("hashtags"),
        },
        "insights_next_actions": [
            "Double down on high-performing keyword clusters.",
            "Introduce a sharper CTA above the fold.",
            "Align social imagery with the article narrative.",
            "Promote the brief internally for brand alignment.",
            "Test alternative hooks for Instagram Reels.",
            "Expand LinkedIn posts with data points.",
            "Create follow-up nurturing emails.",
            "Monitor keyword shifts after publication.",
            "Capture learnings in the knowledge base.",
            "Review analytics to fine-tune targeting.",
        ],
    }
    LOGGER.info("Evaluation complete with coverage primary=%s", coverage_primary)
    return evaluation


__all__ = ["evaluate_run"]
