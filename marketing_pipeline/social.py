"""Social post generation utilities."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

from .config import FeatureConfig

LOGGER = logging.getLogger(__name__)

CHANNEL_TONES = {
    "facebook": "community-oriented",
    "instagram": "punchy",
    "linkedin": "insight-driven",
}


def _build_hashtags(keywords: Iterable[str], minimum: int = 6, maximum: int = 10) -> List[str]:
    tags = []
    for keyword in keywords:
        tag = "#" + "".join(part.title() for part in keyword.split())
        if tag.lower() not in {t.lower() for t in tags}:
            tags.append(tag)
        if len(tags) >= maximum:
            break
    while len(tags) < minimum:
        tags.append(f"#GrowthPlay{len(tags)+1}")
    return tags[:maximum]


def _build_body(channel: str, topic: str, tone: str, language: str) -> str:
    prefix = {
        "facebook": "Let's spark a conversation",
        "instagram": "Swipe-worthy insight",
        "linkedin": "Strategy spotlight",
    }.get(channel, "Share")
    body = f"{prefix} on {topic}. This {tone} update keeps your {language.upper()} audience engaged."
    return body


def generate_social_posts(
    topic: str,
    slug: str,
    brief: Optional[Dict[str, object]],
    channels: Iterable[str],
    language: str,
    config: FeatureConfig,
) -> Dict[str, List[Dict[str, object]]]:
    """Generate social posts per channel respecting feature flags."""

    keywords = brief.get("primary_keywords", []) if brief else [topic]
    hashtags = _build_hashtags(keywords)
    variants = 2 if config.FEATURE_SOCIAL_AB_TESTING else 1
    posts: Dict[str, List[Dict[str, object]]] = {}

    for channel in channels:
        tone = CHANNEL_TONES.get(channel, "informative")
        hooks = [
            f"{topic} wins start here",
            f"{topic} momentum unlocked",
        ]
        channel_posts: List[Dict[str, object]] = []
        for variant in range(variants):
            hook = hooks[variant % len(hooks)][:90]
            body = _build_body(channel, topic, tone, language)
            image_prompt = f"{channel} style visual about {topic}"
            utm_url = None
            if config.FEATURE_UTM_LINKS:
                utm_url = f"https://example.com/{slug}?utm_source={channel}&utm_medium=social&utm_campaign={slug}"
            channel_posts.append(
                {
                    "hook": hook,
                    "body": body,
                    "hashtags": hashtags,
                    "image_prompt": image_prompt,
                    "utm_url": utm_url,
                }
            )
        posts[channel] = channel_posts
        LOGGER.info("Generated %d social variants for %s", len(channel_posts), channel)

    return posts


__all__ = ["generate_social_posts"]
