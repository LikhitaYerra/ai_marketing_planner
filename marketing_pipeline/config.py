"""Configuration utilities for the marketing pipeline."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

LOGGER = logging.getLogger(__name__)


def _strtobool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    LOGGER.warning("Unrecognised boolean value '%s', falling back to default %s", value, default)
    return default


def _parse_list(value: Optional[str], default: Iterable[str]) -> List[str]:
    if value is None:
        return list(default)
    value = value.strip()
    if not value:
        return list(default)
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class FeatureConfig:
    """Holds feature flags and runtime defaults for the pipeline."""

    FEATURE_SEO_INGEST: bool = False
    FEATURE_LOCALIZATION_FR: bool = False
    FEATURE_SOCIAL_AB_TESTING: bool = False
    FEATURE_UTM_LINKS: bool = True
    FEATURE_RUN_REPORT: bool = True
    FEATURE_KB_EMBED_STORE: bool = False
    DEFAULT_LANGS: List[str] = field(default_factory=lambda: ["en"])
    SOCIAL_CHANNELS: List[str] = field(default_factory=lambda: ["facebook", "instagram", "linkedin"])

    def snapshot(self) -> dict:
        """Return a serialisable snapshot of the current flag state."""

        return {
            "FEATURE_SEO_INGEST": self.FEATURE_SEO_INGEST,
            "FEATURE_LOCALIZATION_FR": self.FEATURE_LOCALIZATION_FR,
            "FEATURE_SOCIAL_AB_TESTING": self.FEATURE_SOCIAL_AB_TESTING,
            "FEATURE_UTM_LINKS": self.FEATURE_UTM_LINKS,
            "FEATURE_RUN_REPORT": self.FEATURE_RUN_REPORT,
            "FEATURE_KB_EMBED_STORE": self.FEATURE_KB_EMBED_STORE,
            "DEFAULT_LANGS": list(self.DEFAULT_LANGS),
            "SOCIAL_CHANNELS": list(self.SOCIAL_CHANNELS),
        }


def load_config(overrides: Optional[dict] = None) -> FeatureConfig:
    """Load configuration from environment variables with optional overrides."""

    config = FeatureConfig(
        FEATURE_SEO_INGEST=_strtobool(os.getenv("FEATURE_SEO_INGEST"), False),
        FEATURE_LOCALIZATION_FR=_strtobool(os.getenv("FEATURE_LOCALIZATION_FR"), False),
        FEATURE_SOCIAL_AB_TESTING=_strtobool(os.getenv("FEATURE_SOCIAL_AB_TESTING"), False),
        FEATURE_UTM_LINKS=_strtobool(os.getenv("FEATURE_UTM_LINKS"), True),
        FEATURE_RUN_REPORT=_strtobool(os.getenv("FEATURE_RUN_REPORT"), True),
        FEATURE_KB_EMBED_STORE=_strtobool(os.getenv("FEATURE_KB_EMBED_STORE"), False),
        DEFAULT_LANGS=_parse_list(os.getenv("DEFAULT_LANGS"), ["en"]),
        SOCIAL_CHANNELS=_parse_list(
            os.getenv("SOCIAL_CHANNELS"),
            ["facebook", "instagram", "linkedin"],
        ),
    )

    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise AttributeError(f"Unknown config option: {key}")

    LOGGER.debug("Loaded feature config: %s", config.snapshot())
    return config


__all__ = ["FeatureConfig", "load_config"]
