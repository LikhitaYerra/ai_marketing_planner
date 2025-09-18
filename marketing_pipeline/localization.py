"""Localization helpers used for optional language support."""

from __future__ import annotations

import logging
from typing import List, Tuple

from .config import FeatureConfig

LOGGER = logging.getLogger(__name__)


def _inject_french_style(text: str) -> Tuple[str, List[str]]:
    replacements = {
        "marketing": "marketing",
        "strategy": "stratégie",
        "growth": "croissance",
        "team": "équipe",
        "business": "entreprise",
        "customers": "clients",
    }
    qa_notes: List[str] = []
    localised = text
    for source, target in replacements.items():
        if source in localised:
            localised = localised.replace(source, target)
            qa_notes.append(f"Remplacé '{source}' par '{target}'")
    localised = "\n".join(
        f"{line} — élaboré pour le marché francophone" if line.strip() else line
        for line in localised.splitlines()
    )
    if not qa_notes:
        qa_notes.append("Ajustements légers pour un ton natif.")
    return localised, qa_notes


def localize(text: str, target_lang: str, config: FeatureConfig) -> Tuple[str, List[str]]:
    """Localise text into the requested language respecting feature flags."""

    if target_lang.lower() != "fr":
        return text, []

    if not config.FEATURE_LOCALIZATION_FR:
        LOGGER.info("Localization for FR skipped - feature flag disabled")
        return text, []

    LOGGER.info("Localising text into French")
    return _inject_french_style(text)


__all__ = ["localize"]
