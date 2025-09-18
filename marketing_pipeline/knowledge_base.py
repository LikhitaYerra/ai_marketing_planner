"""Lightweight knowledge base helper used for optional SEO storage."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Mapping

LOGGER = logging.getLogger(__name__)

KB_PATH = Path("env") / "kb_store.json"


def _load_store() -> List[Mapping[str, str]]:
    if not KB_PATH.exists():
        return []
    try:
        with KB_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Failed to load KB store: %s", exc)
        return []


def store_entries(entries: Iterable[Mapping[str, str]]) -> None:
    """Persist entries to the lightweight knowledge base."""

    KB_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_store()
    existing.extend(list(entries))
    with KB_PATH.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Stored %d knowledge base entries", len(entries))


__all__ = ["store_entries"]
