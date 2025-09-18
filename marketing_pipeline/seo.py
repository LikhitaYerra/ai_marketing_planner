"""SEO ingestion and enrichment utilities."""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Dict, Iterable, List, Optional

from .config import FeatureConfig
from .knowledge_base import store_entries

LOGGER = logging.getLogger(__name__)


def _normalise_keyword(keyword: str) -> str:
    return " ".join(keyword.strip().split()).lower()


def _limit(values: Iterable[str], maximum: int) -> List[str]:
    result: List[str] = []
    for value in values:
        normalised = _normalise_keyword(value)
        if normalised and normalised not in result:
            result.append(normalised)
        if len(result) >= maximum:
            break
    return result


def seo_ingest(
    report_raw: Optional[str],
    report_kind: Optional[str],
    config: FeatureConfig,
) -> Optional[Dict[str, List[str]]]:
    """Parse the SEO report into lightweight insights.

    Returns None when no report is supplied or when ingestion is disabled.
    """

    if not config.FEATURE_SEO_INGEST or not report_raw:
        LOGGER.info("SEO ingest skipped - flag disabled or no data provided")
        return None

    kind = (report_kind or "").lower()
    try:
        if kind == "json":
            payload = json.loads(report_raw)
            rows = payload if isinstance(payload, list) else payload.get("rows", [])
            parsed_rows = [row for row in rows if isinstance(row, dict)]
        elif kind == "csv":
            reader = csv.DictReader(io.StringIO(report_raw))
            parsed_rows = [row for row in reader]
        else:
            LOGGER.warning("Unknown SEO report kind '%s', skipping", report_kind)
            return None
    except (json.JSONDecodeError, csv.Error) as exc:
        LOGGER.warning("Failed to parse SEO report: %s", exc)
        return None

    keywords: List[str] = []
    gaps: List[str] = []
    issues: List[str] = []
    notes: List[str] = []

    for row in parsed_rows:
        if not isinstance(row, dict):
            continue
        keyword = row.get("keyword") or row.get("search_term") or row.get("term")
        gap = row.get("gap") or row.get("opportunity")
        issue = row.get("issue") or row.get("technical_issue")
        note = row.get("note") or row.get("notes")
        if keyword:
            keywords.append(str(keyword))
        if gap:
            gaps.append(str(gap))
        if issue:
            issues.append(str(issue))
        if note:
            notes.append(str(note))

    summary = {
        "top_keywords": _limit(keywords, 25),
        "keyword_gaps": _limit(gaps, 20),
        "technical_issues": _limit(issues, 20),
        "notes": notes[:10],
    }

    LOGGER.info("SEO ingest produced %d keywords and %d gaps", len(summary["top_keywords"]), len(summary["keyword_gaps"]))

    if config.FEATURE_KB_EMBED_STORE:
        entries = []
        for keyword in summary["top_keywords"]:
            entries.append({"text": keyword, "tag": "keyword"})
        for gap in summary["keyword_gaps"]:
            entries.append({"text": gap, "tag": "gap"})
        if entries:
            store_entries(entries)

    return summary


__all__ = ["seo_ingest"]
