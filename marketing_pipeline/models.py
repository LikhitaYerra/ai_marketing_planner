"""Shared datamodels for the marketing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunInput:
    """Input payload for running the pipeline."""

    goal: str
    topic: str
    site_url: Optional[str] = None
    keywords: Optional[List[str]] = None
    publish_date: Optional[str] = None
    languages: Optional[List[str]] = None
    channels: Optional[List[str]] = None
    seo_report_raw: Optional[str] = None
    seo_report_kind: Optional[str] = None
    attach_seo_report: bool = False


@dataclass
class RunResult:
    """Result payload produced by the pipeline."""

    plan: Dict[str, Any]
    articles: Optional[List[Dict[str, Any]]] = None
    posts: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]] = None
    seo_summary: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    case_study: Optional[str] = None
    pipeline_explainer: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the result for external consumers."""

        return {
            "plan": self.plan,
            "articles": self.articles,
            "posts": self.posts,
            "seo_summary": self.seo_summary,
            "evaluation": self.evaluation,
            "case_study": self.case_study,
            "pipeline_explainer": self.pipeline_explainer,
            "metadata": self.metadata,
        }


__all__ = ["RunInput", "RunResult"]
