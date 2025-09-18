"""Pipeline orchestration module."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from .articles import generate_article
from .backfill import keyword_backfill
from .brief import build_content_brief
from .config import FeatureConfig
from .evaluation import evaluate_run
from .models import RunInput, RunResult
from .seo import seo_ingest
from .social import generate_social_posts

LOGGER = logging.getLogger(__name__)


class StepTimer:
    """Context manager to log step durations."""

    def __init__(self, name: str):
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        duration = time.perf_counter() - self.start
        LOGGER.info("Step '%s' completed in %.3fs", self.name, duration)


_DEFAULT_PLAN = {
    "status": "scheduled",
    "milestones": [
        "Collect insights",
        "Draft article",
        "Review social posts",
    ],
}


def _build_plan(run_input: RunInput) -> Dict[str, object]:
    plan = dict(_DEFAULT_PLAN)
    plan.update({
        "goal": run_input.goal,
        "topic": run_input.topic,
    })
    return plan


def run_pipeline(run_input: RunInput, config: Optional[FeatureConfig] = None) -> RunResult:
    """Execute the marketing pipeline respecting feature flags."""

    config = config or FeatureConfig()
    LOGGER.info("Starting pipeline with feature flags: %s", config.snapshot())

    with StepTimer("plan"):
        plan = _build_plan(run_input)

    keyword_data = None
    if run_input.seo_report_raw and config.FEATURE_SEO_INGEST:
        with StepTimer("seo_ingest"):
            keyword_data = seo_ingest(run_input.seo_report_raw, run_input.seo_report_kind, config)
    if not keyword_data and run_input.site_url:
        with StepTimer("keyword_backfill"):
            keyword_data = keyword_backfill(run_input.site_url)

    with StepTimer("content_brief"):
        brief = build_content_brief(keyword_data)

    languages = run_input.languages or config.DEFAULT_LANGS
    channels = run_input.channels or config.SOCIAL_CHANNELS

    articles: List[Dict[str, object]] = []
    posts: Dict[str, List[Dict[str, object]]] = {}
    if brief:
        with StepTimer("article_generation"):
            for language in languages:
                article = generate_article(run_input.topic, brief, language, config)
                articles.append(article)
        with StepTimer("social_posts"):
            for language in languages:
                posts[language] = generate_social_posts(
                    run_input.topic,
                    articles[0]["slug"] if articles else run_input.topic,
                    brief,
                    channels,
                    language,
                    config,
                )
    else:
        LOGGER.info("No brief available; skipping article and social generation")

    seo_summary = keyword_data if config.FEATURE_SEO_INGEST else None

    evaluation = None
    case_study = None
    pipeline_explainer: Optional[List[str]] = None
    if config.FEATURE_RUN_REPORT and brief:
        with StepTimer("evaluation"):
            evaluation = evaluate_run({"articles": articles, "posts": posts}, brief, config)
        case_study = (
            "This run transformed strategic goals into assets: articles, social activation, and insights ready for stakeholders."
        )
        pipeline_explainer = [
            "SEO ingest: parsed keyword opportunities.",
            "Brief builder: aligned tone and goals.",
            "Article generator: produced publication-ready drafts.",
            "Social suite: created channel-aware hooks.",
            "Evaluation: captured impact and next actions.",
        ]

    metadata = {
        "languages": languages,
        "channels": channels,
        "attach_seo_report": run_input.attach_seo_report,
    }

    return RunResult(
        plan=plan,
        articles=articles or None,
        posts=posts or None,
        seo_summary=seo_summary,
        evaluation=evaluation,
        case_study=case_study,
        pipeline_explainer=pipeline_explainer,
        metadata=metadata,
    )


__all__ = ["run_pipeline"]
