"""Command line interface for the marketing pipeline."""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .models import RunInput
from .orchestrator import run_pipeline


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Run the marketing pipeline")
    parser.add_argument("goal", help="Business goal for the run")
    parser.add_argument("topic", help="Primary topic")
    parser.add_argument("--site-url", dest="site_url")
    parser.add_argument("--keywords")
    parser.add_argument("--publish-date")
    parser.add_argument("--languages", help="Comma separated languages")
    parser.add_argument("--channels", help="Comma separated channels")
    parser.add_argument("--seo-report-path")
    parser.add_argument("--seo-report-kind", choices=["json", "csv"])
    parser.add_argument("--attach-seo-report", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv or sys.argv[1:])
    config = load_config()
    seo_report_raw = None
    if args.seo_report_path:
        with open(args.seo_report_path, "r", encoding="utf-8") as handle:
            seo_report_raw = handle.read()
    run_input = RunInput(
        goal=args.goal,
        topic=args.topic,
        site_url=args.site_url,
        keywords=[item.strip() for item in args.keywords.split(",")] if args.keywords else None,
        publish_date=args.publish_date,
        languages=[item.strip() for item in args.languages.split(",") if item.strip()] if args.languages else None,
        channels=[item.strip() for item in args.channels.split(",") if item.strip()] if args.channels else None,
        seo_report_raw=seo_report_raw,
        seo_report_kind=args.seo_report_kind,
        attach_seo_report=args.attach_seo_report,
    )
    result = run_pipeline(run_input, config)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
