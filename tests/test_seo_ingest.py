import json

import pytest

from marketing_pipeline.config import FeatureConfig
from marketing_pipeline.seo import seo_ingest


@pytest.mark.parametrize("kind", ["json", "csv"])
def test_seo_ingest_parses_reports(kind):
    config = FeatureConfig(FEATURE_SEO_INGEST=True)
    if kind == "json":
        report = json.dumps(
            [
                {"keyword": "ai marketing", "gap": "voice search", "issue": "slow pages", "note": "focus"},
                {"keyword": "automation"},
            ]
        )
    else:
        report = "keyword,gap,issue,note\nai marketing,voice search,slow pages,focus\nautomation,,,"

    summary = seo_ingest(report, kind, config)
    assert summary is not None
    assert "ai marketing" in summary["top_keywords"]
    if kind == "json":
        assert "voice search" in summary["keyword_gaps"]
