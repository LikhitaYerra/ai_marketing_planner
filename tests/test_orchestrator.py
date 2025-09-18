from marketing_pipeline.config import FeatureConfig
from marketing_pipeline.models import RunInput
from marketing_pipeline.orchestrator import run_pipeline


def test_orchestrator_old_shape_with_flags_off():
    config = FeatureConfig(
        FEATURE_SEO_INGEST=False,
        FEATURE_LOCALIZATION_FR=False,
        FEATURE_SOCIAL_AB_TESTING=False,
        FEATURE_UTM_LINKS=False,
        FEATURE_RUN_REPORT=False,
    )
    run_input = RunInput(
        goal="Increase awareness",
        topic="AI Strategy",
        site_url="https://example.com",
    )
    result = run_pipeline(run_input, config)
    assert result.plan["topic"] == "AI Strategy"
    assert result.articles
    assert result.posts
    assert result.seo_summary is None
    assert result.evaluation is None


def test_orchestrator_extended_shape_with_flags_on():
    config = FeatureConfig(
        FEATURE_SEO_INGEST=True,
        FEATURE_LOCALIZATION_FR=True,
        FEATURE_SOCIAL_AB_TESTING=True,
        FEATURE_UTM_LINKS=True,
        FEATURE_RUN_REPORT=True,
    )
    run_input = RunInput(
        goal="Increase awareness",
        topic="AI Strategy",
        site_url="https://example.com",
        languages=["en", "fr"],
        channels=["facebook"],
        seo_report_kind="json",
        seo_report_raw='[{"keyword": "ai marketing", "gap": "voice search"}]',
    )
    result = run_pipeline(run_input, config)
    assert result.articles and len(result.articles) == 2
    assert "seo_summary" in result.to_dict()
    assert result.posts
    assert result.evaluation
    assert result.case_study
    assert result.pipeline_explainer
