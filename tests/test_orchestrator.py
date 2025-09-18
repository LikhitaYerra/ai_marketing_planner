from marketing_pipeline.config import FeatureConfig
from marketing_pipeline.models import RunInput
from marketing_pipeline.orchestrator import run_pipeline


def test_orchestrator_defaults_run_full_pipeline():
    config = FeatureConfig()
    run_input = RunInput(
        goal="Increase awareness",
        topic="AI Strategy",
        site_url="https://example.com",
    )
    result = run_pipeline(run_input, config)
    assert result.plan["topic"] == "AI Strategy"
    assert result.articles and len(result.articles) == 2
    assert result.posts and "en" in result.posts and "fr" in result.posts
    assert result.seo_summary is not None
    assert result.evaluation is not None
    assert result.case_study is not None
    assert result.pipeline_explainer is not None


def test_orchestrator_can_disable_features():
    config = FeatureConfig(
        FEATURE_SEO_INGEST=False,
        FEATURE_LOCALIZATION_FR=False,
        FEATURE_SOCIAL_AB_TESTING=False,
        FEATURE_UTM_LINKS=False,
        FEATURE_RUN_REPORT=False,
        FEATURE_KB_EMBED_STORE=False,
        DEFAULT_LANGS=["en"],
    )
    run_input = RunInput(
        goal="Increase awareness",
        topic="AI Strategy",
        site_url="https://example.com",
    )
    result = run_pipeline(run_input, config)
    assert result.plan["topic"] == "AI Strategy"
    assert result.articles and len(result.articles) == 1
    assert result.posts and "en" in result.posts
    assert result.seo_summary is None
    assert result.evaluation is None
