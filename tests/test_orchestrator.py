from marketing_pipeline.config import FeatureConfig
from marketing_pipeline.models import RunInput
from marketing_pipeline.orchestrator import run_pipeline


def test_orchestrator_defaults_preserve_single_language_flow():
    config = FeatureConfig()
    run_input = RunInput(
        goal="Increase awareness",
        topic="AI Strategy",
        site_url="https://example.com",
    )
    result = run_pipeline(run_input, config)

    assert result.plan["topic"] == "AI Strategy"
    assert result.articles and len(result.articles) == 1
    assert result.posts and set(result.posts.keys()) == {"en"}
    for channel_posts in result.posts["en"].values():
        assert len(channel_posts) == 1
        assert channel_posts[0]["utm_url"].startswith("https://example.com")
    assert result.seo_summary is None
    assert result.evaluation is not None
    assert result.case_study is not None
    assert result.pipeline_explainer is not None


def test_orchestrator_full_feature_bundle_extends_shape():
    config = FeatureConfig(
        FEATURE_SEO_INGEST=True,
        FEATURE_LOCALIZATION_FR=True,
        FEATURE_SOCIAL_AB_TESTING=True,
        FEATURE_UTM_LINKS=True,
        FEATURE_RUN_REPORT=True,
        FEATURE_KB_EMBED_STORE=False,
        DEFAULT_LANGS=["en", "fr"],
    )
    run_input = RunInput(
        goal="Increase awareness",
        topic="AI Strategy",
        site_url="https://example.com",
        seo_report_raw="""[{\"keyword\":\"AI marketing\",\"gap\":\"voice\"}]""",
        seo_report_kind="json",
    )
    result = run_pipeline(run_input, config)

    assert result.articles and {article["language"] for article in result.articles} == {"en", "fr"}
    assert result.posts and "fr" in result.posts
    for channel_posts in result.posts["fr"].values():
        assert len(channel_posts) == 2
    assert result.seo_summary is not None and "keyword_gaps" in result.seo_summary
    assert result.evaluation is not None and result.case_study is not None
