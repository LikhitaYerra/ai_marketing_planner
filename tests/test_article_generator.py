from marketing_pipeline.articles import generate_article
from marketing_pipeline.brief import build_content_brief
from marketing_pipeline.config import FeatureConfig


def test_article_generator_respects_language_and_length():
    brief = build_content_brief({"top_keywords": [f"keyword{i}" for i in range(12)], "keyword_gaps": []})
    config = FeatureConfig(FEATURE_LOCALIZATION_FR=True)
    article_en = generate_article("AI Strategy", brief, "en", config)
    article_fr = generate_article("AI Strategy", brief, "fr", config)

    assert 800 <= len(article_en["content"].split()) <= 1200
    assert 800 <= len(article_fr["content"].split()) <= 1200
    assert "francophone" in article_fr["content"]
    assert article_fr["language"] == "fr"
