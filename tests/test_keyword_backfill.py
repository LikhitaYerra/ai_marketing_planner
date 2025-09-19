from marketing_pipeline.backfill import keyword_backfill


def test_keyword_backfill_generates_keywords():
    data = keyword_backfill("https://example.com/marketing/insights")
    assert 5 <= len(data["primary_keywords"]) <= 10
    assert 5 <= len(data["secondary_keywords"]) <= 10
    assert data["gaps"]
