from marketing_pipeline.brief import build_content_brief
from marketing_pipeline.config import FeatureConfig
from marketing_pipeline.social import generate_social_posts


def test_social_posts_hashtags_and_utm():
    brief = build_content_brief({"top_keywords": ["growth marketing", "automation"], "keyword_gaps": []})
    config = FeatureConfig(FEATURE_SOCIAL_AB_TESTING=True, FEATURE_UTM_LINKS=True)
    posts = generate_social_posts(
        topic="Growth Playbook",
        slug="growth-playbook",
        brief=brief,
        channels=["facebook", "instagram"],
        language="en",
        config=config,
    )
    for channel_posts in posts.values():
        assert len(channel_posts) == 2
        for post in channel_posts:
            assert 6 <= len(post["hashtags"]) <= 10
            assert post["utm_url"].endswith(f"utm_campaign=growth-playbook")
