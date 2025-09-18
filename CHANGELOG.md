# Changelog

## Unreleased

- Added modular marketing pipeline package with feature-flagged SEO ingest, keyword backfill, content brief, article generator, social distribution, evaluation, and CLI orchestration.
- Introduced automated tests covering SEO ingestion, keyword fallback, article generation, social post creation, and orchestrator flows.
- Added lightweight knowledge base storage for SEO artifacts when enabled.

## New Environment Variables

- `FEATURE_SEO_INGEST`
- `FEATURE_LOCALIZATION_FR`
- `FEATURE_SOCIAL_AB_TESTING`
- `FEATURE_UTM_LINKS`
- `FEATURE_RUN_REPORT`
- `FEATURE_KB_EMBED_STORE`
- `DEFAULT_LANGS`
- `SOCIAL_CHANNELS`
