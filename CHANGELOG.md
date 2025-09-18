# Changelog

## Unreleased

- Added modular marketing pipeline package with feature-flagged SEO ingest, keyword backfill, content brief, article generator, social distribution, evaluation, and CLI orchestration.
- Introduced automated tests covering SEO ingestion, keyword fallback, article generation, social post creation, and orchestrator flows.
- Enabled all marketing pipeline features by default (including FR localisation, SEO ingest/backfill, A/B social, reporting, and knowledge base storage) so no manual flag toggling is required post-merge.

## New Environment Variables

- `FEATURE_SEO_INGEST`
- `FEATURE_LOCALIZATION_FR`
- `FEATURE_SOCIAL_AB_TESTING`
- `FEATURE_UTM_LINKS`
- `FEATURE_RUN_REPORT`
- `FEATURE_KB_EMBED_STORE`
- `DEFAULT_LANGS`
- `SOCIAL_CHANNELS`

All feature flags now default to `true`; set an environment variable to `false` if you need to disable a capability.
