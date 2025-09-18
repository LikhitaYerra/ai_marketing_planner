# Changelog

## Unreleased

- Added modular marketing pipeline package with feature-flagged SEO ingest, keyword backfill, content brief, article generator, social distribution, evaluation, and CLI orchestration.
- Introduced automated tests covering SEO ingestion, keyword fallback, article generation, social post creation, and orchestrator flows.
- Defaulted new capabilities **off** (except UTM links + reporting) so existing environments keep their behaviour until flags are flipped on.

## New Environment Variables

- `FEATURE_SEO_INGEST` (default `false`)
- `FEATURE_LOCALIZATION_FR` (default `false`)
- `FEATURE_SOCIAL_AB_TESTING` (default `false`)
- `FEATURE_UTM_LINKS` (default `true`)
- `FEATURE_RUN_REPORT` (default `true`)
- `FEATURE_KB_EMBED_STORE` (default `false`)
- `DEFAULT_LANGS` (default `["en"]`)
- `SOCIAL_CHANNELS` (default `["facebook","instagram","linkedin"]`)
