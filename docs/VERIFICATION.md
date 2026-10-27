# Verification record

Verified on 27 October 2026 in the provided Windows workspace using Python 3.12.14, Node.js 24.22.1, scikit-learn 1.9.1 and Vite 7.3.7. Exact installed Python and frontend dependencies are captured in requirements-lock.txt and package-lock.json.

## Automated checks

- Backend: 15 tests passed. Coverage includes source preservation, normalization, invalid values, missing columns, duplicates, mixed explicit rating scales, leakage feature restrictions, recommendation constraints/points/rank changes, empty datasets, API routes, comparison limits, unknown categories, input validation, sklearn training/held-out evaluation/CV, saved model loading, real pipeline predictions, stale/corrupt/missing artifacts, single/rare class withholding, class weighting and group separation.
- Frontend: 6 tests passed. Checks include query encoding, comparison limits/removal, explicit missing values, JSON request integration, structured API failures and offline errors.
- TypeScript and Vite production build passed. React and chart bundles are split; a production dist directory is included.
- OpenAPI schema was generated from the FastAPI application into docs/openapi.json.

Every hospital row used by tests is an explicitly labeled synthetic test fixture, generated in temporary directories. Fixture models and their scores are excluded from the deliverable and are never application data. Test success verifies software behavior, not hospital data validity or clinical performance.

## Runtime and browser limit

FastAPI and Vite started and printed their local listening URLs. However, loopback HTTP requests and the browser preview could not reach those server ports in this execution environment. Browser opening failed with a connection timeout. API behavior was verified with FastAPI TestClient (in-process ASGI), including actual saved model prediction calls on synthetic fixtures. Browser-rendered layout, interactions and mobile visual QA remain unverified. Test this on your local machine after extraction using the README commands.

Nonfatal dependency notices: Starlette's test client warns about future migration from httpx to httpx2; pandas warns about future categorical missing-value downcasting behavior. Tests pass with the locked dependency versions. Neither warning is a fabricated application result.

## Required after you add the real CSV

Inspect actual schema, missingness, organization groups, rating scale and class distribution. Train and review real model results. Check the actual records, recommendation explanations and charts in a local browser. Withheld models should remain visibly unavailable. Do not use test-fixture scores in your project report.
