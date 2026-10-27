# Verification - Streamlit edition

Verified on 27 October 2026 with Python 3.12 on Windows.

## Results

- `python -m pytest -q --tb=short`: **14 passed** (final run: 13.68 seconds).
- Python syntax parsed successfully for all 11 application/script/test files.
- `.streamlit/config.toml` parsed successfully.
- A real Streamlit server started in headless production mode.
- `GET /_stcore/health`: **HTTP 200**, body `ok`.
- `GET /`: **HTTP 200**, Streamlit HTML returned.
- `python -m scripts.train --help`: succeeded.

The test environment used the pinned runtime dependency versions in requirements.txt.
The isolated workspace installation needed `--global.developmentMode=false` for
the server smoke test because its package path is outside a normal site-packages
installation. Normal virtualenv and Community Cloud installs use site-packages.

## Covered workflows

1. CSV alias normalization, source-byte preservation, required-name validation,
   rating-scale rejection, file-size and row-count limits.
2. Missing numeric values receive zero recommendation points even when all known
   numeric values are identical.
3. Required city/specialties/beds/accreditation constraints and score components.
4. Real random-forest training, held-out metrics, training-only cross validation,
   rating predictions and normalized classification probabilities.
5. Unavailable models on undersized datasets; invalid prediction input rejection.
6. Model ZIP generation, fingerprint matching and corrupt-model handling.
7. All nine screens in the empty state.
8. All nine screens with name-only hospital records.
9. All nine screens with a fuller dataset; search, add-to-comparison and recommendation submission.
10. Trained rating/NABH forms and metrics display.
11. Upload form submission clears old models, while an invalid CSV retains the
    previous valid dataset. AppTest lacks a file-uploader setter, so this test
    supplies the upload widget's return value and exercises the real submit handler.
12. Training through the interface followed by a rating estimate.
13. Startup from a bundled CSV plus matching trained model artifacts, then prediction.
14. Independent Streamlit sessions do not share uploaded data or models.

Tests use clearly labeled synthetic software fixtures in temporary directories.
No synthetic dataset or generated test models are loaded by the delivered app.

## Limits

The automated tests and HTTP smoke test verify software operation. This package
has not been published to the user's GitHub or Streamlit account. Browser visual
and cross-device accessibility checks were not performed. Training is not clinical
validation and no real-dataset performance is claimed.

The inherited preprocessing emits a non-fatal pandas FutureWarning about object
array downcasting during some prediction tests. The pinned pandas version passes
all checks; this warning does not block deployment.
