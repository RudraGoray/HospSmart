# HospSmart AI - Streamlit edition

A standalone Streamlit conversion of HospSmart AI. The Python data cleaning,
recommendation engine and random-forest model design are reused from the supplied
full-stack project. The interface is rebuilt with native Streamlit widgets.
No Node.js, frontend build, API key, FastAPI process or reverse proxy is required.

## Deploy to Streamlit Community Cloud

1. Extract this ZIP. Open the `hospsmart-streamlit` folder.
2. Create a GitHub repository and upload **the contents of that folder**. At the
   repository root you should see `streamlit_app.py`, `requirements.txt`, `app/`,
   `data/`, `models/`, and `.streamlit/config.toml`. Upload extracted files, not the ZIP.
   Ensure the hidden `.streamlit` folder is included.
3. Visit https://share.streamlit.io and sign in with GitHub. Click **Create app**
   and choose the existing repository, branch, and main file `streamlit_app.py`.
4. Under **Advanced settings**, choose **Python 3.12**. Leave secrets empty.
5. Click **Deploy**. After startup, open **Data & models** and load your CSV.
6. Click **Train models on current data** to enable eligible prediction models.
   The other pages work without trained models once a dataset is loaded.

If you uploaded the outer folder itself, use
`hospsmart-streamlit/streamlit_app.py` as the main file path instead. The simplest
layout is the repository-root layout above.

Official deployment documentation:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
Dependency documentation:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

## Run locally

Use Python 3.12 (the version used for verification). From the extracted folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

macOS/Linux equivalent:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, normally http://localhost:8501.

## Dataset and initial state

**No real hospital data or trained models were included in the source ZIP.**
This app deploys and starts with an honest empty state. It includes a blank
CSV template, not invented hospital records or evaluation scores.

Upload a UTF-8 CSV on **Data & models**. Only a hospital name is mandatory.
Recommended headers:

```csv
Hospital Name,City,Hospital Type,Number of Beds,Medical Specialties,Hospital Rating,NABH Accreditation Status
```

Separate specialties with semicolons, e.g. `Cardiology;Neurology`. Quote any CSV
field containing a comma. See `data/README.md` for header aliases, missing-value
rules, normalization and rating-scale handling. The importer preserves original
bytes, reports cleaning decisions, and rejects ambiguous column mappings or
mixed explicit rating denominators.

Limits: CSV size 10 MB, at most 20,000 source rows, and at most 5,000 cleaned
hospitals for in-app training. Larger training runs can use the offline script.

## Keep a dataset and models available after restart

Uploads, comparison selections, estimates and in-app models are specific to the
current Streamlit session. They are not saved to a shared database or committed
to GitHub. A refresh, new visitor, cloud sleep/restart, or redeployment may require
uploading and training again.

For a fixed dataset available to every visitor:

1. Put the exact original CSV at `data/raw/hospitals.csv` in your repository.
2. Optionally train models before deployment, from the project root:

   ```bash
   python -m scripts.train --dataset data/raw/hospitals.csv
   ```

3. Commit the CSV and generated `models/` files along with the application. Use
   the same Python and pinned scikit-learn dependencies during training and deployment.
4. Alternatively, train in the app, download **trained model files**, and extract
   that ZIP into the project root. The files land inside `models/`. Download the
   active original CSV and save it as `data/raw/hospitals.csv`, then commit both.

Model fingerprints must match the exact CSV bytes. An edited CSV or different
scikit-learn version disables bundled models until retraining. Load only trusted
model files produced by this project; the app does not accept model uploads.
Bundled CSV data is visible to all visitors who can access the app.

## Included screens

- Home: entry point and current data/model status.
- Dashboard: city/type/accreditation/rating charts, top recorded ratings, data quality.
- Hospital explorer: search, filters, sorting, paging, record export, profiles.
- Recommendations: mandatory constraints, optional preferences, weighted point breakdown.
- Rating lab: regression estimate using available model-supported features.
- NABH lab: estimated label and uncalibrated forest probabilities.
- Compare hospitals: up to three recorded hospital profiles side by side.
- Data & models: upload, validation, session reset, training, metrics and model downloads.
- Methodology: cleaning, leakage controls, model evaluation and limitations.

## Model policy

Models require at least 20 known-target rows and two distinct target values.
NABH classification also needs at least five examples per class, with all classes
in both train and test. Tasks that cannot satisfy the policy stay unavailable
with an explanation; this is not a deployment error.

Training retains seed 42, an 80/20 holdout, 200 trees, minimum leaf size 2,
group-aware splitting for repeated names, dummy baselines, and valid training-only
three-fold cross validation. Models remain fitted on the training partition.
Single-worker forests limit CPU pressure on shared hosting. There is no claim
that the model is clinically validated or that it outperforms its dummy baseline.

The original recommendation normalization was corrected so missing numeric values
receive zero points even when all known values are identical.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Tests use clearly labeled synthetic records only in temporary directories.
They verify application behavior, not hospital quality or clinical validity.
See `VERIFICATION.md` for the checks performed for this delivery.

## Project layout

```text
streamlit_app.py           Streamlit entrypoint and screens
requirements.txt          Pinned runtime dependencies
.streamlit/config.toml    Theme and 10 MB upload limit
app/                      Cleaning, features, recommendations, schemas, ML, session services
scripts/train.py          Optional offline training
models/                   Optional trusted pre-trained artifacts
data/raw/hospitals.csv    Optional bundled CSV (not supplied)
data/hospitals-template.csv
README.md                 Deployment and operation instructions
VERIFICATION.md           Verification results and limits
tests/                    Temporary-fixture software tests
docs/                     ML report and limitations
```

## Scope

Educational hospital discovery and ML demonstration. Predictions and match scores
are not guarantees of hospital quality, accreditation, treatment success, or
suitability for a medical condition. Verify services and accreditation with
official sources. The original full-stack ZIP is unchanged.
