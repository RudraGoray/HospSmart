# HospSmart AI

A local, full-stack educational hospital intelligence application. React + Vite + TypeScript + Tailwind + Recharts + Lucide + React Router frontend; Python + FastAPI + pandas + scikit-learn backend. No API keys or paid services.

## Delivery status — read first

**No hospital dataset was attached to the supplied specification.** This package deliberately contains no hospital records, trained production models, or invented evaluation scores. The app starts with clear empty states. Import your CSV and run training to enable records, recommendations and supported prediction models. Synthetic records exist only in automated tests and are never loaded by the application.

All eight pages are implemented: landing, dashboard, explorer/detail, recommendations, rating lab, NABH lab, comparison and methodology. Models are withheld when the dataset fails validation requirements. This state is visible in the app.

## Requirements

- Python 3.11 or 3.12 (verified with 3.12).
- Node.js 22.12+ and npm (verified with Node 24). See https://vite.dev/guide/ for Vite's runtime requirements.
- Two terminal windows: backend on 8000, frontend on 5173.

Extract the ZIP before running commands. Start from the extracted hospsmart-ai folder.

## Windows PowerShell setup

First terminal:

~~~powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
~~~

Inspect and import your own CSV, replacing the example path:

~~~powershell
.\.venv\Scripts\python.exe -m scripts.inspect_dataset "C:\path\to\hospitals.csv"
.\.venv\Scripts\python.exe -m scripts.import_dataset "C:\path\to\hospitals.csv"
.\.venv\Scripts\python.exe -m scripts.train
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
~~~

You can start the backend without data to view the empty-state interface. Importing and training are local terminal operations. Restart the backend after either operation; dataset and models are loaded once per process.

Second terminal, from the extracted project root:

~~~powershell
cd frontend
npm ci
npm run dev
~~~

Open **http://localhost:5173** (or Vite's printed Local URL). API documentation: **http://127.0.0.1:8000/docs**. Health: **http://127.0.0.1:8000/api/health**.

To replace data, run the importer with --replace. It archives the previous raw CSV before replacement:

~~~powershell
.\.venv\Scripts\python.exe -m scripts.import_dataset "C:\path\to\new-hospitals.csv" --replace
.\.venv\Scripts\python.exe -m scripts.train
~~~

The original input file is never edited. Archived imports live in data/raw/archive-<SHA256>.csv. Dataset fingerprint mismatches disable stale models until retraining and restart.

## macOS / Linux setup

~~~bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m scripts.inspect_dataset /path/to/hospitals.csv
python -m scripts.import_dataset /path/to/hospitals.csv
python -m scripts.train
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
~~~

In a second terminal: cd frontend, then npm ci and npm run dev.

## CSV contract

Only a hospital name column is mandatory to import. Optional features remain unavailable when absent. Preferred headers:

~~~csv
Hospital Name,City,Hospital Type,Number of Beds,Medical Specialties,Hospital Rating,NABH Accreditation Status
~~~

See data/README.md for aliases, cleaning and rating-scale caveats. data/hospitals-template.csv has headers only; it is not a dataset. Missing data remains visible; unknown accreditation is never replaced by a prediction.

## Train and evaluate

From backend with the virtual environment active:

~~~bash
python -m scripts.inspect_dataset /path/to/hospitals.csv
python -m scripts.import_dataset /path/to/hospitals.csv
python -m scripts.train
python -m scripts.evaluate_recommendations
~~~

Without activation on Windows, replace python with .\.venv\Scripts\python.exe. Training executes both prediction tasks and recommendation proxy evaluation. Alternate paths are supported with --dataset and --models. To serve alternate paths, set optional process variables HOSPSMART_DATASET and HOSPSMART_MODELS before starting FastAPI. No environment file is needed for default setup.

Generated in models/ when training succeeds:

- rating.joblib and nabh.joblib: fitted preprocessing, forest and training schema. A model file is absent when validation withholds that task.
- metrics.json: measured test metrics, dummy baselines, training-only CV, feature importance, proxy metrics and reproducibility configuration.
- dataset_report.json: cleaning and inspection report.
- feature_schemas.json: exact supported features and training categories.
- rating_evaluation.json and nabh_evaluation.json: held-out IDs, actual values, predictions and baseline outputs.

Models use seed 42, an 80/20 split, 200 trees and minimum leaf size 2. Imputation, encoding and specialty vocabulary fit within training folds only. City and identifiers are excluded. Repeated normalized names trigger group-aware splitting. Saved models remain fitted on the training split so test metrics describe the model actually served. New training runs replace old artifacts; copy models/ to retain a historical run.

Minimum policy: 20 known-target rows and two distinct target values. Classification additionally needs five known examples per class and every class in train and test. Three-fold training-only CV runs when valid. Unknown NABH labels are excluded from classification. Review docs/ML_REPORT.md before interpreting results.

## Tests and production build

From backend:

~~~powershell
.\.venv\Scripts\python.exe -m pytest
~~~

Or pytest with the virtual environment active. From frontend:

~~~bash
npm test
npm run build
npm run preview
~~~

Preview serves the production frontend and proxies /api to local FastAPI. A verified frontend/dist build is included for convenience. Source is the editable deliverable. For external hosting, serve frontend and reverse-proxy /api to FastAPI on the same origin; configure SPA fallback to index.html. This local project is not configured as a public multi-user service.

Tests generate clearly labeled synthetic data in temporary directories to verify software behavior, including actual sklearn training and prediction APIs. These checks are not real-dataset or clinical validation. See docs/VERIFICATION.md.

## Demonstration sequence

1. Inspect and import your actual dataset. Review missing values and source categories.
2. Train and review withheld models and measured evaluations.
3. Open Overview for charts and data quality; filter Explorer and view profiles.
4. Add up to three hospitals and open Compare.
5. Run Recommendations and inspect required constraints, component points and rank order.
6. Try available prediction labs and compare each forest with its dummy baseline.
7. Open Models & Methodology for provenance, CV, leakage controls and limitations.

## Documentation

- docs/ARCHITECTURE.md: structure and request flow.
- docs/API.md: endpoints, validation, errors and examples.
- docs/ML_REPORT.md: college-report explanations.
- docs/LIMITATIONS.md: limitations and future improvements.
- docs/VERIFICATION.md: actual software checks performed.

## Medical disclaimer

HospSmart AI is an educational hospital discovery and machine learning demonstration tool. Predictions and match scores are estimates based on the available dataset and are not guarantees of hospital quality, accreditation, clinical outcomes, or suitability for a particular medical condition. Verify current hospital services and accreditation through official sources before making healthcare decisions.
