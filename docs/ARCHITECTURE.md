# Architecture

~~~text
hospsmart-ai/
  frontend/
    src/main.tsx          Navigation, comparison state, routes, disclaimer
    src/pages.tsx         Eight screens and hospital detail screen
    src/components.tsx    Shared forms, cards, charts and request lifecycle
    src/api.ts            Fetch client, query encoding and comparison constraints
    src/api.test.ts       Frontend API and state tests
    src/style.css         Responsive navy/teal visual system plus Tailwind
    dist/                 Verified production build
    package-lock.json     Exact frontend dependency tree
  backend/
    app/data.py           Alias mapping, cleaning, inspection, fingerprint
    app/features.py       Training-only column transformer and specialty encoder
    app/training.py       Forests, dummy baselines, splits, CV and artifacts
    app/recommend.py      Cosine similarity, weighted ranking, proxy evaluation
    app/schemas.py        Pydantic request and response schemas
    app/main.py           API factory and cached trusted local model loading
    scripts/              Inspect, import, train and evaluate CLI commands
    tests/                Synthetic-fixture integration tests
    requirements.txt      Compatible dependency ranges
    requirements-lock.txt Exact versions used in verification
  data/raw/               Imported source and archived previous imports
  models/                 Generated pipelines, schemas, evaluation and run manifest
  docs/                   API, ML report, limitations and verification
~~~

## Data flow

CSV -> immutable raw import -> deterministic normalization -> cleaned in-memory table.
The directory, comparison, filters and chart APIs read this table directly. Listings retain missing values.

For prediction training, split eligible rows before fitting learned transformations. ColumnTransformer handles numeric median imputation, categorical one-hot encoding and a training-only multi-hot specialty vocabulary. A Pipeline fits the forest. A separate dummy pipeline uses the same split. Persist the fitted pipeline and schema, measured metrics and test predictions.

At API startup, load the dataset and one metrics manifest. Check the manifest's SHA-256 dataset fingerprint before loading model artifacts. Load each model once. Missing, corrupt or stale artifacts produce visible unavailable states and HTTP 503 on prediction endpoints. Import/train offline, then restart. Training is never triggered by an HTTP request.

The frontend fetches /api using a Vite same-origin proxy to FastAPI. React Router supplies navigation, Recharts displays backend distributions, and localStorage stores only up to three selected hospital IDs. There are no accounts, patient records, external data APIs or required database. Server-side persistence is unnecessary for this scope.

## Runtime boundaries

Pydantic rejects unknown prediction fields and invalid values. The NABH input schema rejects its target. Categories are checked against the fitted training schema. Unknown data remains distinct from a predicted category. API errors do not include filesystem paths or stack traces; server logs retain diagnostic details.

The importer is a local CLI, which avoids introducing an unauthenticated upload/training service. Run this educational app locally. Before public deployment, implement access controls, resource limits and production operational safeguards appropriate to the intended use.
