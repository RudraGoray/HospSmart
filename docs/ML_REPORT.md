# HospSmart AI — college project report

## Objective

Integrate data inspection, two supervised random forests and a content-based recommendation engine into one inspectable hospital discovery application. The task is educational statistical modeling; it does not diagnose, prescribe treatment, certify accreditation or predict outcomes.

## Dataset and integrity

No hospital dataset was supplied with the brief. Actual schema, sample size, missingness, class distribution and all measured model results will be established by the import and training commands. The package contains no illustrative clinical records or made-up scores. Software test fixtures are synthetic, explicitly labeled and isolated in temporary directories.

CSV is preserved byte-for-byte. Deterministic cleaning resolves documented aliases, normalizes categories, parses numbers, splits multi-valued specialties, recognizes known binary NABH statuses and removes exact normalized duplicates. Invalid or ambiguous data becomes missing, or triggers a schema/scale error when interpretation would be unsafe. Outliers are reported for review. Name-based group heuristics help separate repeated organizations, with manual review still necessary for chains and near-duplicates.

## Model 1: rating regression

The target is the recorded rating. A RandomForestRegressor averages predictions from many decision trees fitted on bootstrap samples. Available inputs are bed capacity, hospital type, multi-hot specialty indicators and known NABH category. City is deliberately excluded to reduce geographic memorization in a small dataset. Names and IDs are excluded. No feature is derived from the rating target.

The pipeline fits median bed imputation, categorical missing-value encoding and one-hot categories, and a multi-hot specialty vocabulary on training data only. The comparator is a DummyRegressor predicting the training mean. Both use identical partitions. Report held-out MAE, RMSE and R², plus three-fold training-only cross-validation MAE when feasible. Undefined R² is null. The model is not refitted on full data after evaluation, so the saved model matches the one evaluated.

Predictions are not clamped. A random forest regression estimate averages learned leaf values. The UI distinguishes an explicitly recorded rating denominator from the observed sample range; it does not infer a five-star scale. Inputs outside the training bed range trigger an extrapolation warning. Impurity-based feature importances describe the fitted model and have no causal or clinical interpretation.

## Model 2: NABH classification

A RandomForestClassifier estimates recorded known NABH categories from hospital type, capacity and multi-hot specialties. It excludes NABH itself, rating, city, name and ID. The strict classification request schema also rejects these target/identifier fields. Unknown, expired, pending and unrecognized source statuses remain Unknown and are excluded from supervised training; they are never filled with predictions in listings.

At least 20 known-target rows and five examples per class are required by the project's conservative validation policy. A single class is insufficient. Stratified 80/20 splitting is used for independent records. Repeated normalized names use a group-aware split, which is withheld if train/test does not contain every class. When the training majority/minority ratio is at least 2, balanced class weights are used; otherwise none. Actual class counts, weighting rationale and split sizes are recorded.

Compare with a DummyClassifier using training priors. Report per-class and macro precision, recall and F1, accuracy, confusion matrix, and binary ROC-AUC when defined. Undefined precision/recall/F1 values are null; macro averages omit undefined entries. Class supports are reported. Three-fold CV uses the training partition only and is withheld if group/class coverage cannot support it.

The forest's class probabilities are uncalibrated statistical outputs. They are not verification or certification probabilities. The app labels predicted and recorded categories separately and never replaces a hospital's unknown source status.

## Model 3: personalized recommendations

This is a content-based recommendation component, not a supervised treatment-outcome model. MultiLabelBinarizer creates specialty indicators; cosine similarity compares a query's requested specialties to each candidate's specialty vector. For two nonzero binary vectors, cosine similarity is the count of shared specialties divided by the product of their vector lengths.

First apply mandatory constraints: selected city, ALL selected specialties, minimum beds, minimum recorded rating and required accreditation. Unknown values cannot satisfy numeric or required accreditation constraints. Type preference and preferred accreditation are optional ranking factors. not_required means no accreditation constraint, not a requirement to be unaccredited.

For the remaining candidates, use these raw weights:

| Factor | Raw weight | Representation |
| --- | --- | --- |
| Specialty | 1 - rating_weight, only when specialties selected | Cosine similarity |
| Rating | rating_weight, only when recorded ratings exist | Dataset min-max normalization |
| Capacity | 0.10, only when capacity exists | Dataset min-max normalization |
| Preferred type | 0.10, only when selected | Exact type match indicator |
| Preferred accreditation | 0.10, only for preferred mode | Recorded Accredited indicator |

Normalize active weights to sum to one. A dataset with a constant known numeric value gives known values equal points; missing numeric values receive zero. All-inactive factors produce zero ties resolved deterministically. Required accreditation is filtered, not boosted again. Every factor enters the sum once. Match score is 100 times the weighted sum. Ranking is descending score, then hospital name and ID for ties. Deduplicate normalized name + city and return at most five records. Each card and inspection panel explains exact component points and weights, making rank differences inspectable.

### Proxy recommendation evaluation

Construct up to 100 reproducible lexically sorted recorded city/specialty scenarios. Binary proxy relevance means matching city AND specialty. Compare the weighted engine against a rating-only baseline restricted to the same city, ignoring specialty. Precision@5 divides relevant returned results by five, including short lists. NDCG@5 compares discounted binary relevance to the ideal order of deduplicated city candidates. No click logs, patient choices or user feedback are fabricated.

This design measures constraint compliance and favors an engine that enforces specialty. It does not independently validate ranking within the matching subset, nor real patient preferences. The report and UI explicitly disclose this limitation. Real preference validation needs consented feedback, a separate evaluation design and appropriate governance.

## Reproducibility

Seed 42; 20% holdout; 200 trees; minimum leaf size 2. Record Python and scikit-learn versions, timestamp in UTC, source SHA-256, train/test counts, class distribution, feature schema, CV outcomes, held-out predictions and metrics. Exact-name group splits prevent one group occurring on both sides of a holdout; GroupKFold is used when applicable. No cross-validation uses the final test set. Learned transformations fit per training fold.

The provided scripts combine both model trainers and evaluation into one reproducible workflow. The sklearn Pipeline is serialized with its fitted ColumnTransformer. Only trusted local artifacts should be loaded. The API verifies source fingerprint, caches artifacts and never trains per request.

## Suggested presentation

Show the actual dataset report, then the baseline comparison, confusion matrix and saved run before running prediction demos. Explain why an unavailable task is withheld. Demonstrate hard constraints, optional preferences and weighted rank changes. Discuss dataset bias and evaluation limits candidly.

Primary implementation references: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html and https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html. Dataset-specific conclusions require your actual data and measured run.
