# Limitations and future improvements

## Current limitations

- No real dataset was supplied. The full local app and training workflow are delivered, but dataset-specific models and measured performance must be produced after import. Software tests do not establish clinical validity.
- Source services, capacity, ratings and NABH labels can be incomplete or stale. Recorded status is not independently verified certification. No external official registry is queried.
- Automatic mappings intentionally handle a small documented set of aliases. Ambiguous specialty names, source categories and implicit rating scales need manual review.
- Organization groups use normalized exact names. Near-duplicates and chains with different branch names can still cross validation boundaries. Inspect and extend grouping for the real dataset before drawing performance conclusions.
- An 80/20 holdout from a small dataset has substantial uncertainty. Three-fold CV helps characterize variability but does not remove source bias. No confidence interval or external validation is claimed.
- The minimum sample/class policies prevent obvious invalid training; they do not guarantee reliable model performance. Poor forest performance versus baseline should be reported rather than concealed.
- Forest class probabilities are uncalibrated. Feature importance is impurity-based, can favor some representations and is not causal.
- Recommendation weights are transparent design choices. Broad service portfolios can lower cosine similarity for a narrow query. Mandatory specialty matching can make high Precision@5/NDCG@5 trivial; those proxies do not validate ranking among relevant hospitals or actual patient satisfaction.
- No geolocation/travel-time, cost, insurance, emergency availability, clinician data, clinical outcomes or personal medical information is modeled.
- Uploads, comparison choices and in-app models persist only within the current Streamlit session. There are no accounts or durable feedback stores.
- This Streamlit edition supports CSV uploads and session-local training. Uploaded datasets and trained models are not durable across session loss or server restart. A bundled CSV and matching trusted model artifacts are needed for startup availability to every visitor. The app has no user account system.
- Streamlit interaction tests verify empty, minimal, loaded and trained states. They do not establish clinical validity, cross-device accessibility, or a successful deployment to the user's Streamlit account.

## Future work

1. Obtain a licensed, dated source dataset and record provenance and official accreditation source URLs.
2. Add reviewed specialty synonyms and verified organization/branch groups.
3. Evaluate on external temporal and geographic samples and report bootstrap uncertainty where valid.
4. Investigate probability calibration within training-only validation, followed by a genuinely untouched test set.
5. Add consented user relevance assessments; evaluate ranking among matching hospitals independently of hard constraints.
6. Add accessible screen-reader, keyboard and cross-device browser tests on a reachable local installation.
7. Introduce SQLite-backed saved preferences/feedback only if required, with clear data minimization and access controls.
8. Add measured model-drift monitoring and reviewed artifact/version management if this grows beyond a college demo.
