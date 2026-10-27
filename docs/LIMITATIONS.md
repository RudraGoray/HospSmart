# Limitations and future improvements

## Current limitations

- No real dataset was supplied. The full local app and training workflow are delivered, but dataset-specific models and measured performance must be produced after import. Software tests do not establish clinical validity.
- Source services, capacity, ratings and NABH labels can be incomplete or stale. Recorded status is not independently verified certification. No external official registry is queried.
- Automatic mappings intentionally handle a small documented set of aliases. Ambiguous specialty names, source categories and implicit rating scales need manual review.
- Organization groups use normalized exact names. Near-duplicates and chains with different branch names can still cross validation boundaries. Inspect and extend grouping for the real dataset before drawing performance conclusions.
- An 80/20 holdout from roughly 463 records has substantial uncertainty. Three-fold CV helps characterize variability but does not remove source bias. No confidence interval or external validation is claimed.
- The minimum sample/class policies prevent obvious invalid training; they do not guarantee reliable model performance. Poor forest performance versus baseline should be reported rather than concealed.
- Forest class probabilities are uncalibrated. Feature importance is impurity-based, can favor some representations and is not causal.
- Recommendation weights are transparent design choices. Broad service portfolios can lower cosine similarity for a narrow query. Mandatory specialty matching can make high Precision@5/NDCG@5 trivial; those proxies do not validate ranking among relevant hospitals or actual patient satisfaction.
- No geolocation/travel-time, cost, insurance, emergency availability, clinician data, clinical outcomes or personal medical information is modeled.
- Hospital comparison selection persists in browser localStorage only. There are no accounts or server-side feedback stores.
- The provided server is intended for local educational use. It has no public upload endpoint or online retraining. Restart after import/training. External deployment requires a same-origin API proxy, SPA fallback and operational configuration.
- The browser-preview route could not be reached in the execution environment. Build, API integration and software behavior were checked; rendered desktop/mobile visual QA remains unverified.

## Future work

1. Obtain a licensed, dated source dataset and record provenance and official accreditation source URLs.
2. Add reviewed specialty synonyms and verified organization/branch groups.
3. Evaluate on external temporal and geographic samples and report bootstrap uncertainty where valid.
4. Investigate probability calibration within training-only validation, followed by a genuinely untouched test set.
5. Add consented user relevance assessments; evaluate ranking among matching hospitals independently of hard constraints.
6. Add accessible screen-reader, keyboard and cross-device browser tests on a reachable local installation.
7. Introduce SQLite-backed saved preferences/feedback only if required, with clear data minimization and access controls.
8. Add measured model-drift monitoring and reviewed artifact/version management if this grows beyond a college demo.
