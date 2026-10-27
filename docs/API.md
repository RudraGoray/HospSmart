# API reference

Base: http://127.0.0.1:8000/api. Interactive OpenAPI: /docs. Machine-readable schema: /openapi.json (a snapshot is included in docs/openapi.json).

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /health | Process, dataset and per-model availability |
| GET | /overview | Dataset inspection, statistics, distributions, top recorded ratings |
| GET | /filters | Recorded city, type, specialty, NABH options; available columns and observed rating range |
| GET | /hospitals | Filtered, sorted, paginated directory |
| GET | /hospitals/compare?ids=id1,id2 | One to three unique hospital profiles |
| GET | /hospitals/{hospital_id} | One source-derived profile |
| POST | /recommendations | Up to five constrained weighted matches and component explanations |
| POST | /predict/rating | Statistical rating estimate; range, features, missing inputs and warnings |
| POST | /predict/nabh | Estimated class and uncalibrated forest probabilities |
| GET | /models/metrics | Current saved run, held-out forest/baseline results, CV and proxy results |
| GET | /models/methodology | Current dataset report, training manifest, preprocessing and limitations |

## Directory

Query parameters: search, city, specialty, hospital_type, nabh, min_rating, min_beds, sort (name/rating/beds), page (default 1), page_size (default 12, maximum 100). Name sort is ascending; rating and capacity descending; missing values last. Name search uses literal case-insensitive text, never a user-provided regex. Unknown directory filter values return an empty result.

Response shape: items (hospital array), total, page, page_size. Hospital fields: id, name, city, hospital_type, beds, specialties (array), rating, nabh. Missing scalar fields are null; missing specialties are []; unknown NABH is Unknown.

## Recommendations

~~~json
{"city":null,"specialties":[],"hospital_type":null,"min_beds":null,"min_rating":null,"nabh":"no_preference","rating_weight":0.5}
~~~

Choose values from the actual filters endpoint. NABH choices: no_preference, not_required, preferred, required. Selected city, every requested specialty, minimum beds/rating and required accreditation are mandatory constraints. Hospital type and preferred accreditation affect ranking only. not_required does not exclude accredited hospitals.

Each result contains a hospital, match_score, components, normalized weights, reasons and ranking_explanation. Scores are preference alignment, never outcome or quality probabilities. Unknown categories return HTTP 422 with guidance. No matching candidates return HTTP 200 with an empty list and a clear explanation. Missing columns cannot satisfy an explicitly requested constraint.

## Prediction

~~~json
{"hospital_type":"a category present in the training schema","beds":200,"specialties":[]}
~~~

For rating only, an optional nabh field accepts Accredited, Not accredited or Unknown. NABH classification rejects that field, rating, city, name and IDs. Send only fields supported by the saved model. Missing feature inputs are imputed using its fitted pipeline. Provide at least one supported characteristic. Unknown training categories/specialties return 422; the UI forms use the saved schema. Bed values outside training range are accepted within the documented plausibility bound, with an extrapolation warning.

Response: prediction, probabilities (classification only), observed_range and explicit_denominator (rating only), provided_features, unavailable_features, warnings. No output clamping. A recorded status in a listing is never replaced by a predicted class.

## Errors

- 422: invalid request, unknown category in a recommendation or prediction, unsupported feature, more than three comparison hospitals.
- 404: hospital or comparison ID absent from the current dataset.
- 503: no usable dataset, insufficient-data model, missing/corrupt artifact, or dataset/model fingerprint mismatch.
- 500: generic unexpected failure message. Paths and server stack traces are not returned.

Errors have a detail field. FastAPI validation details can be an array; the frontend converts this to a useful input-check message. An absent dataset still allows health, empty directory, overview, filters and methodology responses. There are no mutation or public model-training endpoints.
