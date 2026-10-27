# Dataset contract

No dataset was supplied. The template contains headers only. Import real data as documented in README.md. Preprocessing never edits source bytes.

## Accepted aliases

Case, whitespace and punctuation in headers are normalized to lowercase underscores.

| Canonical | Aliases |
| --- | --- |
| name | name, hospital, hospital_name, name_of_hospital |
| city | city, location_city, town |
| hospital_type | hospital_type, type, ownership, type_of_hospital |
| beds | beds, bed_count, number_of_beds, no_of_beds, bed_capacity |
| specialties | specialties, medical_specialties, specialities, specialty, specialization |
| rating | rating, hospital_rating, ratings, overall_rating |
| nabh | nabh, nabh_status, nabh_accreditation, nabh_accreditation_status, accreditation_status |

Unsupported columns are inspected but excluded from ML. Rename unsupported headers in a separate working CSV. Ambiguous mappings are rejected. UTF-8 with an optional BOM is supported.

## Cleaning policies

- Blank, NA, null and similar tokens become missing. Categories are whitespace-normalized and title-cased.
- Bangalore/Bombay/Madras/Calcutta map to Bengaluru/Mumbai/Chennai/Kolkata. Public/Govt and Pvt aliases map to Government and Private. Review and extend the explicit maps in app/data.py for your source.
- Beds accept positive whole numbers, comma separators and trailing beds. Zero, negative values, ranges and counts over 100,000 become missing. The upper limit is an explicit plausibility policy.
- Ratings accept numbers and value/denominator text. Negative and malformed ratings, or explicit values exceeding the denominator, become missing. Multiple explicit denominators stop import. Without a denominator the official scale is unknown; only observed min/max is shown. Implicitly mixed scales require manual review.
- Specialties split at commas, semicolons, pipes and slashes. Quote CSV fields containing commas. No medical synonym inference is applied.
- yes/true/1/accredited/certified map to Accredited; no/false/0/not accredited map to Not accredited. Pending, expired, entry-level and other unrecognized labels map to Unknown. Original source label counts remain visible. Review this policy against the meaning of your dataset. Source categories are not independently verified certification.
- Remove unnamed records and exact normalized duplicates; count removals. Distinct profiles sharing names remain. Normalized exact names form validation groups; recommendations deduplicate name + city. Manually inspect chains and near-duplicates.
- Report IQR outliers without automatically discarding them. Missing clinical facts are not filled in listings. Numeric imputation is confined to model pipelines.

The inspection report includes source filename, original columns and types (read as strings), missing values, mappings, duplicate counts, outliers, source NABH categories, observed rating range and SHA-256 fingerprint.
