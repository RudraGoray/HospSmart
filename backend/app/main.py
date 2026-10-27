from pathlib import Path
import json
import logging
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from app.data import load_dataset, records
from app.features import frame
from app.recommend import recommend
from app.schemas import (Health, Overview, Filters, Directory, Hospital, Comparison, RecommendationRequest,
                         Recommendations, Profile, RatingProfile, Prediction, Metrics, Methodology)

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger('hospsmart')

def create_app(dataset_path=None, model_dir=None):
    dataset_path = Path(dataset_path or os.environ.get('HOSPSMART_DATASET', ROOT / 'data/raw/hospitals.csv'))
    model_dir = Path(model_dir or os.environ.get('HOSPSMART_MODELS', ROOT / 'models'))
    api = FastAPI(title='HospSmart AI', version='1.0.0', description='Educational hospital discovery and ML APIs. Recorded status is never replaced with prediction.')
    try:
        df, report = load_dataset(dataset_path)
    except (ValueError, OSError, pd.errors.ParserError):
        logger.exception('Dataset load failed')
        df, report = load_dataset(Path('__no_hospital_dataset__'))
        report.update(status='invalid', message='Dataset could not be read. Run the inspection script to review the CSV.')
    metrics = {'status': 'unavailable', 'rating': {'status': 'unavailable', 'reason': 'Import data and train models.'}, 'nabh': {'status': 'unavailable', 'reason': 'Import data and train models.'}, 'recommendation': {'status': 'unavailable', 'reason': 'Import data and run training/evaluation.'}, 'run': None}
    bundles = {}
    manifest = model_dir / 'metrics.json'
    if manifest.exists():
        try:
            candidate = json.loads(manifest.read_text(encoding='utf-8'))
            Metrics.model_validate(candidate)
            if candidate.get('run', {}).get('fingerprint') != report.get('fingerprint'):
                metrics['status'] = 'stale'
                for t in ['rating', 'nabh']:
                    metrics[t]['reason'] = 'Dataset fingerprint changed. Retrain and restart the backend.'
            else:
                metrics = candidate
                for t in ['rating', 'nabh']:
                    if metrics[t]['status'] == 'trained':
                        try:
                            bundles[t] = joblib.load(model_dir / f'{t}.joblib')
                            if not bundles[t].get('schema') or 'pipeline' not in bundles[t]:
                                raise ValueError('Invalid local model artifact')
                        except Exception:
                            logger.exception('Local model artifact unavailable')
                            bundles.pop(t, None)
                            metrics[t] = {'status': 'unavailable', 'reason': 'Model artifact could not be loaded. Retrain and restart.'}
        except Exception:
            logger.exception('Metrics manifest could not be read')
            metrics['status'] = 'invalid'
    api.state.dataset = df
    api.state.metrics = metrics

    @api.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        logger.error('Request failed', exc_info=exc)
        return JSONResponse(status_code=500, content={'detail': 'Unable to complete this request. Check the server log and dataset configuration.'})

    def require_data():
        if report['status'] != 'loaded':
            raise HTTPException(503, report.get('message', 'Import a dataset and restart the backend.'))

    def available(column):
        if column not in report['available_columns']:
            raise HTTPException(422, f'The dataset does not contain {column}; remove that constraint.')

    def known(value, column):
        if value is not None:
            available(column)
            options = {s for row in df.specialties for s in row} if column == 'specialties' else set(df[column].dropna())
            if value not in options:
                raise HTTPException(422, f'Unknown {column}: {value}. Choose an available value from the filters.')

    @api.get('/api/health', response_model=Health)
    def health():
        return {'status': 'ok', 'dataset': report['status'], 'models': {t: metrics[t]['status'] for t in ['rating', 'nabh']}}

    @api.get('/api/filters', response_model=Filters)
    def filters():
        return {'cities': sorted(df.city.dropna().unique().tolist()), 'specialties': sorted({s for row in df.specialties for s in row}),
                'hospital_types': sorted(df.hospital_type.dropna().unique().tolist()), 'nabh': ['Accredited', 'Not accredited', 'Unknown'],
                'available_columns': report['available_columns'], 'rating_range': report.get('rating_observed_range')}

    @api.get('/api/overview', response_model=Overview)
    def overview():
        distributions = {}
        for c in ['city', 'hospital_type', 'nabh']:
            distributions[c] = [{'name': str(k), 'count': int(v)} for k,v in df[c].fillna('Unknown').value_counts().items()]
        for c in ['rating', 'beds']:
            values = pd.to_numeric(df[c], errors='coerce').dropna()
            if not len(values):
                distributions[c] = []
            elif values.nunique() <= 10:
                distributions[c] = [{'name': str(k), 'count': int(v)} for k,v in values.value_counts().sort_index().items()]
            else:
                counts, bins = np.histogram(values, bins=min(10, values.nunique()))
                distributions[c] = [{'name': f'{bins[i]:.1f}–{bins[i+1]:.1f}', 'count': int(v)} for i,v in enumerate(counts)]
        return {'dataset': report, 'total': len(df), 'cities': int(df.city.nunique()), 'specialties': len({s for row in df.specialties for s in row}),
                'distributions': distributions, 'top_rated': records(df[df.rating.notna()].sort_values(['rating', 'name'], ascending=[False, True]).head(5))}

    @api.get('/api/hospitals', response_model=Directory)
    def hospitals(search: str = Query('', max_length=200), city: str | None = None, specialty: str | None = None,
                  hospital_type: str | None = None, nabh: str | None = None, min_rating: float | None = Query(None, ge=0),
                  min_beds: float | None = Query(None, ge=1, le=100000), sort: str = Query('name', pattern='^(name|rating|beds)$'),
                  page: int = Query(1, ge=1), page_size: int = Query(12, ge=1, le=100)):
        subset = df.copy()
        if search:
            subset = subset[subset.name.str.contains(search, case=False, regex=False, na=False)]
        for col, value in [('city',city), ('hospital_type',hospital_type), ('nabh',nabh)]:
            if value:
                subset = subset[subset[col] == value]
        if specialty:
            subset = subset[subset.specialties.map(lambda s: specialty in s)]
        if min_rating is not None:
            subset = subset[subset.rating >= min_rating]
        if min_beds is not None:
            subset = subset[subset.beds >= min_beds]
        subset = subset.sort_values([sort, 'id'], ascending=[sort == 'name', True], na_position='last')
        total = len(subset)
        return {'items': records(subset.iloc[(page-1)*page_size:page*page_size]), 'total': total, 'page': page, 'page_size': page_size}

    # Static route precedes the hospital-ID route.
    @api.get('/api/hospitals/compare', response_model=Comparison)
    def compare(ids: str = Query(..., max_length=200)):
        selected = list(dict.fromkeys(ids.split(',')))
        if not 1 <= len(selected) <= 3:
            raise HTTPException(422, 'Choose one to three hospitals.')
        if any(v not in set(df.id) for v in selected):
            raise HTTPException(404, 'A selected hospital is not in the current dataset. Clear the comparison selection.')
        return {'hospitals': [records(df[df.id == i])[0] for i in selected]}

    @api.get('/api/hospitals/{hospital_id}', response_model=Hospital)
    def hospital(hospital_id: str):
        subset = df[df.id == hospital_id]
        if subset.empty:
            raise HTTPException(404, 'Hospital not found in the current dataset.')
        return records(subset)[0]

    @api.post('/api/recommendations', response_model=Recommendations)
    def recommendations(body: RecommendationRequest):
        require_data()
        known(body.city, 'city')
        known(body.hospital_type, 'hospital_type')
        for s in body.specialties:
            known(s, 'specialties')
        for column, enabled in [('beds',body.min_beds is not None), ('rating',body.min_rating is not None), ('nabh',body.nabh in ['required','preferred'])]:
            if enabled:
                available(column)
        result, message = recommend(df, body)
        return {'results': result, 'message': message}

    def predict(target, body):
        require_data()
        if target not in bundles:
            raise HTTPException(503, metrics[target].get('reason', 'Model unavailable. Train and restart the backend.'))
        bundle = bundles[target]
        schema = bundle['schema']
        cols = schema['features']
        values = body.model_dump()
        explicit = body.model_fields_set
        for c in explicit:
            if c not in cols and values[c] not in [None, [], 'Unknown']:
                raise HTTPException(422, f'{c} is not supported by this trained model.')
        for c, categories in schema['categories'].items():
            if values.get(c) not in [None, 'Unknown'] and values[c] not in categories:
                raise HTTPException(422, f'{c} category was not present in training. See model feature schema.')
        if any(s not in schema['specialties'] for s in values.get('specialties', [])):
            raise HTTPException(422, 'A specialty was not present in the model training partition. See model feature schema.')
        provided = [c for c in cols if values.get(c) not in [None, [], 'Unknown']]
        if not provided:
            raise HTTPException(422, 'Provide at least one supported, known hospital characteristic.')
        row = pd.DataFrame([{c: values.get(c, [] if c == 'specialties' else None) for c in cols}])
        X = frame(row, cols)
        prediction = bundle['pipeline'].predict(X)[0]
        warnings = ['This statistical estimate is not an official rating, certification, clinical safety measure, or guarantee.', 'Profiles unlike the training data can produce unreliable estimates. Missing features are imputed using the fitted training pipeline.']
        beds_range = schema.get('beds_observed')
        if beds_range and body.beds is not None and not beds_range[0] <= body.beds <= beds_range[1]:
            warnings.append('Bed capacity is outside the training range; the model extrapolates poorly.')
        probs = None
        if target == 'nabh':
            warnings.append('Forest probabilities are uncalibrated estimates, not verified accreditation probabilities.')
            probs = dict(zip(bundle['pipeline'].named_steps['model'].classes_.tolist(), map(float, bundle['pipeline'].predict_proba(X)[0])))
        return {'prediction': float(prediction) if target == 'rating' else str(prediction), 'probabilities': probs,
                'observed_range': report.get('rating_observed_range') if target == 'rating' else None,
                'explicit_denominator': report.get('rating_explicit_denominator') if target == 'rating' else None,
                'provided_features': provided, 'unavailable_features': [c for c in cols if c not in provided], 'warnings': warnings}

    @api.post('/api/predict/rating', response_model=Prediction)
    def rating(body: RatingProfile):
        return predict('rating', body)

    @api.post('/api/predict/nabh', response_model=Prediction)
    def accreditation(body: Profile):
        return predict('nabh', body)

    @api.get('/api/models/metrics', response_model=Metrics)
    def model_metrics():
        return metrics

    @api.get('/api/models/methodology', response_model=Methodology)
    def methodology():
        return {'dataset': report, 'training': metrics,
                'preprocessing': ['Read raw UTF-8 CSV as strings; preserve original bytes.', 'Resolve documented aliases; reject ambiguous column mappings.', 'Normalize whitespace, city aliases, type aliases and binary NABH representations. Unknown categories remain unknown.', 'Parse numeric values; reject ambiguous ranges, negative ratings and impossible bed values; split specialty lists.', 'Remove exact normalized duplicates and unnamed rows. Flag IQR outliers without automatically deleting plausible values.', 'Fit numeric median, categorical unknown encoding and specialty vocabulary inside each training fold only.'],
                'limitations': ['No diagnosis, treatment, or outcome prediction.', 'Recorded accreditation is supplied source data and has not been independently verified.', 'Observed rating minimum/maximum is not necessarily the official rating scale. Mixed implicit scales require manual source review.', 'Small samples, class imbalance and source bias can make validation unstable.', 'Group heuristic does not resolve every chain or near-duplicate. Review before training.', 'Random forest probabilities are uncalibrated; feature importance is not causal.', 'Recommendation proxy metrics reward constraint compliance and do not validate real patient preferences.', 'Models are loaded once on startup. Restart after importing data or training. Only load locally produced, trusted joblib artifacts.']}
    return api

app = create_app()
