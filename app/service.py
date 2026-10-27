"""Streamlit service layer. Uploaded files and generated models stay session-local."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import zipfile

import joblib
import pandas as pd
import sklearn
from app.data import load_dataset
from app.features import frame
from app.training import train

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ROWS = 20000
MAX_TRAIN_ROWS = 5000


def read_csv_bytes(content, source_name='hospitals.csv'):
    if not content:
        raise ValueError('Choose a CSV containing at least one named hospital.')
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError('The CSV exceeds the 10 MB upload limit.')
    # Bound row count before the cleaning and reporting work.
    probe = pd.read_csv(BytesIO(content), nrows=MAX_ROWS + 1, dtype=str, encoding='utf-8-sig')
    if len(probe) > MAX_ROWS:
        raise ValueError('Use at most 20,000 source rows per dataset.')
    with TemporaryDirectory(prefix='hospsmart-csv-') as directory:
        path = Path(directory) / 'hospitals.csv'
        path.write_bytes(content)
        df, report = load_dataset(path)
    report['source_filename'] = Path(source_name.replace('\\', '/')).name
    return df, report


def empty_metrics():
    return {'status': 'unavailable', 'run': None,
            'rating': {'status': 'unavailable', 'reason': 'Load a CSV, then train models on the Data & models page.'},
            'nabh': {'status': 'unavailable', 'reason': 'Load a CSV, then train models on the Data & models page.'},
            'recommendation': {'status': 'unavailable'}}


def train_session(content):
    df, report = read_csv_bytes(content)
    if len(df) > MAX_TRAIN_ROWS:
        raise ValueError('In-app training supports at most 5,000 cleaned hospitals. Use the offline training script for larger data.')
    with TemporaryDirectory(prefix='hospsmart-train-') as directory:
        root = Path(directory)
        dataset = root / 'hospitals.csv'
        dataset.write_bytes(content)
        model_dir = root / 'models'
        metrics = train(dataset, model_dir)
        bundles = {target: joblib.load(model_dir / f'{target}.joblib')
                   for target in ['rating', 'nabh'] if metrics[target]['status'] == 'trained'}
        artifacts = {file.name: file.read_bytes() for file in model_dir.iterdir() if file.is_file()}
    return metrics, bundles, artifacts


def load_bundled_models(model_dir, report):
    metrics, bundles = empty_metrics(), {}
    manifest = model_dir / 'metrics.json'
    if not manifest.exists():
        return metrics, bundles
    try:
        candidate = json.loads(manifest.read_text(encoding='utf-8'))
        if candidate.get('run', {}).get('fingerprint') != report.get('fingerprint'):
            reason = 'The bundled models belong to a different CSV. Train models for the current data.'
            for target in ['rating', 'nabh']:
                metrics[target]['reason'] = reason
            metrics['status'] = 'stale'
            return metrics, bundles
        if candidate['run'].get('sklearn') != sklearn.__version__:
            raise ValueError('The scikit-learn version differs from the saved models. Retrain with the deployed requirements.')
        metrics = candidate
        for target in ['rating', 'nabh']:
            if metrics[target]['status'] == 'trained':
                try:
                    bundle = joblib.load(model_dir / f'{target}.joblib')
                    if not bundle.get('schema') or 'pipeline' not in bundle:
                        raise ValueError('Invalid model bundle')
                    bundles[target] = bundle
                except Exception:
                    metrics[target] = {'status': 'unavailable', 'reason': 'A bundled model could not be loaded. Train again.'}
    except Exception as exc:
        metrics = empty_metrics()
        metrics['status'] = 'invalid'
        for target in ['rating', 'nabh']:
            metrics[target]['reason'] = f'Bundled models could not be loaded: {exc}'
    return metrics, bundles


def predict_profile(target, values, bundle, report):
    schema = bundle['schema']
    columns = schema['features']
    provided = [key for key in columns if values.get(key) not in [None, [], 'Unknown']]
    if not provided:
        raise ValueError('Provide at least one supported hospital characteristic.')
    for key, categories in schema['categories'].items():
        if values.get(key) not in [None, 'Unknown'] and values[key] not in categories:
            raise ValueError(f'{key} was not present in the training partition.')
    if any(value not in schema['specialties'] for value in values.get('specialties', [])):
        raise ValueError('A specialty was not present in the training partition.')
    row = pd.DataFrame([{key: values.get(key, [] if key == 'specialties' else None) for key in columns}])
    features = frame(row, columns)
    pipeline = bundle['pipeline']
    estimate = pipeline.predict(features)[0]
    warnings = ['Estimates are statistical outputs, not verified ratings, accreditation decisions, or clinical quality measures.',
                'Missing values use the preprocessing fitted on the training partition.']
    observed = schema.get('beds_observed')
    if observed and values.get('beds') is not None and not observed[0] <= values['beds'] <= observed[1]:
        warnings.append('Bed capacity is outside the training range. The forest extrapolates poorly.')
    probabilities = None
    if target == 'nabh':
        probabilities = dict(zip(pipeline.named_steps['model'].classes_.tolist(), map(float, pipeline.predict_proba(features)[0])))
        warnings.append('Forest probabilities are uncalibrated and do not verify accreditation.')
    return {'prediction': float(estimate) if target == 'rating' else str(estimate),
            'probabilities': probabilities, 'provided_features': provided,
            'missing_features': [key for key in columns if key not in provided],
            'observed_range': report.get('rating_observed_range') if target == 'rating' else None,
            'explicit_denominator': report.get('rating_explicit_denominator') if target == 'rating' else None,
            'warnings': warnings}


def artifacts_zip(artifacts):
    output = BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in artifacts.items():
            archive.writestr(f'models/{name}', content)
    return output.getvalue()


def fingerprint(content):
    return hashlib.sha256(content).hexdigest()
