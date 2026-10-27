"""Deterministic cleaning; statistical imputers are fitted in training pipelines only."""
from pathlib import Path
import hashlib
import re
import numpy as np
import pandas as pd

COLUMNS = ['name', 'city', 'hospital_type', 'beds', 'specialties', 'rating', 'nabh']
ALIASES = {
    'name': ['name', 'hospital', 'hospital_name', 'name_of_hospital'],
    'city': ['city', 'location_city', 'town'],
    'hospital_type': ['hospital_type', 'type', 'ownership', 'type_of_hospital'],
    'beds': ['beds', 'bed_count', 'number_of_beds', 'no_of_beds', 'bed_capacity'],
    'specialties': ['specialties', 'medical_specialties', 'specialities', 'specialty', 'specialization'],
    'rating': ['rating', 'hospital_rating', 'ratings', 'overall_rating'],
    'nabh': ['nabh', 'nabh_status', 'nabh_accreditation', 'nabh_accreditation_status', 'accreditation_status'],
}
NA = {'', 'nan', 'none', 'null', 'n/a', 'na', 'unknown', 'not available', '-'}

def key(value):
    return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')

def label(value):
    if pd.isna(value) or str(value).strip().lower() in NA:
        return None
    return re.sub(r'\s+', ' ', str(value).strip()).title()

def number(value, beds=False):
    if pd.isna(value):
        return np.nan
    raw = str(value).strip().lower().replace(',', '')
    if raw in NA:
        return np.nan
    pattern = r'(-?\d+(?:\.\d+)?)\s*(?:beds?)?' if beds else r'(-?\d+(?:\.\d+)?)\s*(?:/\s*\d+(?:\.\d+)?)?'
    m = re.fullmatch(pattern, raw)
    if not m:
        return np.nan
    n = float(m.group(1))
    if n < 0 or (beds and (n == 0 or n > 100000 or not n.is_integer())):
        return np.nan
    if not beds and '/' in raw:
        denom = float(raw.split('/')[1])
        if denom <= 0 or n > denom:
            return np.nan
    return n

def specialties(value):
    if label(value) is None:
        return []
    return sorted({v for item in re.split(r'[,;|/]', str(value)) if (v := label(item))})

def nabh(value):
    raw = str(value).strip().lower() if not pd.isna(value) else ''
    if raw in {'yes', 'y', 'true', '1', '1.0', 'accredited', 'nabh accredited', 'certified'}:
        return 'Accredited'
    if raw in {'no', 'n', 'false', '0', '0.0', 'not accredited', 'non-accredited', 'not nabh accredited'}:
        return 'Not accredited'
    return 'Unknown'

def load_dataset(path: Path):
    if not path.exists():
        return pd.DataFrame(columns=['id', *COLUMNS, 'group']), {'status': 'missing', 'message': 'Import your hospital CSV to begin.', 'available_columns': [], 'warnings': []}
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if not len(raw):
        raise ValueError('The CSV contains no hospital rows.')
    normalized = {key(c): c for c in raw.columns}
    if len(normalized) != len(raw.columns):
        raise ValueError('Column names collide after normalization; rename them before import.')
    mapping = {}
    for canonical, aliases in ALIASES.items():
        matches = [normalized[a] for a in aliases if a in normalized]
        if len(matches) > 1:
            raise ValueError(f'Multiple columns map to {canonical}; keep one or rename explicitly.')
        if matches:
            mapping[canonical] = matches[0]
    if 'name' not in mapping:
        raise ValueError('A hospital name column is required. See data/README.md for aliases.')
    df = pd.DataFrame({c: raw[mapping[c]] if c in mapping else None for c in COLUMNS}, index=raw.index)
    original_missing = {str(c): int(raw[c].map(lambda v: str(v).strip().lower() in NA).sum()) for c in raw}
    for c in ['name', 'city', 'hospital_type']:
        df[c] = df[c].map(label)
    df['city'] = df.city.replace({'Bangalore': 'Bengaluru', 'Bombay': 'Mumbai', 'Madras': 'Chennai', 'Calcutta': 'Kolkata'})
    df['hospital_type'] = df.hospital_type.replace({'Govt': 'Government', 'Govt.': 'Government', 'Public': 'Government', 'Pvt': 'Private', 'Pvt.': 'Private'})
    df['beds'] = df.beds.map(lambda v: number(v, beds=True))
    df['rating'] = df.rating.map(number)
    scales = {float(m.group(1)) for v in raw[mapping['rating']] if (m := re.search(r'/\s*(\d+(?:\.\d+)?)\s*$', str(v)))} if 'rating' in mapping else set()
    if len(scales) > 1:
        raise ValueError('Mixed rating denominators detected. Standardize the source rating scale before import.')
    if scales and df.rating.max() > next(iter(scales)):
        raise ValueError('Some ratings exceed the explicit rating denominator.')
    df['specialties'] = df.specialties.map(specialties)
    df['nabh'] = df.nabh.map(nabh)
    no_name = int(df.name.isna().sum())
    df = df[df.name.notna()].copy()
    serialized = df.assign(specialties=df.specialties.map(tuple))
    duplicates = int(serialized.duplicated().sum())
    df = df.loc[~serialized.duplicated()].reset_index(drop=True)
    if df.empty:
        raise ValueError('No named hospitals remain after cleaning.')
    df['group'] = df.name.map(lambda n: re.sub(r'[^a-z0-9]', '', n.lower()))
    df['id'] = [hashlib.sha256(str(row).encode()).hexdigest()[:16] for row in df[COLUMNS].astype(str).itertuples(index=False, name=None)]
    repeated = int(df.group.duplicated(keep=False).sum())
    warnings = ['City is excluded from predictive models to reduce geographic memorization. Hospital names and IDs are never predictive features.', 'Organization groups use normalized exact names only; review near-duplicates and chain names manually before relying on validation.']
    if repeated:
        warnings.append(f'{repeated} rows share normalized names; prediction evaluation uses group-aware splitting.')
    missing = {c: int(df[c].map(lambda x: not x if isinstance(x, list) else pd.isna(x) or x == 'Unknown').sum()) for c in COLUMNS}
    outliers = {}
    for c in ['beds', 'rating']:
        vals = df[c].dropna()
        q1, q3 = vals.quantile(.25), vals.quantile(.75)
        outliers[c] = int(((vals < q1 - 1.5*(q3-q1)) | (vals > q3 + 1.5*(q3-q1))).sum()) if len(vals) else 0
    report = {'status': 'loaded', 'fingerprint': hashlib.sha256(path.read_bytes()).hexdigest(), 'source_filename': path.name,
              'source_rows': len(raw), 'rows': len(df), 'source_columns': list(raw.columns), 'column_mapping': mapping,
              'available_columns': list(mapping), 'source_types': {str(c): str(raw[c].dtype) for c in raw},
              'missing_before': original_missing, 'missing_after': missing, 'removed_unnamed': no_name,
              'removed_duplicates': duplicates, 'repeated_name_rows': repeated, 'outlier_counts_iqr': outliers,
              'rating_observed_range': [float(df.rating.min()), float(df.rating.max())] if df.rating.notna().any() else None,
              'rating_explicit_denominator': next(iter(scales)) if scales else None,
              'nabh_source_values': raw[mapping['nabh']].value_counts().to_dict() if 'nabh' in mapping else {},
              'warnings': warnings}
    return df, report

def records(df):
    clean = df.drop(columns=['group'], errors='ignore')
    return clean.astype(object).where(pd.notna(clean), None).to_dict('records')
