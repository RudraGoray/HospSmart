"""Reproducible held-out evaluation. Saved models remain fitted on the training split."""
from pathlib import Path
from datetime import datetime, timezone
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, StratifiedKFold, GroupKFold, cross_val_score
from app.data import load_dataset
from app.features import feature_columns, pipeline, frame
from app.recommend import evaluate_proxy

SEED = 42

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value

def finite(value):
    return float(value) if np.isfinite(value) else None

def train_one(df, target, model_dir):
    data = df[df[target].notna() & (df[target] != 'Unknown')].copy()
    counts = data[target].value_counts().to_dict() if target == 'nabh' else {}
    info = {'status': 'unavailable', 'target': target, 'eligible_rows': len(data), 'class_distribution': counts}
    def stop(message):
        return {**info, 'reason': message}
    if len(data) < 20:
        return stop('At least 20 rows with known target values are required by this project validation policy.')
    if data[target].nunique() < 2:
        return stop('At least two distinct target values are needed.')
    if target == 'nabh' and min(counts.values()) < 5:
        return stop('At least five examples in each known class are required for defensible splitting.')
    grouped = data.group.duplicated().any()
    idx = np.arange(len(data))
    if grouped:
        if data.group.nunique() < 5:
            return stop('Too few independent hospital-name groups for validation.')
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(data, groups=data.group))
    else:
        tr, te = train_test_split(idx, test_size=.2, random_state=SEED, stratify=data[target] if target == 'nabh' else None)
    if target == 'nabh' and (data.iloc[tr][target].nunique() != len(counts) or data.iloc[te][target].nunique() != len(counts)):
        return stop('The fixed group split does not cover all classes in both partitions; classifier withheld. Review groups or acquire more data.')
    if len(te) < 2 or data.iloc[tr][target].nunique() < 2:
        return stop('The training/test partition has insufficient target variation.')
    cols = feature_columns(data.iloc[tr], target)
    if not cols:
        return stop('No supported non-target features are available in the training partition.')
    X, y = frame(data, cols), data[target]
    weights = None
    if target == 'nabh' and max(data.iloc[tr][target].value_counts()) / min(data.iloc[tr][target].value_counts()) >= 2:
        weights = 'balanced'
    estimator = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1) if target == 'rating' else RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight=weights, random_state=SEED, n_jobs=-1)
    model = pipeline(cols, estimator).fit(X.iloc[tr], y.iloc[tr])
    dummy = pipeline(cols, DummyRegressor(strategy='mean') if target == 'rating' else DummyClassifier(strategy='prior')).fit(X.iloc[tr], y.iloc[tr])
    pred = model.predict(X.iloc[te])
    base = dummy.predict(X.iloc[te])
    def regression(p):
        return {'mae': finite(mean_absolute_error(y.iloc[te], p)), 'rmse': finite(np.sqrt(mean_squared_error(y.iloc[te], p))), 'r2': finite(r2_score(y.iloc[te], p, force_finite=False))}
    labels = model.named_steps['model'].classes_.tolist() if target == 'nabh' else []
    def classification(p, pipe):
        report = json_safe(classification_report(y.iloc[te], p, labels=labels, output_dict=True, zero_division=np.nan))
        auc = None
        if len(labels) == 2 and y.iloc[te].nunique() == 2:
            auc = finite(roc_auc_score((y.iloc[te] == labels[1]).astype(int), pipe.predict_proba(X.iloc[te])[:, 1]))
        return {'accuracy': report.get('accuracy'), 'precision_macro': report['macro avg']['precision'], 'recall_macro': report['macro avg']['recall'], 'f1_macro': report['macro avg']['f1-score'], 'roc_auc': auc, 'per_class': report, 'confusion_matrix': confusion_matrix(y.iloc[te], p, labels=labels).tolist(), 'labels': labels, 'zero_division_policy': 'Undefined precision/recall/F1 and ROC-AUC are null; class support is reported. Macro averages omit undefined class metrics.'}
    cv = {'status': 'unavailable', 'reason': 'Insufficient class counts or independent groups for three valid training-only folds.'}
    if grouped and data.iloc[tr].group.nunique() >= 3:
        splitter = GroupKFold(n_splits=3)
        splits = list(splitter.split(X.iloc[tr], y.iloc[tr], data.iloc[tr].group))
    elif not grouped:
        splitter = StratifiedKFold(3, shuffle=True, random_state=SEED) if target == 'nabh' else KFold(3, shuffle=True, random_state=SEED)
        splits = list(splitter.split(X.iloc[tr], y.iloc[tr])) if target != 'nabh' or y.iloc[tr].value_counts().min() >= 3 else []
    else:
        splits = []
    valid = bool(splits) and all(target != 'nabh' or (y.iloc[tr].iloc[a].nunique() == len(labels) and y.iloc[tr].iloc[b].nunique() == len(labels)) for a,b in splits)
    if valid:
        scoring = 'neg_mean_absolute_error' if target == 'rating' else 'f1_macro'
        vals = cross_val_score(model, X.iloc[tr], y.iloc[tr], cv=splits, scoring=scoring, error_score='raise')
        if target == 'rating':
            vals = -vals
        cv = {'status': 'evaluated', 'folds': 3, 'metric': 'mae' if target == 'rating' else 'f1_macro', 'scores': vals.tolist(), 'mean': float(vals.mean()), 'std': float(vals.std()), 'scope': 'Training partition only; untouched test partition used for final metrics.'}
    names = model.named_steps['features'].get_feature_names_out()
    importance = sorted([{'feature': str(n), 'importance': float(v)} for n,v in zip(names, model.named_steps['model'].feature_importances_)], key=lambda v: -v['importance'])[:20]
    schema = {'features': cols, 'categories': {c: sorted(data.iloc[tr][c].dropna().unique().tolist()) for c in cols if c in ['nabh','hospital_type']}, 'specialties': sorted({s for row in data.iloc[tr].specialties for s in row}) if 'specialties' in cols else [],
              'beds_observed': [float(data.iloc[tr].beds.min()), float(data.iloc[tr].beds.max())] if 'beds' in cols else None}
    joblib.dump({'pipeline': model, 'schema': schema}, model_dir / f'{target}.joblib')
    evaluation = [{'id': data.iloc[i].id, 'actual': float(y.iloc[i]) if target == 'rating' else str(y.iloc[i]), 'prediction': float(p) if target == 'rating' else str(p), 'baseline': float(b) if target == 'rating' else str(b)} for i,p,b in zip(te,pred,base)]
    (model_dir / f'{target}_evaluation.json').write_text(json.dumps(evaluation, indent=2, allow_nan=False), encoding='utf-8')
    return {**info, 'status': 'trained', 'algorithm': type(estimator).__name__, 'schema': schema,
            'split': 'GroupShuffleSplit by normalized hospital name' if grouped else 'Stratified random split' if target == 'nabh' else 'Random split',
            'train_rows': len(tr), 'test_rows': len(te), 'train_groups': int(data.iloc[tr].group.nunique()), 'test_groups': int(data.iloc[te].group.nunique()),
            'class_weight': weights, 'class_weight_reason': 'Balanced weights when training majority/minority ratio >= 2; otherwise no weighting.',
            'model': regression(pred) if target == 'rating' else classification(pred, model), 'baseline': regression(base) if target == 'rating' else classification(base, dummy),
            'cross_validation': cv, 'feature_importance': importance, 'feature_importance_note': 'Impurity-based importance describes this fitted forest; it is not a causal or clinical explanation.'}

def train(dataset_path: Path, model_dir: Path):
    df, report = load_dataset(dataset_path)
    if report['status'] != 'loaded':
        raise ValueError('No dataset imported. Run scripts.import_dataset first.')
    model_dir.mkdir(parents=True, exist_ok=True)
    # Remove manifest first so a failed run can never pair stale metrics and new model files.
    (model_dir / 'metrics.json').unlink(missing_ok=True)
    for name in ['rating', 'nabh']:
        (model_dir / f'{name}.joblib').unlink(missing_ok=True)
        (model_dir / f'{name}_evaluation.json').unlink(missing_ok=True)
    result = {'status': 'evaluated', 'run': {'trained_at': datetime.now(timezone.utc).isoformat(), 'seed': SEED, 'test_size': .2, 'fingerprint': report['fingerprint'], 'python': platform.python_version(), 'sklearn': sklearn.__version__, 'configuration': {'n_estimators': 200, 'min_samples_leaf': 2, 'minimum_target_rows': 20, 'minimum_class_count': 5, 'prediction_models_refit_on_full_data': False}},
              'rating': train_one(df, 'rating', model_dir), 'nabh': train_one(df, 'nabh', model_dir), 'recommendation': evaluate_proxy(df)}
    (model_dir / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    (model_dir / 'dataset_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (model_dir / 'feature_schemas.json').write_text(json.dumps({t: result[t].get('schema') for t in ['rating','nabh']}, indent=2), encoding='utf-8')
    return result
