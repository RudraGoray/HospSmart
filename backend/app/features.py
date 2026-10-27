import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

class SpecialtyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoder_ = MultiLabelBinarizer().fit(X.iloc[:, 0])
        return self

    def transform(self, X):
        known = set(self.encoder_.classes_)
        return self.encoder_.transform([[v for v in row if v in known] for row in X.iloc[:, 0]]).astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.encoder_.classes_, dtype=object)

def feature_columns(df, target):
    candidates = ['hospital_type', 'beds', 'specialties'] + (['nabh'] if target == 'rating' else [])
    return [c for c in candidates if (df[c].map(bool).any() if c == 'specialties' else (df[c].notna() & (df[c] != 'Unknown')).any())]

def pipeline(columns, estimator):
    transforms = []
    if 'beds' in columns:
        transforms.append(('numeric', SimpleImputer(strategy='median', keep_empty_features=True), ['beds']))
    cats = [c for c in columns if c in ['hospital_type', 'nabh']]
    if cats:
        transforms.append(('category', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Unknown')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cats))
    if 'specialties' in columns:
        transforms.append(('specialty', SpecialtyEncoder(), ['specialties']))
    return Pipeline([('features', ColumnTransformer(transforms, remainder='drop')), ('model', estimator)])

def frame(df, columns):
    X = df[columns].copy()
    for c in columns:
        if c not in ['specialties', 'beds']:
            X[c] = X[c].fillna(np.nan)
    return X
