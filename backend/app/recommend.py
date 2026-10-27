import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend(df, request):
    subset = df.copy()
    if request.city:
        subset = subset[subset.city == request.city]
    if request.specialties:
        wanted = set(request.specialties)
        subset = subset[subset.specialties.map(lambda s: wanted.issubset(s))]
    if request.min_beds is not None:
        subset = subset[subset.beds >= request.min_beds]
    if request.min_rating is not None:
        subset = subset[subset.rating >= request.min_rating]
    if request.nabh == 'required':
        subset = subset[subset.nabh == 'Accredited']
    if subset.empty:
        return [], 'No hospitals satisfy all mandatory constraints. Check recorded cities and specialties or relax minimum beds/rating and required accreditation.'
    mlb = MultiLabelBinarizer().fit(df.specialties)
    similarity = cosine_similarity(mlb.transform(subset.specialties), mlb.transform([request.specialties])).ravel() if request.specialties else np.zeros(len(subset))
    def norm(values, all_values):
        lo, hi = all_values.min(), all_values.max()
        return values.fillna(lo).map(lambda x: 0.0 if not np.isfinite(lo) or not np.isfinite(hi) else (float(x-lo)/(hi-lo) if hi > lo else 1.0)).fillna(0).to_numpy()
    parts = {'specialty': similarity, 'rating': norm(subset.rating, df.rating), 'capacity': norm(subset.beds, df.beds),
             'type': (subset.hospital_type == request.hospital_type).astype(float).to_numpy(),
             'accreditation': (subset.nabh == 'Accredited').astype(float).to_numpy()}
    weights = {'specialty': 1-request.rating_weight if request.specialties else 0,
               'rating': request.rating_weight if df.rating.notna().any() else 0,
               'capacity': .1 if df.beds.notna().any() else 0, 'type': .1 if request.hospital_type else 0,
               'accreditation': .1 if request.nabh == 'preferred' else 0}
    total = sum(weights.values())
    weights = {k: v / total if total else 0 for k, v in weights.items()}
    subset['score'] = sum(parts[k] * v for k, v in weights.items())
    subset['_dedup'] = subset.group + '|' + subset.city.fillna('')
    subset['_position'] = np.arange(len(subset))
    subset = subset.sort_values(['score', 'name', 'id'], ascending=[False, True, True]).drop_duplicates('_dedup').head(5)
    from app.data import records
    result = []
    for _, row in subset.iterrows():
        p = int(row['_position'])
        contributions = {k: round(float(parts[k][p] * v * 100), 4) for k, v in weights.items() if v}
        reasons = [f'{k.title()}: {value:.1f} weighted points' for k, value in contributions.items()]
        result.append({'hospital': records(df[df.id == row.id])[0], 'match_score': round(float(row.score)*100, 2), 'components': contributions, 'weights': weights, 'reasons': reasons,
                       'ranking_explanation': 'Ordered by the sum of weighted points (descending). Equal scores use hospital name, then ID. Missing numeric values receive zero points.'})
    return result, 'Match scores express preference alignment, not treatment success or clinical quality.'

def evaluate_proxy(df):
    from app.schemas import RecommendationRequest
    queries = sorted({(r.city, s) for r in df.itertuples() if r.city for s in r.specialties})[:100]
    scores = {'weighted': [], 'rating_only': []}
    for city, specialty in queries:
        candidates = df[df.city == city].copy().sort_values('name').drop_duplicates(['group', 'city'])
        relevance = {r.id: int(specialty in r.specialties) for r in candidates.itertuples()}
        actual, _ = recommend(df, RecommendationRequest(city=city, specialties=[specialty]))
        ranked = [r['hospital']['id'] for r in actual]
        baseline = candidates.sort_values(['rating', 'name'], ascending=[False, True], na_position='last').head(5).id.tolist()
        ideal = sorted(relevance.values(), reverse=True)[:5]
        idcg = sum(v / np.log2(i+2) for i, v in enumerate(ideal))
        for name, ids in [('weighted', ranked), ('rating_only', baseline)]:
            gains = [relevance.get(i, 0) for i in ids[:5]]
            scores[name].append({'precision_at_5': sum(gains)/5, 'ndcg_at_5': sum(v/np.log2(i+2) for i, v in enumerate(gains))/idcg if idcg else None})
    return {'status': 'evaluated' if queries else 'unavailable', 'query_count': len(queries),
            'design': 'Up to 100 lexically sorted recorded city/specialty queries. Binary relevance = city AND specialty match. Precision@5 denominator is 5 even for short lists. NDCG ideal pool contains all deduplicated city candidates. Rating-only baseline filters city but ignores specialty. The weighted engine enforces specialty, so this proxy favors constraint compliance and does not independently validate ranking among matching hospitals.',
            'results': {name: {metric: float(np.mean([v[metric] for v in vals if v[metric] is not None])) if any(v[metric] is not None for v in vals) else None for metric in ['precision_at_5', 'ndcg_at_5']} for name, vals in scores.items()},
            'limitation': 'No real user feedback. Proxy relevance is not patient preference, safety, treatment outcome, or real-world recommendation validation.'}
