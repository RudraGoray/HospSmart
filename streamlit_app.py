"""HospSmart AI: run with `streamlit run streamlit_app.py`."""
from pathlib import Path
import json
import math

import pandas as pd
import streamlit as st
from app.data import load_dataset
from app.recommend import recommend
from app.schemas import RecommendationRequest
from app.service import (read_csv_bytes, empty_metrics, train_session,
                         load_bundled_models, predict_profile, artifacts_zip, MAX_TRAIN_ROWS)

ROOT = Path(__file__).resolve().parent
PAGES = ['Home', 'Dashboard', 'Hospital explorer', 'Recommendations', 'Rating lab',
         'NABH lab', 'Compare hospitals', 'Data & models', 'Methodology']
LABELS = {'name': 'Hospital', 'city': 'City', 'hospital_type': 'Hospital type',
          'beds': 'Beds', 'specialties': 'Specialties', 'rating': 'Recorded rating',
          'nabh': 'Recorded NABH status'}


def activate_dataset(content, name, bundled=False):
    df, report = read_csv_bytes(content, name)
    st.session_state.pop('hs_startup_error', None)
    # Commit only after validation. A bad upload leaves the active dataset intact.
    for key in list(st.session_state):
        if key.startswith('ui_'):
            del st.session_state[key]
    metrics, bundles = load_bundled_models(ROOT / 'models', report) if bundled else (empty_metrics(), {})
    st.session_state.update(hs_df=df, hs_report=report, hs_raw=content,
                            hs_metrics=metrics, hs_bundles=bundles, hs_artifacts={},
                            hs_comparison=[], hs_recommendations=None, hs_prediction={})


def initialize():
    if 'hs_df' in st.session_state:
        return
    dataset = ROOT / 'data' / 'raw' / 'hospitals.csv'
    if dataset.exists():
        try:
            activate_dataset(dataset.read_bytes(), dataset.name, bundled=True)
            return
        except (ValueError, OSError, UnicodeError, pd.errors.ParserError) as exc:
            st.session_state.hs_startup_error = str(exc)
    df, report = load_dataset(ROOT / 'data' / 'raw' / '__missing__')
    st.session_state.update(hs_df=df, hs_report=report, hs_raw=None,
                            hs_metrics=empty_metrics(), hs_bundles={}, hs_artifacts={},
                            hs_comparison=[], hs_recommendations=None, hs_prediction={})


def show_table(df, **kwargs):
    columns = [c for c in LABELS if c in df]
    display = df[columns].copy()
    if 'specialties' in display:
        display['specialties'] = display.specialties.map(lambda values: ', '.join(values) or 'Not recorded')
    st.dataframe(display.rename(columns=LABELS), hide_index=True, use_container_width=True, **kwargs)


def value_text(value):
    if isinstance(value, list):
        return ', '.join(value) or 'Not recorded'
    return 'Not recorded' if pd.isna(value) else str(value)


def options(df, column):
    if column == 'specialties':
        return sorted({item for values in df.specialties for item in values})
    return sorted(df[column].dropna().unique().tolist())


def require_data():
    if st.session_state.hs_report['status'] != 'loaded':
        st.info('Open Data & models to upload your hospital CSV. No hospital dataset was included in the original project.')
        return False
    return True


def profile_details(row):
    st.subheader(row['name'])
    for key in ['city', 'hospital_type', 'beds', 'rating', 'nabh', 'specialties']:
        st.text(f'{LABELS[key]}: {value_text(row[key])}')
    st.caption('Recorded source data. Unknown accreditation remains unknown.')


def home():
    st.title('Hospital intelligence, in one place')
    st.write('Explore hospital records, compare facilities, and inspect transparent recommendations and machine learning estimates.')
    a, b, c = st.columns(3)
    a.metric('Hospitals loaded', len(st.session_state.hs_df))
    b.metric('Models available', len(st.session_state.hs_bundles))
    c.metric('Data source', 'Your CSV' if st.session_state.hs_raw else 'Awaiting CSV')
    st.subheader('Start with your data')
    st.write('1. Open **Data & models** and upload a UTF-8 hospital CSV.\n\n'
             '2. Browse the dashboard, explorer, recommendations, and comparison screens.\n\n'
             '3. Train models to unlock the rating and NABH labs. Review their held-out results before interpreting an estimate.')
    with st.container(border=True):
        st.markdown('**Designed for exploration**')
        st.write('Recommendations show how each preference contributes to the score. Prediction labs use only features supported by the current model and preserve recorded hospital information.')
    st.info('The app starts empty because the supplied project contains no hospital dataset or trained models. A header-only CSV template is available on Data & models.')


def dashboard(df):
    st.title('Dashboard')
    if not require_data():
        return
    report = st.session_state.hs_report
    a, b, c, d = st.columns(4)
    a.metric('Hospitals', len(df))
    b.metric('Cities', df.city.nunique())
    c.metric('Specialties', len(options(df, 'specialties')))
    d.metric('Recorded accredited', int((df.nabh == 'Accredited').sum()))
    left, right = st.columns(2)
    with left:
        st.subheader('Hospitals by city')
        counts = df.city.fillna('Not recorded').value_counts().head(15)
        st.bar_chart(counts, color='#0f766e')
    with right:
        st.subheader('Recorded NABH status')
        st.bar_chart(df.nabh.value_counts(), color='#2563eb')
    left, right = st.columns(2)
    with left:
        st.subheader('Hospital types')
        st.bar_chart(df.hospital_type.fillna('Not recorded').value_counts().head(15), color='#0f766e')
    with right:
        st.subheader('Recorded ratings')
        ratings = df.rating.dropna()
        if ratings.empty:
            st.info('No ratings were recorded in this CSV.')
        else:
            counts = ratings.value_counts().sort_index() if ratings.nunique() <= 15 else pd.cut(ratings, bins=10).value_counts(sort=False)
            counts.index = counts.index.astype(str)
            st.bar_chart(counts, color='#2563eb')
    st.subheader('Highest recorded ratings')
    show_table(df[df.rating.notna()].sort_values(['rating', 'name'], ascending=[False, True]).head(5))
    with st.expander('Data quality'):
        st.write(f"Removed {report['removed_unnamed']} unnamed rows and {report['removed_duplicates']} exact duplicates.")
        st.dataframe(pd.DataFrame({'Field': list(report['missing_after']), 'Missing or unknown': list(report['missing_after'].values())}), hide_index=True)
        for warning in report['warnings']:
            st.caption(warning)


def explorer(df):
    st.title('Hospital explorer')
    if not require_data():
        return
    search = st.text_input('Search hospital name', key='ui_search', max_chars=200)
    a, b, c = st.columns(3)
    city = a.selectbox('City', [None] + options(df, 'city'), format_func=lambda v: v or 'All cities', key='ui_city')
    specialty = b.selectbox('Specialty', [None] + options(df, 'specialties'), format_func=lambda v: v or 'All specialties', key='ui_specialty')
    kind = c.selectbox('Hospital type', [None] + options(df, 'hospital_type'), format_func=lambda v: v or 'All types', key='ui_kind')
    a, b, c = st.columns(3)
    nabh = a.selectbox('Recorded NABH status', [None, 'Accredited', 'Not accredited', 'Unknown'], format_func=lambda v: v or 'All statuses', key='ui_status')
    rating = b.number_input('Minimum recorded rating (optional)', min_value=0.0, value=None, step=0.1, disabled=not df.rating.notna().any(), key='ui_min_rating')
    beds = c.number_input('Minimum beds (optional)', min_value=1, max_value=100000, value=None, disabled=not df.beds.notna().any(), key='ui_min_beds')
    subset = df.copy()
    if search:
        subset = subset[subset.name.str.contains(search, case=False, regex=False, na=False)]
    for key, value in [('city', city), ('hospital_type', kind), ('nabh', nabh)]:
        if value:
            subset = subset[subset[key] == value]
    if specialty:
        subset = subset[subset.specialties.map(lambda values: specialty in values)]
    if rating is not None:
        subset = subset[subset.rating >= rating]
    if beds is not None:
        subset = subset[subset.beds >= beds]
    sort = st.selectbox('Sort by', ['name', 'rating', 'beds'], format_func=LABELS.get, key='ui_sort')
    subset = subset.sort_values([sort, 'id'], ascending=[sort == 'name', True], na_position='last')
    st.caption(f'{len(subset):,} matching hospitals')
    if subset.empty:
        st.info('No hospitals match these filters.')
        return
    page_count = math.ceil(len(subset) / 25)
    page = st.selectbox('Results page', list(range(1, page_count + 1)), key=f'ui_results_page_{page_count}')
    page_df = subset.iloc[(page-1)*25:page*25]
    show_table(page_df)
    export = subset.drop(columns=['group']).copy()
    export['specialties'] = export.specialties.map(';'.join)
    st.download_button('Download filtered records', export.to_csv(index=False).encode('utf-8-sig'), 'hospitals-filtered.csv', 'text/csv')
    names = {r.id: f'{r.name} | {r.city or "City not recorded"} | {r.id[:6]}' for r in page_df.itertuples()}
    selected = st.selectbox('View a hospital from this page', page_df.id.tolist(), format_func=names.get, key='ui_profile')
    row = df[df.id == selected].iloc[0]
    with st.container(border=True):
        profile_details(row)
        if st.button('Add to comparison', key='ui_add_compare'):
            ids = st.session_state.hs_comparison
            if selected in ids:
                st.info('This hospital is already in the comparison.')
            elif len(ids) >= 3:
                st.warning('Compare up to three hospitals. Remove one on Compare hospitals first.')
            else:
                st.session_state.hs_comparison = ids + [selected]
                st.session_state.pop('ui_compare_ids', None)
                st.success('Added. Open Compare hospitals to see the selected facilities.')


def recommendations(df):
    st.title('Recommendations')
    if not require_data():
        return
    st.write('Set mandatory constraints and optional preferences. The score measures alignment with these choices.')
    with st.form('recommendation_form'):
        a, b = st.columns(2)
        city = a.selectbox('Required city', [None] + options(df, 'city'), format_func=lambda v: v or 'Any city', key='ui_rec_city')
        kind = b.selectbox('Preferred hospital type', [None] + options(df, 'hospital_type'), format_func=lambda v: v or 'No preference', key='ui_rec_type')
        specialties = st.multiselect('Required specialties (all must be recorded)', options(df, 'specialties'), max_selections=50, key='ui_rec_specialties')
        a, b = st.columns(2)
        beds = a.number_input('Required minimum beds', min_value=1, max_value=100000, value=None, disabled=not df.beds.notna().any(), key='ui_rec_beds')
        rating = b.number_input('Required minimum recorded rating', min_value=0.0, value=None, step=0.1, disabled=not df.rating.notna().any(), key='ui_rec_rating')
        nabh = st.selectbox('Accreditation preference', ['no_preference', 'preferred', 'required'],
                            format_func=lambda v: {'no_preference': 'No preference', 'preferred': 'Prefer recorded accredited', 'required': 'Require recorded accredited'}[v],
                            disabled=not (df.nabh != 'Unknown').any(), key='ui_rec_nabh')
        weight = st.slider('Rating weight versus specialty match', 0.0, 1.0, 0.5, 0.05, key='ui_rec_weight', help='0 favors specialty similarity; 1 favors recorded rating. Other active preferences add smaller components.')
        submitted = st.form_submit_button('Find hospitals', type='primary')
    if submitted:
        request = RecommendationRequest(city=city, hospital_type=kind, specialties=specialties,
                                        min_beds=beds, min_rating=rating, nabh=nabh, rating_weight=weight)
        results, message = recommend(df, request)
        st.session_state.hs_recommendations = {'results': results, 'message': message, 'request': request.model_dump()}
    saved = st.session_state.hs_recommendations
    if saved is not None:
        st.info(saved['message'])
        with st.expander('Preferences used for these results'):
            st.json(saved['request'])
        for index, result in enumerate(saved['results'], 1):
            with st.container(border=True):
                st.subheader(f"{index}. {result['hospital']['name']}")
                st.metric('Preference match', f"{result['match_score']:.1f} / 100")
                st.text(f"City: {value_text(result['hospital']['city'])} | Recorded NABH: {result['hospital']['nabh']}")
                st.text(f"Specialties: {value_text(result['hospital']['specialties'])}")
                st.dataframe(pd.DataFrame([{'Component': k.title(), 'Weighted points': v} for k, v in result['components'].items()]), hide_index=True, use_container_width=True)
                st.caption(result['ranking_explanation'])


def prediction_lab(target):
    st.title('Rating lab' if target == 'rating' else 'NABH lab')
    st.caption('A model estimate never replaces recorded hospital data.')
    if not require_data():
        return
    bundles = st.session_state.hs_bundles
    if target not in bundles:
        st.info(st.session_state.hs_metrics[target].get('reason', 'Train a model on Data & models.'))
        return
    bundle = bundles[target]
    schema = bundle['schema']
    values = {}
    with st.form(f'prediction_{target}'):
        st.write('Use one or more supported characteristics. Blank values are imputed by the fitted pipeline.')
        if 'hospital_type' in schema['features']:
            values['hospital_type'] = st.selectbox('Hospital type', [None] + schema['categories']['hospital_type'], format_func=lambda v: v or 'Not provided', key=f'ui_{target}_type')
        if 'beds' in schema['features']:
            values['beds'] = st.number_input('Bed capacity', min_value=1, max_value=100000, value=None, key=f'ui_{target}_beds')
        if 'specialties' in schema['features']:
            values['specialties'] = st.multiselect('Specialties', schema['specialties'], max_selections=50, key=f'ui_{target}_specialties')
        if 'nabh' in schema['features']:
            values['nabh'] = st.selectbox('Recorded NABH status', ['Unknown'] + [v for v in schema['categories']['nabh'] if v != 'Unknown'], key=f'ui_{target}_nabh')
        submitted = st.form_submit_button('Estimate rating' if target == 'rating' else 'Estimate NABH label', type='primary')
    if submitted:
        try:
            result = predict_profile(target, values, bundle, st.session_state.hs_report)
            st.session_state.hs_prediction[target] = {'result': result, 'inputs': values}
        except ValueError as exc:
            st.session_state.hs_prediction.pop(target, None)
            st.error(str(exc))
    saved = st.session_state.hs_prediction.get(target)
    if saved:
        result = saved['result']
        if target == 'rating':
            denominator = result['explicit_denominator']
            st.metric('Estimated rating', f"{result['prediction']:.2f}" + (f' / {denominator:g}' if denominator else ''))
            st.caption(f"Observed dataset range: {result['observed_range']}. An observed range is not an official scale.")
        else:
            st.metric('Estimated label', result['prediction'])
            st.dataframe(pd.DataFrame([{'Label': k, 'Uncalibrated model probability': v} for k, v in result['probabilities'].items()]), hide_index=True)
        st.caption('Provided: ' + ', '.join(result['provided_features']))
        st.caption('Imputed or absent: ' + (', '.join(result['missing_features']) or 'None'))
        for warning in result['warnings']:
            st.warning(warning)
        with st.expander('Inputs used for this estimate'):
            st.json(saved['inputs'])
    with st.expander('Model validation and supported features'):
        st.json(st.session_state.hs_metrics[target])


def compare(df):
    st.title('Compare hospitals')
    if not require_data():
        return
    names = {r.id: f'{r.name} | {r.city or "City not recorded"} | {r.id[:6]}' for r in df.itertuples()}
    selected = st.multiselect('Choose up to three hospitals', list(names), default=st.session_state.hs_comparison,
                              format_func=names.get, max_selections=3, key='ui_compare_ids')
    st.session_state.hs_comparison = selected
    if not selected:
        st.info('Select facilities here, or add them from Hospital explorer.')
        return
    rows = df.set_index('id').loc[selected]
    columns = st.columns(len(selected))
    for column, (_, row) in zip(columns, rows.iterrows()):
        with column, st.container(border=True):
            profile_details(row)
    table = rows[list(LABELS)].copy()
    for key in table:
        table[key] = table[key].map(value_text)
    table.columns = [LABELS[key] for key in table.columns]
    st.dataframe(table.set_index('Hospital').T, use_container_width=True)
    st.caption('Missing values mean the source did not record them; they are not zero.')


def model_summary(metrics):
    for target, title in [('rating', 'Rating model'), ('nabh', 'NABH model')]:
        with st.container(border=True):
            item = metrics[target]
            st.subheader(title)
            if item['status'] != 'trained':
                st.info(item.get('reason', 'Model unavailable.'))
                continue
            st.caption(f"{item['algorithm']} | {item['train_rows']} training rows | {item['test_rows']} test rows | {item['split']}")
            metric_keys = ['mae', 'rmse', 'r2'] if target == 'rating' else ['accuracy', 'f1_macro', 'roc_auc']
            st.dataframe(pd.DataFrame([{'Metric': key.upper(), 'Forest': item['model'].get(key), 'Dummy baseline': item['baseline'].get(key)} for key in metric_keys]), hide_index=True, use_container_width=True)
            st.caption('Lower MAE/RMSE is better. Higher R2, accuracy, F1 and ROC-AUC is better. Small test sets can give unstable results.')
            if item.get('feature_importance'):
                with st.expander('Feature importance and training-only cross validation'):
                    st.bar_chart(pd.DataFrame(item['feature_importance']).set_index('feature'), color='#0f766e')
                    st.caption(item['feature_importance_note'])
                    st.json(item['cross_validation'])


def data_models():
    st.title('Data & models')
    st.write('Upload a UTF-8 CSV with a hospital name column. Optional fields enable filtering, recommendations, and model training.')
    st.code('Hospital Name,City,Hospital Type,Number of Beds,Medical Specialties,Hospital Rating,NABH Accreditation Status', language=None)
    st.download_button('Download blank CSV template', (ROOT / 'data' / 'hospitals-template.csv').read_bytes(), 'hospitals-template.csv', 'text/csv')
    st.caption('Upload limit: 10 MB and 20,000 rows. Use semicolons between specialties. Only upload hospital-level information intended for this app.')
    with st.form('csv_import', clear_on_submit=True):
        upload = st.file_uploader('Hospital CSV', type=['csv'], key='csv_upload')
        submitted = st.form_submit_button('Load CSV', type='primary')
    if submitted:
        if upload is None:
            st.error('Select a CSV first.')
        else:
            try:
                activate_dataset(upload.getvalue(), upload.name)
                st.rerun()
            except (ValueError, OSError, UnicodeError, pd.errors.ParserError) as exc:
                st.error(f'CSV could not be loaded: {exc}. The previous dataset, if any, remains active.')
    if st.button('Reset session to bundled data', key='reset_session'):
        for key in list(st.session_state):
            if key.startswith(('hs_', 'ui_')):
                del st.session_state[key]
        st.rerun()
    if not require_data():
        return
    df, report = st.session_state.hs_df, st.session_state.hs_report
    st.success(f"{len(df):,} hospitals loaded from {report['source_filename']}.")
    st.caption('Uploaded CSVs and in-app models belong to this browser session. Reloading or a server restart may require another upload/training run. Bundle your CSV and model artifacts in the repository for startup availability.')
    show_table(df.head(10))
    with st.expander('Cleaning report, aliases, and rating scale'):
        st.json(report)
        st.markdown((ROOT / 'data' / 'README.md').read_text(encoding='utf-8'))
    st.download_button('Download active original CSV', st.session_state.hs_raw, 'hospitals.csv', 'text/csv')
    st.subheader('Train and evaluate models')
    st.write('Training uses 200-tree random forests, a fixed 80/20 split, seed 42, and three-fold training-only cross validation when valid. Models need at least 20 known targets and two distinct target values; NABH also needs at least five examples per class.')
    st.caption('In-app training is limited to 5,000 cleaned hospitals to keep cloud resource use manageable. The offline script supports larger datasets.')
    if st.button('Train models on current data', type='primary', disabled=len(df) > MAX_TRAIN_ROWS, key='train_models'):
        try:
            with st.spinner('Training and evaluating models. This may take a few minutes.'):
                metrics, bundles, artifacts = train_session(st.session_state.hs_raw)
            st.session_state.update(hs_metrics=metrics, hs_bundles=bundles, hs_artifacts=artifacts, hs_prediction={})
            st.success('Training finished. Review the results below; tasks with insufficient data remain unavailable.')
        except Exception as exc:
            st.error(f'Training could not complete: {exc}. Try a smaller dataset or use the offline script. Previously available models are retained.')
    metrics = st.session_state.hs_metrics
    model_summary(metrics)
    if metrics.get('run'):
        st.download_button('Download evaluation metrics', json.dumps(metrics, indent=2, allow_nan=False), 'metrics.json', 'application/json')
        if st.session_state.hs_artifacts:
            st.download_button('Download trained model files', artifacts_zip(st.session_state.hs_artifacts), 'hospsmart-models.zip', 'application/zip')
            st.caption('For startup models: extract this download into the project root so files land in models/, then add the exact active CSV as data/raw/hospitals.csv and commit both. Use the same requirements and Python version.')
        with st.expander('Recommendation proxy evaluation'):
            st.json(metrics['recommendation'])


def methodology():
    st.title('Models & methodology')
    st.write('This educational application supports hospital discovery and machine learning exploration. It does not diagnose, recommend treatment, or predict clinical outcomes.')
    st.subheader('Data and recommendations')
    st.write('The importer normalizes documented column aliases, city names, ownership labels, numbers, specialties and known NABH labels. It removes exact duplicates and unnamed rows. Missing information remains visible. Recommendations enforce city, specialties, minimum beds/rating, and required recorded accreditation. Optional preferences contribute weighted points.')
    st.subheader('Prediction and evaluation')
    st.write('Hospital names, identifiers, and city are excluded from prediction features. Rating is never a NABH predictor. Preprocessing is fitted inside training partitions and cross-validation folds. Repeated normalized names trigger a group split. Saved forests remain fitted on the training partition so the displayed test metrics describe the served model.')
    st.write('Compare each forest with its dummy baseline. Feature importance is descriptive, not causal. Recommendation proxy relevance rewards constraint compliance and does not validate patient preferences or outcomes. Recorded accreditation is supplied by the source and is not independently verified.')
    st.subheader('Deployment and session behavior')
    st.write('Streamlit runs the interface and Python logic in one process. Uploads and in-app models are stored in session memory and temporary processing directories, without a shared global dataset cache. A CSV bundled at data/raw/hospitals.csv loads for every visitor. Matching trusted model files in models/ load at startup.')
    st.write('Changing the CSV clears session predictions, comparison choices, and models. Bundled model fingerprints must match the exact CSV bytes. No API keys, React build, or separate FastAPI server are required.')
    with st.expander('Original ML report'):
        st.caption('This report describes the inherited model design. References to FastAPI endpoints and terminal setup refer to the original application; use this package README for Streamlit operation.')
        st.markdown((ROOT / 'docs' / 'ML_REPORT.md').read_text(encoding='utf-8'))
    with st.expander('Project limitations'):
        st.markdown((ROOT / 'docs' / 'LIMITATIONS.md').read_text(encoding='utf-8'))


def main():
    st.set_page_config(page_title='HospSmart AI', page_icon='🏥', layout='wide')
    initialize()
    with st.sidebar:
        st.title('HospSmart AI')
        st.caption('Hospital discovery & ML exploration')
        page = st.radio('Navigate', PAGES, key='page')
        st.divider()
        st.caption(f"{len(st.session_state.hs_df):,} hospitals | {len(st.session_state.hs_bundles)} prediction models")
        st.caption('Educational use. Verify hospital services and accreditation with official sources.')
    if st.session_state.get('hs_startup_error'):
        st.warning('The bundled CSV could not be loaded: ' + st.session_state.hs_startup_error)
    df = st.session_state.hs_df
    routes = {'Home': home, 'Dashboard': lambda: dashboard(df),
              'Hospital explorer': lambda: explorer(df), 'Recommendations': lambda: recommendations(df),
              'Rating lab': lambda: prediction_lab('rating'), 'NABH lab': lambda: prediction_lab('nabh'),
              'Compare hospitals': lambda: compare(df), 'Data & models': data_models, 'Methodology': methodology}
    routes[page]()
    st.divider()
    st.caption('HospSmart AI is an educational demonstration. Match scores and model estimates do not guarantee hospital quality, treatment success, or verified accreditation.')


if __name__ == '__main__':
    main()
