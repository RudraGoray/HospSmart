from io import BytesIO
from pathlib import Path
import json
import zipfile
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from app.recommend import recommend
from app.schemas import RecommendationRequest
from app.service import (read_csv_bytes, train_session, predict_profile,
                         load_bundled_models, artifacts_zip, MAX_UPLOAD_BYTES)
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def csv_bytes():
    rows = [{'name': f'TEST ONLY Facility {i}', 'city': 'Bangalore' if i % 2 else 'Delhi',
             'hospital_type': 'Private' if i % 3 else 'Govt', 'beds': 50 + i * 10,
             'rating': round(2.5 + (i % 20) / 10, 1),
             'specialties': 'Cardiology;Neurology' if i % 2 else 'Neurology',
             'nabh': 'yes' if i % 3 else 'no'} for i in range(60)]
    return pd.DataFrame(rows).to_csv(index=False).encode()

@pytest.fixture(scope='module')
def trained(csv_bytes):
    return train_session(csv_bytes)

def app_with_data(content, trained_result=None):
    app = AppTest.from_file(str(ROOT / 'streamlit_app.py'), default_timeout=120).run()
    df, report = read_csv_bytes(content, 'SYNTHETIC-TEST-ONLY.csv')
    app.session_state['hs_df'] = df
    app.session_state['hs_report'] = report
    app.session_state['hs_raw'] = content
    if trained_result:
        metrics, bundles, artifacts = trained_result
        app.session_state['hs_metrics'] = metrics
        app.session_state['hs_bundles'] = bundles
        app.session_state['hs_artifacts'] = artifacts
    return app

def navigate(app, page):
    app.radio(key='page').set_value(page).run()
    assert not app.exception, [entry.message for entry in app.exception]
    return app

def test_import_contract(csv_bytes):
    original = bytes(csv_bytes)
    df, report = read_csv_bytes(csv_bytes, '../SYNTHETIC-TEST-ONLY.csv')
    assert len(df) == 60 and 'Bengaluru' in set(df.city)
    assert 'Government' in set(df.hospital_type)
    assert report['source_filename'] == 'SYNTHETIC-TEST-ONLY.csv'
    assert len(report['fingerprint']) == 64 and csv_bytes == original
    with pytest.raises(ValueError, match='name column'):
        read_csv_bytes(b'rating\n4\n')
    with pytest.raises(ValueError, match='Mixed rating'):
        read_csv_bytes(b'name,rating\nTEST ONLY A,4/5\nTEST ONLY B,8/10\n')
    with pytest.raises(ValueError, match='10 MB'):
        read_csv_bytes(b'x' * (MAX_UPLOAD_BYTES + 1))
    with pytest.raises(ValueError, match='20,000'):
        read_csv_bytes(b'name\n' + b'TEST ONLY\n' * 20001)

def test_missing_numeric_points():
    df, _ = read_csv_bytes(b'name,rating,beds\nTEST ONLY A,5,100\nTEST ONLY B,,\n')
    rows, _ = recommend(df, RecommendationRequest())
    missing = next(row for row in rows if row['hospital']['name'] == 'Test Only B')
    assert missing['match_score'] == 0

def test_recommendation_constraints(csv_bytes):
    df, _ = read_csv_bytes(csv_bytes)
    rows, _ = recommend(df, RecommendationRequest(city='Bengaluru', specialties=['Cardiology'], min_beds=100, nabh='required'))
    assert rows
    for row in rows:
        hospital = row['hospital']
        assert hospital['city'] == 'Bengaluru' and 'Cardiology' in hospital['specialties']
        assert hospital['beds'] >= 100 and hospital['nabh'] == 'Accredited'
        assert abs(sum(row['components'].values()) - row['match_score']) < .02

def test_models_predictions_and_export(csv_bytes, trained):
    metrics, bundles, artifacts = trained
    _, report = read_csv_bytes(csv_bytes)
    assert all(metrics[target]['status'] == 'trained' for target in ['rating', 'nabh'])
    assert metrics['rating']['cross_validation']['status'] == 'evaluated'
    rating = predict_profile('rating', {'beds': 200}, bundles['rating'], report)
    nabh = predict_profile('nabh', {'beds': 200}, bundles['nabh'], report)
    assert isinstance(rating['prediction'], float)
    assert abs(sum(nabh['probabilities'].values()) - 1) < 1e-9
    with pytest.raises(ValueError, match='at least one'):
        predict_profile('rating', {}, bundles['rating'], report)
    with pytest.raises(ValueError, match='training partition'):
        predict_profile('rating', {'hospital_type': 'Imaginary'}, bundles['rating'], report)
    with zipfile.ZipFile(BytesIO(artifacts_zip(artifacts))) as archive:
        assert 'models/metrics.json' in archive.namelist()
        assert 'models/rating.joblib' in archive.namelist()
    assert json.loads(artifacts['metrics.json'])['run']['fingerprint'] == report['fingerprint']

def test_withheld_models():
    metrics, bundles, _ = train_session(b'name,beds,rating,nabh\nTEST ONLY A,50,4,yes\nTEST ONLY B,100,5,no\n')
    assert not bundles and '20 rows' in metrics['rating']['reason']
    assert metrics['nabh']['status'] == 'unavailable'

def test_bundled_fingerprint_corruption(csv_bytes, trained, tmp_path):
    _, report = read_csv_bytes(csv_bytes)
    for name, content in trained[2].items():
        (tmp_path / name).write_bytes(content)
    _, bundles = load_bundled_models(tmp_path, report)
    assert set(bundles) == {'rating', 'nabh'}
    _, changed = read_csv_bytes(csv_bytes + b'\n')
    metrics, bundles = load_bundled_models(tmp_path, changed)
    assert metrics['status'] == 'stale' and not bundles
    (tmp_path / 'nabh.joblib').write_bytes(b'not a model')
    metrics, bundles = load_bundled_models(tmp_path, report)
    assert set(bundles) == {'rating'} and metrics['nabh']['status'] == 'unavailable'

def test_empty_all_pages():
    app = AppTest.from_file(str(ROOT / 'streamlit_app.py'), default_timeout=60).run()
    assert not app.exception
    for page in app.radio(key='page').options:
        navigate(app, page)

def test_minimal_all_pages():
    app = app_with_data(b'Hospital Name\nTEST ONLY A\nTEST ONLY B\n')
    for page in app.radio(key='page').options:
        navigate(app, page)
    navigate(app, 'Recommendations')
    next(button for button in app.button if button.label == 'Find hospitals').click().run()
    assert not app.exception and app.session_state['hs_recommendations']['results']

def test_loaded_pages_search_compare_recommendations(csv_bytes):
    app = app_with_data(csv_bytes)
    for page in app.radio(key='page').options:
        navigate(app, page)
    navigate(app, 'Hospital explorer')
    app.text_input(key='ui_search').set_value('Facility 59').run()
    assert not app.exception and len(app.dataframe) == 1
    app.button(key='ui_add_compare').click().run()
    assert len(app.session_state['hs_comparison']) == 1
    navigate(app, 'Compare hospitals')
    assert len(app.multiselect(key='ui_compare_ids').value) == 1
    navigate(app, 'Recommendations')
    app.selectbox(key='ui_rec_city').set_value('Bengaluru')
    app.multiselect(key='ui_rec_specialties').set_value(['Cardiology'])
    next(button for button in app.button if button.label == 'Find hospitals').click().run()
    assert not app.exception
    results = app.session_state['hs_recommendations']['results']
    assert results and all(result['hospital']['city'] == 'Bengaluru' for result in results)

def test_trained_labs_and_metrics(csv_bytes, trained):
    app = app_with_data(csv_bytes, trained)
    for page in ['Data & models', 'Rating lab', 'NABH lab', 'Methodology']:
        navigate(app, page)
    for target, page in [('rating', 'Rating lab'), ('nabh', 'NABH lab')]:
        navigate(app, page)
        app.number_input(key=f'ui_{target}_beds').set_value(200)
        next(button for button in app.button if button.label.startswith('Estimate')).click().run()
        assert not app.exception and target in app.session_state['hs_prediction']

def test_upload_replaces_models_and_rejects_bad_csv(csv_bytes, trained, monkeypatch):
    import streamlit
    class Upload:
        name = 'TEST-ONLY-UPLOAD.csv'
        content = b'name,beds\nTEST ONLY replacement,90\n'
        def getvalue(self):
            return self.content
    upload = Upload()
    # AppTest lacks a file-uploader setter; supply the widget return value and
    # exercise the real form submission, validator and session transition.
    monkeypatch.setattr(streamlit, 'file_uploader', lambda *args, **kwargs: upload)
    app = app_with_data(csv_bytes, trained)
    navigate(app, 'Data & models')
    next(button for button in app.button if button.label == 'Load CSV').click().run()
    assert not app.exception and len(app.session_state['hs_df']) == 1
    assert not app.session_state['hs_bundles']
    upload.content = b'rating\n4\n'
    next(button for button in app.button if button.label == 'Load CSV').click().run()
    assert not app.exception and app.error
    assert len(app.session_state['hs_df']) == 1

def test_train_button_and_predict(csv_bytes):
    app = app_with_data(csv_bytes)
    navigate(app, 'Data & models')
    app.button(key='train_models').click().run(timeout=120)
    assert not app.exception and set(app.session_state['hs_bundles']) == {'rating', 'nabh'}
    navigate(app, 'Rating lab')
    app.number_input(key='ui_rating_beds').set_value(200)
    next(button for button in app.button if button.label == 'Estimate rating').click().run()
    assert not app.exception and app.session_state['hs_prediction']['rating']

def test_bundled_dataset_and_models_startup(csv_bytes, trained, tmp_path):
    import shutil
    shutil.copyfile(ROOT / 'streamlit_app.py', tmp_path / 'streamlit_app.py')
    (tmp_path / 'data/raw').mkdir(parents=True)
    (tmp_path / 'data/raw/hospitals.csv').write_bytes(csv_bytes)
    (tmp_path / 'models').mkdir()
    for name, content in trained[2].items():
        (tmp_path / 'models' / name).write_bytes(content)
    app = AppTest.from_file(str(tmp_path / 'streamlit_app.py'), default_timeout=60).run()
    assert not app.exception
    assert len(app.session_state['hs_df']) == 60
    assert set(app.session_state['hs_bundles']) == {'rating', 'nabh'}
    navigate(app, 'NABH lab')
    app.number_input(key='ui_nabh_beds').set_value(200)
    next(button for button in app.button if button.label.startswith('Estimate')).click().run()
    assert not app.exception and app.session_state['hs_prediction']['nabh']


def test_separate_sessions_do_not_share_data(csv_bytes):
    first = app_with_data(csv_bytes)
    second = AppTest.from_file(str(ROOT / 'streamlit_app.py'), default_timeout=60).run()
    assert len(first.session_state['hs_df']) == 60
    assert len(second.session_state['hs_df']) == 0
    assert not second.session_state['hs_bundles']
