"""Synthetic fixtures test software only; never used as application hospital data."""
import json
import shutil
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app.data import load_dataset
from app.features import feature_columns
from app.main import create_app
from app.schemas import RecommendationRequest
from app.recommend import recommend
from app.training import train

@pytest.fixture
def csv(tmp_path):
    path=tmp_path/'SYNTHETIC-TEST-ONLY.csv'
    pd.DataFrame([{'Hospital Name':f'TEST ONLY Facility {i}', 'City':'Bangalore' if i%2 else 'Delhi',
        'Hospital Type':'Private' if i%3 else 'Govt', 'Number of Beds':f'{50+i*10} beds',
        'Medical Specialties':'Cardiology;Orthopedics' if i%3 else 'Neurology',
        'Hospital Rating':str(round(2.5+(i%20)/10,1)), 'NABH Accreditation Status':'yes' if i%2 else 'no'} for i in range(60)]).to_csv(path,index=False)
    return path

@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    root=tmp_path_factory.mktemp('training');path=root/'SYNTHETIC-TEST-ONLY.csv'
    pd.DataFrame([{'name':f'TEST ONLY {i}','city':'City A' if i%2 else 'City B','hospital_type':'Private' if i%3 else 'Government',
        'beds':100+i*5,'rating':2.5+(i%20)/10,'specialties':'Cardiology;Neurology' if i%2 else 'Neurology','nabh':'yes' if i%3 else 'no'} for i in range(60)]).to_csv(path,index=False)
    models=root/'models';result=train(path,models)
    return path,models,result

def test_missing_and_normalization(csv,tmp_path):
    assert load_dataset(tmp_path/'absent')[1]['status']=='missing'
    original=csv.read_bytes();df,r=load_dataset(csv)
    assert 'Bengaluru' in set(df.city) and 'Government' in set(df.hospital_type)
    assert df.beds.iloc[0]==50 and len(df.specialties.iloc[1])==2
    assert set(df.nabh)=={'Accredited','Not accredited'}
    assert csv.read_bytes()==original and len(r['fingerprint'])==64

def test_invalid_duplicates_unknown(tmp_path):
    path=tmp_path/'bad.csv'
    pd.DataFrame([{'name':'TEST ONLY A','beds':'-5','rating':'-1','nabh':'Pending'}, {'name':' TEST ONLY A ','beds':'-5','rating':'-1','nabh':'Pending'}, {'name':'TEST ONLY B','beds':'100-200','rating':'4.2/5','nabh':'expired'}]).to_csv(path,index=False)
    df,r=load_dataset(path)
    assert len(df)==2 and r['removed_duplicates']==1 and df.beds.isna().all()
    assert df.rating.isna().sum()==1 and set(df.nabh)=={'Unknown'}

def test_minimal_and_invalid_schema(tmp_path):
    p=tmp_path/'minimal.csv';pd.DataFrame({'Hospital Name':['TEST ONLY A']}).to_csv(p,index=False)
    df,r=load_dataset(p);assert r['available_columns']==['name'] and df.rating.isna().all()
    pd.DataFrame({'rating':[4]}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='name column'):load_dataset(p)
    pd.DataFrame({'name':['TEST ONLY A','TEST ONLY B'],'rating':['4/5','8/10']}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='Mixed rating'):load_dataset(p)

def test_no_target_leakage(csv):
    df,_=load_dataset(csv)
    assert not {'nabh','rating','name','id','city','group'} & set(feature_columns(df,'nabh'))
    assert 'rating' not in feature_columns(df,'rating')

def test_recommendation_constraints_and_points(csv):
    df,_=load_dataset(csv)
    rows,_=recommend(df,RecommendationRequest(city='Bengaluru',specialties=['Cardiology'],min_beds=100,nabh='required'))
    assert 0<len(rows)<=5
    assert all(r['hospital']['city']=='Bengaluru' and 'Cardiology' in r['hospital']['specialties'] and r['hospital']['beds']>=100 and r['hospital']['nabh']=='Accredited' for r in rows)
    assert len({r['hospital']['id'] for r in rows})==len(rows)
    assert [r['match_score'] for r in rows]==sorted([r['match_score'] for r in rows],reverse=True)
    for r in rows:assert abs(sum(r['components'].values())-r['match_score'])<.02
    assert recommend(df,RecommendationRequest(city='Nonexistent'))[0]==[]

def test_weight_changes_ranking(tmp_path):
    p=tmp_path/'rank.csv';pd.DataFrame([{'name':'TEST ONLY A','specialties':'Cardiology','rating':2},{'name':'TEST ONLY B','specialties':'Cardiology;Neurology;Oncology','rating':5}]).to_csv(p,index=False)
    df,_=load_dataset(p)
    a,_=recommend(df,RecommendationRequest(specialties=['Cardiology'],rating_weight=0));b,_=recommend(df,RecommendationRequest(specialties=['Cardiology'],rating_weight=1))
    assert a[0]['hospital']['name']=='Test Only A' and b[0]['hospital']['name']=='Test Only B'

def test_empty_api(tmp_path):
    with TestClient(create_app(tmp_path/'absent',tmp_path/'models')) as c:
        assert c.get('/api/health').json()['dataset']=='missing'
        assert c.get('/api/overview').json()['total']==0
        assert c.get('/api/hospitals').json()['items']==[]
        assert c.post('/api/predict/rating',json={'beds':100}).status_code==503
        assert c.post('/api/recommendations',json={}).status_code==503

def test_routes_filters_and_comparison(csv,tmp_path):
    with TestClient(create_app(csv,tmp_path/'models')) as c:
        for route in ['/api/overview','/api/filters','/api/models/metrics','/api/models/methodology']:assert c.get(route).status_code==200
        listing=c.get('/api/hospitals?city=Bengaluru&page_size=2&sort=rating').json()
        assert listing['total']==30 and len(listing['items'])==2
        ids=[h['id'] for h in listing['items']]
        assert c.get('/api/hospitals/'+ids[0]).status_code==200
        assert len(c.get('/api/hospitals/compare?ids='+','.join(ids)).json()['hospitals'])==2
        assert c.get('/api/hospitals/compare?ids=a,b,c,d').status_code==422
        assert c.get('/api/hospitals/not-real').status_code==404
        assert c.get('/api/hospitals?search=not-real').json()['total']==0
        assert c.post('/api/recommendations',json={'city':'not-real'}).status_code==422
        assert c.post('/api/recommendations',json={'specialties':['not-real']}).status_code==422
        assert c.post('/api/recommendations',json={'city':'Bengaluru'}).status_code==200

def test_input_validation(csv,tmp_path):
    with TestClient(create_app(csv,tmp_path/'models')) as c:
        assert c.post('/api/predict/rating',json={'beds':-1}).status_code==422
        assert c.post('/api/predict/nabh',json={'nabh':'Accredited'}).status_code==422
        assert c.post('/api/recommendations',json={'rating_weight':2}).status_code==422
        assert c.get('/api/hospitals?page=0').status_code==422

def test_real_training_workflow_on_test_fixture(trained):
    path,models,r=trained
    assert r['rating']['status']=='trained' and r['nabh']['status']=='trained'
    assert set(r['rating']['model'])=={'mae','rmse','r2'}
    assert r['rating']['cross_validation']['status']=='evaluated'
    assert r['nabh']['model']['roc_auc'] is not None and r['recommendation']['query_count']>0
    assert json.loads((models/'metrics.json').read_text())['run']['fingerprint']==r['run']['fingerprint']

def test_model_predictions(trained):
    path,models,r=trained
    with TestClient(create_app(path,models)) as c:
        response=c.post('/api/predict/rating',json={'beds':200,'hospital_type':'Private'})
        assert response.status_code==200 and isinstance(response.json()['prediction'],float)
        pred=c.post('/api/predict/nabh',json={'beds':200}).json()
        assert abs(sum(pred['probabilities'].values())-1)<1e-9
        assert 'beds' in pred['provided_features'] and 'specialties' in pred['unavailable_features']
        assert c.post('/api/predict/nabh',json={'hospital_type':'Imaginary'}).status_code==422
        assert c.post('/api/predict/rating',json={}).status_code==422

def test_stale_corrupt_and_missing_models(trained,tmp_path):
    path,models,r=trained
    altered=tmp_path/'changed.csv';altered.write_bytes(path.read_bytes()+b'\n')
    with TestClient(create_app(altered,models)) as c:
        assert c.get('/api/models/metrics').json()['status']=='stale'
        assert c.post('/api/predict/rating',json={'beds':200}).status_code==503
    broken=tmp_path/'broken';shutil.copytree(models,broken);(broken/'nabh.joblib').write_bytes(b'invalid');(broken/'rating.joblib').unlink()
    with TestClient(create_app(path,broken)) as c:
        assert all(v=='unavailable' for v in c.get('/api/health').json()['models'].values())

@pytest.mark.parametrize('minority',[0,1])
def test_invalid_class_distribution_withheld(csv,tmp_path,minority):
    df=pd.read_csv(csv);df['NABH Accreditation Status']='yes'
    if minority:df.loc[0,'NABH Accreditation Status']='no'
    df.to_csv(csv,index=False);r=train(csv,tmp_path/'withheld')
    assert r['nabh']['status']=='unavailable'

def test_group_split_and_balancing(tmp_path):
    path=tmp_path/'groups.csv'
    pd.DataFrame([{'name':f'TEST ONLY Group {i//2}','beds':i+20,'rating':2+(i%10)/5,'nabh':'no' if i//2%4==0 else 'yes'} for i in range(80)]).to_csv(path,index=False)
    r=train(path,tmp_path/'models')
    assert r['rating']['split'].startswith('GroupShuffleSplit') and r['nabh']['class_weight']=='balanced'
    df,_=load_dataset(path);evals=json.loads((tmp_path/'models/rating_evaluation.json').read_text());test_ids={v['id'] for v in evals}
    assert not set(df[df.id.isin(test_ids)].group)&set(df[~df.id.isin(test_ids)].group)
