from pathlib import Path
import json
from app.data import load_dataset
from app.recommend import evaluate_proxy

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    df, report = load_dataset(root / 'data/raw/hospitals.csv')
    if report['status'] != 'loaded':
        raise SystemExit('Import a dataset before evaluation.')
    result = evaluate_proxy(df)
    print(json.dumps(result, indent=2, allow_nan=False))
