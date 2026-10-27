from pathlib import Path
import argparse
import json
from app.training import train

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, default=root / 'data/raw/hospitals.csv')
    p.add_argument('--models', type=Path, default=root / 'models')
    args = p.parse_args()
    try:
        result = train(args.dataset, args.models)
    except (ValueError, OSError) as exc:
        p.exit(1, f'Training stopped: {exc}\n')
    print(json.dumps(result, indent=2, allow_nan=False))
