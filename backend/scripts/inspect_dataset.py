from pathlib import Path
import argparse
import json
from app.data import load_dataset

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('csv', type=Path)
    args = p.parse_args()
    _, report = load_dataset(args.csv)
    print(json.dumps(report, indent=2, allow_nan=False))
