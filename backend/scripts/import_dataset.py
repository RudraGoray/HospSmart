from pathlib import Path
import argparse
import hashlib
import json
import shutil
from app.data import load_dataset

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description='Validate and copy a hospital CSV without modifying its contents.')
    parser.add_argument('csv', type=Path)
    parser.add_argument('--replace', action='store_true', help='Archive previous raw data and replace active dataset.')
    args = parser.parse_args()
    _, report = load_dataset(args.csv)
    if report['status'] != 'loaded':
        parser.error('The source CSV does not exist.')
    dest = ROOT / 'data/raw/hospitals.csv'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.read_bytes() != args.csv.read_bytes():
        if not args.replace:
            parser.error('Active CSV already exists. Use --replace to archive it before replacement.')
        fingerprint = hashlib.sha256(dest.read_bytes()).hexdigest()
        shutil.copy2(dest, dest.with_name(f'archive-{fingerprint}.csv'))
    if args.csv.resolve() != dest.resolve():
        shutil.copy2(args.csv, dest)
    print(json.dumps(report, indent=2))
    print('Imported without changing source bytes. Now run: python -m scripts.train')

if __name__ == '__main__':
    main()
