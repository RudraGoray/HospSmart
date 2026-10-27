"""Optional offline training: python -m scripts.train --dataset data/raw/hospitals.csv"""
import argparse
from pathlib import Path
from app.training import train


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Train HospSmart models on a trusted hospital CSV.')
    parser.add_argument('--dataset', type=Path, default=root / 'data/raw/hospitals.csv')
    parser.add_argument('--models', type=Path, default=root / 'models')
    args = parser.parse_args()
    result = train(args.dataset, args.models)
    for target in ['rating', 'nabh']:
        info = result[target]
        print(f"{target}: {info['status']}" + (f" - {info['reason']}" if info.get('reason') else ''))
    print(f'Evaluation and artifacts: {args.models.resolve()}')


if __name__ == '__main__':
    main()
