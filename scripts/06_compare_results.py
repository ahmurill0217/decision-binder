"""Build comparison tables and figures from saved predictions; no model loading."""
import argparse
from pathlib import Path
from decision_binder.compare import comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment.yaml')
    parser.add_argument('--mode', choices=('primary', 'target_holdout'), default='primary')
    parser.add_argument('--training-metadata', help='Matching full-run metadata.json for the loss plot')
    args = parser.parse_args()
    comparison(Path(args.config).resolve().parent.parent, args.mode, args.training_metadata,
               config_path=args.config)


if __name__ == '__main__':
    main()
