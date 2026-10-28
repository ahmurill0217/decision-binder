"""Evaluate a completed LoRA adapter on the baseline's held-out cohort."""
import argparse
from decision_binder.evaluate import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment.yaml')
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--mode', choices=('primary', 'target_holdout'), default='primary')
    parser.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    parser.add_argument('--download-model', action='store_true')
    args = parser.parse_args()
    run(args.config, args.mode, args.download_model, args.device, args.adapter)


if __name__ == '__main__':
    main()
