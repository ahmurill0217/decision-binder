"""Prepare features and create the reproducible primary (or target-holdout) splits."""
import argparse
from decision_binder.data import run
from decision_binder.splits import run as create_splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--mode", choices=("primary", "target_holdout"), default="primary")
    args = parser.parse_args()
    run(args.config)
    create_splits(args.config, args.mode)
