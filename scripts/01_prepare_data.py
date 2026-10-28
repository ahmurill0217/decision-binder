"""Run from an editable install: python scripts/01_prepare_data.py."""
import argparse
from decision_binder.data import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    run(parser.parse_args().config)
