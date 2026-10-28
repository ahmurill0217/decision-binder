"""Fixed logistic-regression baseline with train-only preprocessing and fitting."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
import yaml

from .compare import verify_cohort
from .data import digest, write_json
from .features import FEATURES, model_inputs
from .metrics import choose_threshold, classification_metrics
from .splits import NAMES, validate_assignment

CATEGORICAL = ('target', 'generator', 'sequence_design_method')
NUMERIC = tuple(x for x in FEATURES if x not in CATEGORICAL)
MODEL_ID = 'sklearn/LogisticRegression'


def make_pipeline(options):
    transform = ColumnTransformer([
        ('numeric', StandardScaler(), list(NUMERIC)),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(CATEGORICAL)),
    ], remainder='drop')
    model = LogisticRegression(C=options['C'], solver='lbfgs', l1_ratio=0.0,
                               max_iter=options['max_iter'], tol=options['tol'],
                               class_weight=None, random_state=options['seed'])
    return Pipeline([('features', transform), ('classifier', model)])


def fit_pipeline(train, options):
    if set(train.label) != {0, 1}:
        raise ValueError('Training requires both binary labels')
    if train.uuid.duplicated().any():
        raise ValueError('Duplicate training UUID')
    pipeline = make_pipeline(options)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('error', ConvergenceWarning)
        pipeline.fit(model_inputs(train), train.label.to_numpy())
    return pipeline


def predict(pipeline, frame, split):
    classes = list(pipeline.named_steps['classifier'].classes_)
    if set(classes) != {0, 1}:
        raise ValueError('Expected classifier labels 0 and 1')
    with threadpool_limits(limits=1):
        p = pipeline.predict_proba(model_inputs(frame))[:, classes.index(1)]
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid classifier probabilities')
    return pd.DataFrame({'uuid': frame.uuid.to_numpy(), 'ground_truth_label': frame.label.to_numpy(),
                         'target': frame.target.to_numpy(), 'binder_probability': p,
                         'split': split, 'model_identifier': MODEL_ID})


def run(config_path, mode='primary'):
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text())
    options = config['tabular']
    results = root / config['paths']['results']
    processed = root / config['paths']['processed']
    source = processed / 'splits' / mode
    report_path = results / 'splits' / mode / 'report.json'
    split_report = json.loads(report_path.read_text())
    if split_report['mode'] != mode or split_report['dataset'] != config['dataset'] or split_report['options'] != config['splits']:
        raise ValueError('Split configuration mismatch')
    if digest(processed / 'binders.parquet') != split_report['input_sha256']:
        raise ValueError('Processed cohort checksum mismatch')
    for filename in [*(f'{s}.parquet' for s in NAMES), 'assignments.parquet']:
        if digest(source / filename) != split_report['output_sha256'][filename]:
            raise ValueError('Split checksum mismatch')
    frames = {s: pd.read_parquet(source / f'{s}.parquet').sort_values('uuid').reset_index(drop=True) for s in NAMES}
    validate_assignment(pd.concat(frames.values()), pd.read_parquet(source / 'assignments.parquet'), mode)
    # Only identity/labels/targets are checked; baseline scores never enter fitting.
    for split in ('validation', 'test'):
        verify_cohort(pd.read_parquet(results / 'zero_shot' / mode / f'{split}_predictions.parquet'), frames[split])
    output = results / 'logistic_regression' / mode
    artifact = root / 'artifacts' / 'tabular' / mode
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {output}; preserve it and choose a new results path')
    if (artifact / 'pipeline.joblib').exists():
        raise FileExistsError(f'Refusing to overwrite fitted pipeline in {artifact}')
    output.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True, exist_ok=True)
    manifest = {'model_identifier': MODEL_ID, 'config': config, 'mode': mode,
                'features': list(FEATURES), 'numeric_features': list(NUMERIC), 'categorical_features': list(CATEGORICAL),
                'preprocessing_fit_split': 'train', 'model_fit_split': 'train', 'hyperparameter_search': False,
                'method': 'StandardScaler numeric + one-hot categorical, unknown categories ignored; L2 logistic regression, C fixed, no class weighting',
                'protocol_note': 'Baseline added after d1 test results were known; no test-based hyperparameter/feature search. Not a preregistered independent replication.',
                'split_report_sha256': digest(report_path),
                'input_sha256': {s: digest(source / f'{s}.parquet') for s in NAMES},
                'training_uuids': frames['train'].uuid.tolist(),
                'provenance': {'device': 'cpu', 'dtype': 'float64', 'platform': platform.platform(), 'machine': platform.machine(),
                               'threads': 1, 'versions': {p: version(p) for p in ('scikit-learn', 'numpy', 'pandas', 'joblib')}},
                'code_sha256': {p: digest(Path(__file__).with_name(p)) for p in ('tabular.py', 'features.py', 'metrics.py')}}
    write_json(output / 'run_manifest.json', manifest)
    start = time.monotonic()
    pipeline = fit_pipeline(frames['train'], options)
    elapsed = time.monotonic() - start
    validation = predict(pipeline, frames['validation'], 'validation')
    selection = choose_threshold(validation.ground_truth_label, validation.binder_probability)
    selection['test_used_for_selection'] = False
    # Lock the threshold before computing any test probabilities/metrics.
    write_json(output / 'threshold.json', selection)
    test = predict(pipeline, frames['test'], 'test')
    majority = int(frames['train'].label.value_counts().sort_index().idxmax())
    metrics = {}
    for split, frame in [('validation', validation), ('test', test)]:
        frame['threshold'] = selection['threshold']
        frame['predicted_class'] = (frame.binder_probability >= selection['threshold']).astype(int)
        frame.to_parquet(output / f'{split}_predictions.parquet', index=False)
        frame.to_csv(output / f'{split}_predictions.csv', index=False)
        metrics[split] = classification_metrics(frame.ground_truth_label, frame.binder_probability, selection['threshold'], majority)
        metrics[split]['fixed_0_5_comparison'] = classification_metrics(frame.ground_truth_label, frame.binder_probability, .5, majority)
    joblib.dump(pipeline, artifact / 'pipeline.joblib')
    reloaded = joblib.load(artifact / 'pipeline.joblib')  # Only our freshly written trusted artifact.
    if not np.array_equal(test.binder_probability, predict(reloaded, frames['test'], 'test').binder_probability):
        raise ValueError('Reload changed probabilities')
    report = {'model_identifier': MODEL_ID, 'mode': mode, 'threshold_selection': selection, 'metrics': metrics,
              'fit_seconds': elapsed, 'iterations': pipeline.named_steps['classifier'].n_iter_.tolist(),
              'expanded_feature_count': len(pipeline.named_steps['features'].get_feature_names_out()),
              'pipeline_sha256': digest(artifact / 'pipeline.joblib'), 'reload_verified': True,
              'run_manifest_sha256': digest(output / 'run_manifest.json')}
    write_json(output / 'metrics.json', report)
    print(json.dumps(metrics['test'], indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment.yaml')
    parser.add_argument('--mode', choices=('primary', 'target_holdout'), default='primary')
    args = parser.parse_args()
    run(args.config, args.mode)
