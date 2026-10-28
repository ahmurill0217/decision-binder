import numpy as np
import pandas as pd
import pytest
from decision_binder.features import FEATURES
from decision_binder.tabular import CATEGORICAL, NUMERIC, fit_pipeline, predict


@pytest.fixture
def cohort():
    rows = []
    for i in range(12):
        row = {name: float(i) / 10 for name in FEATURES}
        row.update(uuid=f'id-{i}', label=int(i >= 6), target='X', generator='G', sequence_design_method='M')
        rows.append(row)
    return pd.DataFrame(rows)


def test_train_only_scaling_unknown_categories_and_outcome_exclusion(cohort):
    options = dict(C=1., max_iter=2000, tol=.0001, seed=1337)
    pipeline = fit_pipeline(cohort, options)
    transform = pipeline.named_steps['features']
    scaler = transform.named_transformers_['numeric']
    assert np.allclose(scaler.mean_, cohort[list(NUMERIC)].mean())
    mean = scaler.mean_.copy()
    unseen = cohort.copy()
    unseen.loc[:, list(NUMERIC)] = 1000.
    unseen['target'] = 'unseen-target'
    unseen['label'] = 1 - unseen.label
    unseen['adaptyv_binding'] = 'OUTCOME_SENTINEL'
    before = predict(pipeline, unseen, 'validation')
    after = predict(pipeline, unseen.assign(label=1-unseen.label), 'validation')
    assert np.array_equal(before.binder_probability, after.binder_probability)
    assert np.array_equal(mean, scaler.mean_)
    assert 'unseen-target' not in transform.named_transformers_['categorical'].categories_[0]
    assert set(CATEGORICAL) | set(NUMERIC) == set(FEATURES)


def test_deterministic_fit_and_positive_class(cohort):
    options = dict(C=1., max_iter=2000, tol=.0001, seed=1337)
    first, second = fit_pipeline(cohort, options), fit_pipeline(cohort, options)
    a, b = predict(first, cohort, 'test'), predict(second, cohort, 'test')
    np.testing.assert_array_equal(a.binder_probability, b.binder_probability)
    assert a.binder_probability.iloc[-1] > a.binder_probability.iloc[0]
    assert list(first.named_steps['classifier'].classes_) == [0, 1]
    with pytest.raises(ValueError, match='Duplicate'):
        fit_pipeline(pd.concat([cohort, cohort.iloc[:1]]), options)
