import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="Install the evaluation extra for metric tests")
from decision_binder.evaluate import binder_probability, score_frame
from decision_binder.metrics import choose_threshold, classification_metrics
from decision_binder.prompts import QUESTION


def test_metrics_known_ranking_and_confusion():
    result = classification_metrics([0, 0, 1, 1], [.1, .4, .35, .8], .5, 0)
    assert result["roc_auc"] == .75
    assert result["pr_auc_average_precision"] == pytest.approx(5 / 6)
    assert result["precision"] == 1
    assert result["recall"] == .5
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["balanced_accuracy"] == .75
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]
    assert result["majority_class_accuracy"] == .5


def test_validation_threshold_and_ties():
    selection = choose_threshold([0, 0, 1, 1], [.01, .1, .2, .3])
    assert selection["threshold"] == .2
    assert selection["validation_objective"] == 1
    assert selection["selection_split"] == "validation"
    assert choose_threshold([0, 0, 1, 1], [.1, .4, .35, .8])["threshold"] == .5
    # All decisions tie: use the prespecified 0.5 rather than chasing recall.
    assert choose_threshold([0, 1], [.8, .8])["threshold"] == .5
    with pytest.raises(ValueError, match="both classes"):
        choose_threshold([0, 0], [.1, .2])


def test_single_class_and_no_positive_predictions():
    result = classification_metrics([0, 1], [.1, .2], .9, 0)
    assert result["precision"] == result["recall"] == result["f1"] == 0
    result = classification_metrics([0, 0], [.1, .2], .5, 0)
    assert result["roc_auc"] is None
    assert result["pr_auc_average_precision"] is None
    assert result["balanced_accuracy"] is None
    assert result["majority_class_accuracy"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("probabilities", [[float("nan")], [1.01], [-.1], []])
def test_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        classification_metrics([0], probabilities, .5, 0)


def response(p):
    return {"answers": {"binding": {"type": "choice", "choice": "Binder" if p >= .5 else "Non-binder",
                                   "probabilities": {"Non-binder": 1-p, "Binder": p}}},
            "usage": {"output_tokens": 0, "input_tokens": 100}}


def test_probability_is_selected_by_label_not_position():
    assert binder_probability(response(.73)) == .73
    invalid = response(.5)
    invalid["answers"]["binding"]["probabilities"]["Binder"] = .8
    with pytest.raises(ValueError, match="Invalid native"):
        binder_probability(invalid)


def test_decision_checkpoint_resume_and_prompt_parity(tmp_path, monkeypatch):
    import decision_binder.evaluate as evaluation
    calls = []
    frame = pd.DataFrame({"uuid": ["a", "b"], "target": ["X", "Y"], "label": [1, 0]})
    def build(record, *_):
        return "same prompt", {"target": record["target"]}, {"Non-binder": {}, "Binder": {}}
    monkeypatch.setattr(evaluation, "build_prompt", build)
    def decide(state, questions):
        assert "label" not in state
        assert questions["binding"]["instructions"] == QUESTION
        calls.append(state)
        return response(.73)
    model = SimpleNamespace(engine=SimpleNamespace(render=lambda *_: "same prompt"), system_one=decide)
    native = SimpleNamespace(as_question=lambda q: q)
    path = tmp_path / "decisions.jsonl"
    args = (frame, "validation", model, None, None, native, 1337, "model", "revision", path)
    first = score_frame(*args)
    second = score_frame(*args)
    pd.testing.assert_frame_equal(first, second)
    assert len(calls) == 2
    assert first.binder_probability.eq(.73).all()
    path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        score_frame(*args)


def test_comparison_requires_exact_uuid_labels_targets():
    from decision_binder.compare import verify_cohort
    predictions = pd.DataFrame({'uuid': ['a', 'b'], 'ground_truth_label': [0, 1], 'target': ['X', 'Y']})
    source = predictions.rename(columns={'ground_truth_label': 'label'})
    verify_cohort(predictions.iloc[::-1], source)
    with pytest.raises(ValueError, match='UUID cohort'):
        verify_cohort(predictions.iloc[:1], source)
    changed = source.copy()
    changed.loc[0, 'label'] = 1
    with pytest.raises(ValueError, match='labels/targets'):
        verify_cohort(predictions, changed)


def test_comparison_reproduces_metrics_and_all_figures(tmp_path):
    import yaml
    from decision_binder.compare import comparison
    from decision_binder.data import digest, write_json
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/experiment.yaml').write_text(yaml.safe_dump({'paths': {'results': 'results', 'processed': 'data/processed'}}))
    splits = tmp_path / 'data/processed/splits/primary'
    splits.mkdir(parents=True)
    source = pd.DataFrame({'uuid': ['a', 'b', 'c', 'd'], 'label': [0, 0, 1, 1], 'target': ['X'] * 4})
    for split in ('train', 'validation', 'test'):
        source.to_parquet(splits / f'{split}.parquet')
    meta = tmp_path / 'metadata.json'
    write_json(meta, {'log_history': [{'step': 1, 'loss': 1.2}, {'step': 2, 'eval_loss': .5}]})
    for kind, scores in [('zero_shot', [.1, .4, .35, .8]), ('fine_tuned', [.1, .2, .8, .9]),
                         ('logistic_regression', [.2, .1, .7, .9])]:
        folder = tmp_path / 'results' / kind / 'primary'
        folder.mkdir(parents=True)
        frame = source.rename(columns={'label': 'ground_truth_label'}).assign(binder_probability=scores)
        for split in ('test', 'validation'):
            frame.to_parquet(folder / f'{split}_predictions.parquet')
        threshold = choose_threshold(frame.ground_truth_label, scores)
        threshold['test_used_for_selection'] = False
        write_json(folder / 'threshold.json', threshold)
        write_json(folder / 'run_manifest.json', {'provenance': {'device': 'cpu', 'dtype': 'float32', 'adapter': {'training_metadata_sha256': digest(meta)}}})
        if kind == 'logistic_regression':
            write_json(folder / 'metrics.json', {'completed': True})
    alternate = tmp_path / 'configs/alternate.yaml'
    (tmp_path / 'configs/experiment.yaml').rename(alternate)
    report = comparison(tmp_path, training_metadata=meta, config_path=alternate)
    assert report['metrics']['Base d1-3B']['roc_auc'] == .75
    assert report['metrics']['Fine-tuned d1-3B']['roc_auc'] == 1
    assert report['metrics']['Logistic regression']['roc_auc'] == 1
    generated = (tmp_path / 'results/comparison/primary/metrics.md').read_text()
    assert '| ROC-AUC | 0.750000 | 1.000000 |' in generated
    assert len(list((tmp_path / 'results/figures/primary').glob('*.png'))) == 5
    assert len(list((tmp_path / 'results/figures/primary').glob('*.svg'))) == 5
