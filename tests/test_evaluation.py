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
