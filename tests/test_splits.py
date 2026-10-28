import numpy as np
import pandas as pd
import pytest

from decision_binder.splits import (
    allocate, components, create_assignment, sequence_groups, summarize, validate_assignment,
)


@pytest.fixture
def cohort():
    records = pd.DataFrame({"uuid": [f"id{i:03d}" for i in range(100)],
                            "label": [i % 4 == 0 for i in range(100)],
                            "target": [f"target{i // 10}" for i in range(100)]})
    records.label = records.label.astype(int)
    groups = components(records.uuid, [("id000", "id001"), ("id001", "id002")])
    options = {"fractions": [.7, .15, .15], "seed": 1337, "search_trials": 16,
               "target_holdout": {"validation_targets": [], "test_targets": []}}
    return records, groups, options


def test_reproducibility_partition_and_balance(cohort):
    records, groups, options = cohort
    first = create_assignment(records, groups, options, "primary")
    second = create_assignment(records.sample(frac=1, random_state=4), groups, options, "primary")
    pd.testing.assert_frame_equal(first, second)
    assert first.uuid.nunique() == len(records)
    assert first.groupby("sequence_group").split.nunique().max() == 1
    summary, _, _ = summarize(first)
    for part, fraction in zip(summary, options["fractions"]):
        assert abs(part["records"] - len(records) * fraction) <= 2
        assert abs(part["binders"] - records.label.sum() * fraction) <= 1
    other = create_assignment(records, groups, options | {"seed": 12}, "primary")
    assert not other.split.equals(first.split)


def test_transitive_groups_include_excluded_bridges():
    designs = pd.DataFrame({"uuid": ["A", "B", "C", "D"], "target": ["x"] * 4,
                            "sequence": ["AAAAAAAAAA", "AAAAAAAAAC", "AAAAAAAACC", "CCCCCCCCCC"]})
    groups = sequence_groups(designs, .9)
    # A and C have only 80% identity, but B connects them even if later excluded.
    assert groups["A"] == groups["C"] != groups["D"]


def test_target_holdout_and_label_independent_selection(cohort):
    records, groups, options = cohort
    first = create_assignment(records, groups, options, "target_holdout")
    assert first.groupby("target").split.nunique().max() == 1
    changed = create_assignment(records.assign(label=1 - records.label), groups, options, "target_holdout")
    assert first.split.equals(changed.split)
    explicit = options | {"target_holdout": {"validation_targets": ["target1"], "test_targets": ["target2"]}}
    result = create_assignment(records, groups, explicit, "target_holdout")
    assert set(result.loc[result.split.eq("test"), "target"]) == {"target2"}
    assert set(result.loc[result.split.eq("validation"), "target"]) == {"target1"}


def test_cross_target_relatives_not_separated(cohort):
    records, groups, options = cohort
    groups["id010"] = groups["id020"] = "cross_target"
    automatic = create_assignment(records, groups, options, "target_holdout")
    assert automatic.loc[automatic.target.isin(["target1", "target2"]), "split"].nunique() == 1
    explicit = options | {"target_holdout": {"validation_targets": ["target1"], "test_targets": ["target2"]}}
    with pytest.raises(ValueError, match="component crosses"):
        create_assignment(records, groups, explicit, "target_holdout")


def test_bad_inputs_and_impossible_groups(cohort):
    records, groups, options = cohort
    with pytest.raises(ValueError, match="duplicate UUID"):
        create_assignment(pd.concat([records, records.iloc[:1]]), groups, options, "primary")
    with pytest.raises(ValueError, match="binary"):
        create_assignment(records.assign(label=np.nan), groups, options, "primary")
    with pytest.raises(ValueError, match="fractions"):
        allocate(records, groups, [.7, .2, .2], 1337)
    with pytest.raises(ValueError, match="three independent"):
        allocate(records, dict.fromkeys(records.uuid, "only_group"), [.7, .15, .15], 1337)
    with pytest.raises(ValueError, match="absent"):
        create_assignment(records, {}, options, "primary")


def test_single_class_and_rare_class_reported(cohort):
    records, groups, options = cohort
    assignment = create_assignment(records.assign(label=0), groups, options, "primary")
    summary, targets, warnings = summarize(assignment)
    assert len(warnings) == 3
    assert sum(row["records"] for row in summary) == 100
    assert len(targets) == 30
    rare = records.assign(label=0)
    rare.loc[99, "label"] = 1
    assignment = create_assignment(rare, groups, options, "primary")
    assert len(summarize(assignment)[2]) == 2


def test_validation_rejects_overlap_and_missing_members(cohort):
    records, groups, options = cohort
    assignment = create_assignment(records, groups, options, "primary")
    with pytest.raises(ValueError, match="partition"):
        validate_assignment(records, assignment.iloc[1:], "primary")
    assignment.loc[assignment.uuid.eq("id000"), "split"] = "test"
    assignment.loc[assignment.uuid.eq("id001"), "split"] = "train"
    with pytest.raises(ValueError, match="component crosses"):
        validate_assignment(records, assignment, "primary")


@pytest.mark.parametrize("validation,test", [(["unknown"], ["target1"]),
                                           (["target1"], ["target1"]),
                                           ([], ["target1"])])
def test_invalid_explicit_targets(cohort, validation, test):
    records, groups, options = cohort
    options = options | {"target_holdout": {"validation_targets": validation, "test_targets": test}}
    with pytest.raises(ValueError, match="disjoint, known"):
        create_assignment(records, groups, options, "target_holdout")
