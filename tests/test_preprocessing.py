import numpy as np
import pandas as pd
import pytest

from decision_binder.data import digest, load_tables, prepare, prepare_labels, related_sequences, write_json
from decision_binder.features import FEATURES, METRICS, aggregate_predictions, model_inputs, select_predictions


@pytest.fixture
def tables():
    designs = pd.DataFrame([
        {"uuid": f"00000000-0000-0000-0000-{i:012d}", "target": "EGFR",
         "binder_length": 10, "sequence": sequence, "generator": "test",
         "sequence_design_method": "test", "epitope_n_residues": 3,
         "adaptyv_binding": label, "adaptyv_kd_nM": 999, "binder_final": True}
        for i, (sequence, label) in enumerate([("AAAAAAAAAA", "binder"), ("AAAAAAAAAC", "non_binder"), ("CCCCCCCCCC", None)])
    ])
    predictions = pd.DataFrame([
        {"uuid": uuid, "target": "EGFR", "cofolding_model": "boltz2",
         "stoichiometry": "1to1", "target_form": "", "seed": f"seed_{seed}",
         **{metric: float(seed) for metric in METRICS}}
        for uuid in designs.uuid for seed in range(5)
    ])
    return designs, predictions


def test_reference_statistics_and_sample_std(tables):
    d, p = tables
    a = aggregate_predictions(select_predictions(p, d)).iloc[0]
    assert [a.sc_dockq_min, a.sc_dockq_max, a.sc_dockq_mean, a.sc_dockq_median] == [0, 4, 2, 2]
    assert a.sc_dockq_std == pytest.approx(np.sqrt(2.5))


def test_no_pooling_across_forms_stoichiometry_or_predictors(tables):
    d, p = tables
    extra = pd.concat([p.assign(target_form="alternate"), p.assign(stoichiometry="2to2"), p.assign(cofolding_model="other")])
    extra.loc[:, list(METRICS)] = 999
    selected = select_predictions(pd.concat([p, extra]), d)
    assert len(selected) == len(p)
    assert aggregate_predictions(selected).sc_dockq_mean.eq(2).all()
    assert len(select_predictions(pd.concat([p, extra]), d, "author_pooled")) == 3 * len(p)


def test_label_policy_never_invents_negative():
    d = pd.DataFrame({"adaptyv_binding": ["binder", "non_binder", "not_expressed", "not_tested", "inconclusive", None, "new_state"]})
    assert prepare_labels(d).dropna().tolist() == [1, 0]
    assert prepare_labels(d, "non_binder").dropna().tolist() == [1, 0, 0]
    with pytest.raises(ValueError):
        prepare_labels(d, "guess")


def test_complete_cases_and_metric_specific_nulls(tables):
    d, p = tables
    p.loc[0, "sc_dockq"] = np.nan
    records, audit, _, _ = prepare(d, p)
    assert len(records) == 2
    assert records.iloc[0].sc_dockq_mean == 2.5
    p.loc[p.uuid.eq(d.uuid.iloc[0]), "sc_dockq"] = np.nan
    records, audit, _, _ = prepare(d, p)
    assert len(records) == 1
    assert audit.exclusion_reason.tolist() == ["missing_aggregate", "included", "unresolved_or_excluded_label"]


def test_single_seed_std_and_absent_prediction_excluded(tables):
    d, p = tables
    p = p.loc[~p.uuid.eq(d.uuid.iloc[0]) | p.seed.eq("seed_0")]
    p = p.loc[~p.uuid.eq(d.uuid.iloc[2])]
    records, audit, _, _ = prepare(d, p)
    assert len(records) == 1
    assert audit.exclusion_reason.iloc[0] == "missing_aggregate"


def test_duplicate_keys_and_targets_fail(tables):
    d, p = tables
    with pytest.raises(ValueError, match="Duplicate prediction"):
        prepare(d, pd.concat([p, p.iloc[:1]]))
    with pytest.raises(ValueError, match="duplicate design UUID"):
        prepare(pd.concat([d, d.iloc[:1]]), p)
    p.loc[0, "target"] = "RBX1"
    with pytest.raises(ValueError, match="inconsistent target"):
        prepare(d, p)


def test_allowlist_resists_outcome_injection(tables):
    d, p = tables
    records, _, _, _ = prepare(d, p)
    assert len(FEATURES) == 85
    assert records.columns.tolist() == ["uuid", *FEATURES, "label"]
    injected = records.assign(adaptyv_binding="binder", twist_kd_nM=1, binder_final=True, vendor_agreement="both", sequence="secret", unknown_outcome=1)
    pd.testing.assert_frame_equal(model_inputs(records), model_inputs(injected))
    assert not {"uuid", "label", "adaptyv_binding", "binder_final"} & set(model_inputs(injected))


def test_near_duplicates_and_deterministic_order(tables):
    d, p = tables
    pairs = related_sequences(d)
    assert len(pairs) == 1
    assert pairs.identity.iloc[0] == 0.9
    a, *_ = prepare(d, p)
    b, *_ = prepare(d.sample(frac=1, random_state=1), p.sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(a, b)


def test_missing_schema_and_infinite_values_fail(tables):
    d, p = tables
    with pytest.raises(ValueError, match="Missing required"):
        prepare(d, p.drop(columns="ipsae_min"))
    p.loc[0, "ipsae_min"] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        prepare(d, p)


def test_cached_source_identity_and_checksums(tmp_path, tables):
    d, p = tables
    names = ("design_summary", "insilico_cofold_predictions")
    for name, frame in zip(names, (d, p)):
        frame.to_parquet(tmp_path / f"{name}.parquet")
    config = {"id": "test", "revision": "a" * 40, "split": "full"}
    write_json(tmp_path / "manifest.json", {"source": config, "sha256": {
        name: digest(tmp_path / f"{name}.parquet") for name in names}})
    loaded, _ = load_tables(config, tmp_path)
    assert len(loaded[0]) == len(d)
    with pytest.raises(ValueError, match="different source"):
        load_tables(config | {"revision": "b" * 40}, tmp_path)
    with (tmp_path / "design_summary.parquet").open("ab") as file:
        file.write(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_tables(config, tmp_path)


def test_missing_prediction_exclusion(tables):
    d, p = tables
    records, audit, _, _ = prepare(d, p.loc[~p.uuid.eq(d.uuid.iloc[0])])
    assert len(records) == 1
    assert audit.exclusion_reason.iloc[0] == "no_selected_prediction"
