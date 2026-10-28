"""Seeded group allocation; primary and unseen-target experiments stay separate."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import digest, related_sequences, validate_designs, write_json
from .features import FEATURES, require_columns

NAMES = ("train", "validation", "test")


def components(nodes, edges):
    """Deterministic transitive components, identified by their smallest member."""
    parent = {node: node for node in sorted(nodes)}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in edges:
        x, y = find(a), find(b)
        parent[max(x, y)] = min(x, y)
    return {node: find(node) for node in parent}


def sequence_groups(designs, threshold):
    # Include excluded designs as bridges: A--excluded B--C still forms one group.
    pairs = related_sequences(designs.sort_values("uuid"), threshold)
    return components(designs.uuid, pairs[["uuid_a", "uuid_b"]].itertuples(index=False, name=None))


def allocate(records, groups, fractions, seed, trials=64, balance_labels=True):
    """Greedy multistart allocation minimizes squared count deviations.

    Primary balances class counts; target holdout balances row counts only.
    All trials use the same declared objective, never model performance.
    """
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (3,) or not np.isfinite(fractions).all() or (fractions <= 0).any() or not np.isclose(fractions.sum(), 1):
        raise ValueError("Three positive fractions summing to one are required")
    if not isinstance(trials, int) or trials < 1:
        raise ValueError("search_trials must be a positive integer")
    frame = records[["uuid", "label"]].copy()
    frame["group"] = frame.uuid.map(groups)
    if frame.group.isna().any():
        raise ValueError("Missing group assignment")
    if balance_labels:
        counts = pd.crosstab(frame.group, frame.label).reindex(columns=[0, 1], fill_value=0)
    else:
        counts = frame.groupby("group").size().to_frame("rows")
    counts = counts.sort_index()
    if len(counts) < 3:
        raise ValueError("At least three independent groups are required")
    values = counts.to_numpy(dtype=float)
    desired = fractions[:, None] * values.sum(axis=0)
    scale = np.maximum(values.sum(axis=0), 1)

    def loss(actual):
        return float(np.square((actual - desired) / scale).sum())

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(trials):
        # Randomize whole groups, not rows. Size sorting would systematically
        # send the largest related-design families to training first.
        order = rng.permutation(len(values))
        actual = np.zeros_like(desired)
        assignment = np.empty(len(values), dtype=int)
        occupied = np.zeros(3, dtype=int)
        for position, index in enumerate(order):
            empty = np.flatnonzero(occupied == 0)
            choices = empty if len(order) - position == len(empty) else range(3)
            scores = []
            for split in choices:
                trial = actual.copy()
                trial[split] += values[index]
                scores.append((loss(trial), split))
            _, chosen = min(scores)
            assignment[index] = chosen
            actual[chosen] += values[index]
            occupied[chosen] += 1
        # Prefer both classes in all splits if a trial achieves this.
        missing = int((actual == 0).sum()) if balance_labels else 0
        key = (missing, loss(actual))
        if best is None or key < best[0]:
            best = (key, assignment.copy())
    group_split = dict(zip(counts.index, (NAMES[i] for i in best[1])))
    return frame.uuid.map(lambda uuid: group_split[groups[uuid]])


def create_assignment(records, groups, options, mode):
    if records.uuid.isna().any() or records.uuid.duplicated().any():
        raise ValueError("Missing or duplicate UUID in processed data")
    if records.label.isna().any() or not records.label.isin([0, 1]).all():
        raise ValueError("Labels must be binary and nonmissing")
    if records.target.isna().any():
        raise ValueError("Missing target")
    records = records.sort_values("uuid").reset_index(drop=True)
    assignments = records[["uuid", "target", "label"]].copy()
    assignments["sequence_group"] = records.uuid.map(groups)
    if assignments.sequence_group.isna().any():
        raise ValueError("Processed UUID absent from sequence groups")
    if mode == "primary":
        allocation_groups = groups
    elif mode == "target_holdout":
        # Union targets connected by a sequence component, including bridges in
        # excluded rows; one component must never cross a target holdout boundary.
        edges = []
        for _, group in assignments.groupby("sequence_group"):
            targets = sorted(group.target.unique())
            edges.extend((targets[0], target) for target in targets[1:])
        target_groups = components(records.target.unique(), edges)
        allocation_groups = dict(zip(records.uuid, records.target.map(target_groups)))
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    explicit = options.get("target_holdout", {}) if mode == "target_holdout" else {}
    val, test = set(explicit.get("validation_targets", [])), set(explicit.get("test_targets", []))
    if val or test:
        if not val or not test or val & test or (val | test) - set(records.target):
            raise ValueError("Specify nonempty, disjoint, known validation and test targets")
        assignments["split"] = records.target.map(lambda t: "validation" if t in val else "test" if t in test else "train")
    else:
        assignments["split"] = allocate(records, allocation_groups, options["fractions"], options["seed"], options["search_trials"], mode == "primary")
    validate_assignment(records, assignments, mode)
    return assignments


def validate_assignment(records, assignments, mode):
    if assignments.uuid.duplicated().any() or set(records.uuid) != set(assignments.uuid):
        raise ValueError("Split UUIDs must form a complete, disjoint partition")
    if assignments.split.isna().any() or set(assignments.split) != set(NAMES):
        raise ValueError("All three splits must be nonempty")
    if assignments.groupby("sequence_group").split.nunique().gt(1).any():
        raise ValueError("Related sequence component crosses splits")
    if mode == "target_holdout" and assignments.groupby("target").split.nunique().gt(1).any():
        raise ValueError("Target crosses target-holdout splits")


def summarize(assignments):
    rows = []
    targets = []
    warnings = []
    for split in NAMES:
        part = assignments.loc[assignments.split.eq(split)]
        binders = int(part.label.sum())
        rows.append({"split": split, "records": len(part), "fraction": len(part) / len(assignments),
                     "binders": binders, "non_binders": len(part) - binders,
                     "binder_fraction": binders / len(part), "targets": part.target.nunique(),
                     "sequence_groups": part.sequence_group.nunique()})
        if part.label.nunique() < 2:
            warnings.append(f"{split} contains only one class; some evaluation metrics will be undefined")
        for target in sorted(assignments.target.unique()):
            subset = part.loc[part.target.eq(target)]
            targets.append({"split": split, "target": target, "records": len(subset),
                            "binders": int(subset.label.sum()), "non_binders": int(subset.label.eq(0).sum())})
    return rows, targets, warnings


def run(config_path, mode="primary"):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    root = config_path.parent.parent
    processed = root / config["paths"]["processed"]
    raw = root / config["paths"]["raw"]
    results = root / config["paths"]["results"]
    source_report = json.loads((results / "preprocessing_report.json").read_text())
    if digest(processed / "binders.parquet") != source_report["processed_sha256"]:
        raise ValueError("Processed data differs from preprocessing manifest")
    if digest(raw / "design_summary.parquet") != source_report["source"]["sha256"]["design_summary"]:
        raise ValueError("Raw design snapshot differs from preprocessing manifest")
    if source_report["source"]["source"] != config["dataset"]:
        raise ValueError("Configured dataset differs from preprocessing manifest")
    if source_report["config"]["preprocessing"] != config["preprocessing"]:
        raise ValueError("Preprocessing config changed; rerun preprocessing first")
    records = pd.read_parquet(processed / "binders.parquet")
    require_columns(records, ["uuid", "label", *FEATURES])
    designs = pd.read_parquet(raw / "design_summary.parquet")
    validate_designs(designs)
    groups = sequence_groups(designs, config["preprocessing"]["near_duplicate_identity"])
    assignments = create_assignment(records, groups, config["splits"], mode)
    summary, targets, warnings = summarize(assignments)
    output = processed / "splits" / mode
    report_dir = results / "splits" / mode
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for split in NAMES:
        ids = assignments.loc[assignments.split.eq(split), "uuid"]
        records.loc[records.uuid.isin(ids)].sort_values("uuid").to_parquet(output / f"{split}.parquet", index=False)
    # Audit columns live separately, not in model input Parquets.
    assignments.to_parquet(output / "assignments.parquet", index=False)
    assignments.to_csv(report_dir / "assignments.csv", index=False)
    pd.DataFrame(summary).to_csv(report_dir / "class_balance.csv", index=False)
    pd.DataFrame(targets).to_csv(report_dir / "target_representation.csv", index=False)
    group_sizes = assignments.groupby("sequence_group").size()
    report = {"mode": mode, "options": config["splits"], "dataset": config["dataset"],
              "preprocessing_report_sha256": digest(results / "preprocessing_report.json"),
              "input_sha256": digest(processed / "binders.parquet"),
              "source_designs_sha256": digest(raw / "design_summary.parquet"),
              "code_sha256": {name: digest(Path(__file__).with_name(name)) for name in ("splits.py", "data.py")},
              "sequence_identity_threshold": config["preprocessing"]["near_duplicate_identity"],
              "grouping": "transitive equal-length position-wise identity; all source designs included as bridges",
              "multi_record_groups": int(group_sizes.gt(1).sum()), "largest_group": int(group_sizes.max()),
              "summary": summary, "target_representation": targets, "warnings": warnings,
              "checks": {"uuid_disjoint": True, "complete_partition": True, "sequence_groups_disjoint": True,
                         "targets_disjoint": mode == "target_holdout"},
              "output_sha256": {p.name: digest(p) for p in sorted(output.glob("*.parquet"))}}
    write_json(report_dir / "report.json", report)
    print(pd.DataFrame(summary).to_string(index=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--mode", choices=("primary", "target_holdout"), default="primary")
    args = parser.parse_args()
    run(args.config, args.mode)
