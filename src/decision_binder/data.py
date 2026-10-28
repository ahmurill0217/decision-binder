"""Revision-pinned table loading, labels, audits, and preprocessing entry point."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
import platform
import re

import pandas as pd
import yaml

from .features import FEATURES, METADATA, METRICS, aggregate_predictions, require_columns, select_predictions

SUBSETS = ("design_summary", "insilico_cofold_predictions")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_tables(config, raw):
    """Only two named HF table configs; verify cached snapshots on reuse."""
    from datasets import load_dataset
    raw.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("dataset.revision must be an immutable 40-character commit SHA")
    manifest_path = raw / "manifest.json"
    identity = {k: config[k] for k in ("id", "revision", "split")}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source"] != identity:
            raise ValueError("Raw cache belongs to a different source; use a new raw directory")
        for name in SUBSETS:
            if digest(raw / f"{name}.parquet") != manifest["sha256"][name]:
                raise ValueError(f"Raw snapshot checksum mismatch: {name}")
    else:
        for name in SUBSETS:
            table = load_dataset(config["id"], name, split=config["split"], revision=config["revision"], cache_dir=str(raw / "hf_cache"))
            table.to_parquet(str(raw / f"{name}.parquet"))
        manifest = {"source": identity, "sha256": {name: digest(raw / f"{name}.parquet") for name in SUBSETS}}
        write_json(manifest_path, manifest)
    return [pd.read_parquet(raw / f"{name}.parquet") for name in SUBSETS], manifest


def validate_designs(designs):
    require_columns(designs, ["uuid", "sequence", "adaptyv_binding", *METADATA])
    if designs.uuid.isna().any() or designs.uuid.duplicated().any():
        raise ValueError("Missing or duplicate design UUID")
    if not designs.uuid.str.fullmatch(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}").all():
        raise ValueError("Malformed design UUID")
    if designs.sequence.isna().any() or not designs.sequence.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+").all():
        raise ValueError("Missing or noncanonical binder sequence")
    if not designs.sequence.str.len().eq(designs.binder_length).all():
        raise ValueError("Sequence length differs from binder_length")


def prepare_labels(designs, not_expressed="exclude"):
    if not_expressed not in ("exclude", "non_binder"):
        raise ValueError("not_expressed must be exclude or non_binder")
    mapping = {"binder": 1, "non_binder": 0}
    if not_expressed == "non_binder":
        mapping["not_expressed"] = 0
    labels = designs.adaptyv_binding.map(mapping).astype("Int64")
    return labels


def related_sequences(designs, threshold=0.9):
    """Audit exact and same-length high-identity pairs; not a homology clusterer."""
    if not 0 < threshold <= 1:
        raise ValueError("Identity threshold must be in (0, 1]")
    rows = []
    for _, group in designs.groupby(designs.sequence.str.len()):
        for a, b in combinations(group.itertuples(), 2):
            identity = sum(x == y for x, y in zip(a.sequence, b.sequence)) / len(a.sequence)
            if identity >= threshold:
                rows.append({"uuid_a": a.uuid, "uuid_b": b.uuid, "identity": identity,
                             "same_target": a.target == b.target, "exact": a.sequence == b.sequence})
    return pd.DataFrame(rows, columns=["uuid_a", "uuid_b", "identity", "same_target", "exact"])


def prepare(designs, predictions, not_expressed="exclude", scope="primary"):
    validate_designs(designs)
    selected = select_predictions(predictions, designs, scope)
    aggregate = aggregate_predictions(selected)
    audit = designs[["uuid", "adaptyv_binding"]].copy()
    audit["label"] = prepare_labels(designs, not_expressed)
    audit["exclusion_reason"] = "included"
    audit.loc[audit.label.isna(), "exclusion_reason"] = "unresolved_or_excluded_label"
    missing_metadata = designs[list(METADATA)].isna().any(axis=1)
    audit.loc[audit.label.notna() & missing_metadata, "exclusion_reason"] = "missing_metadata"
    eligible = designs.loc[audit.exclusion_reason.eq("included"), ["uuid", *METADATA]].copy()
    eligible["label"] = audit.loc[eligible.index, "label"].astype(int)
    joined = eligible.merge(aggregate, on="uuid", how="left", validate="one_to_one")
    absent = ~joined.uuid.isin(aggregate.uuid)
    incomplete = joined[list(FEATURES)].isna().any(axis=1)
    for mask, reason in ((incomplete & ~absent, "missing_aggregate"), (absent, "no_selected_prediction")):
        audit.loc[audit.uuid.isin(joined.loc[mask, "uuid"]), "exclusion_reason"] = reason
    processed = joined.loc[~incomplete, ["uuid", *FEATURES, "label"]].sort_values("uuid").reset_index(drop=True)
    if processed.empty:
        raise ValueError("No complete labeled designs remain")
    counts = {
        "designs_raw": len(designs), "predictions_raw": len(predictions),
        "predictions_boltz2": int(predictions.cofolding_model.eq("boltz2").sum()),
        "predictions_selected": len(selected), "aggregated_designs": len(aggregate),
        "resolved_labels": int(audit.label.notna().sum()), "complete_metadata_and_labels": len(eligible),
        "final_records": len(processed),
        "raw_label_distribution": designs.adaptyv_binding.fillna("<missing>").value_counts().to_dict(),
        "final_label_distribution": processed.label.value_counts().sort_index().to_dict(),
        "exclusions": audit.exclusion_reason.value_counts().to_dict(),
        "selected_rows_per_design": selected.groupby("uuid").size().value_counts().sort_index().to_dict(),
        "missing_selected_metrics": selected[list(METRICS)].isna().sum().to_dict(),
    }
    return processed, audit, counts, selected


def run(config_path):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    root = config_path.parent.parent
    paths = {name: root / path for name, path in config["paths"].items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    (designs, predictions), source = load_tables(config["dataset"], paths["raw"])
    options = config["preprocessing"]
    records, audit, counts, selected = prepare(designs, predictions, options["not_expressed"], options["prediction_scope"])
    related = related_sequences(designs, options["near_duplicate_identity"])
    related["both_retained"] = related.uuid_a.isin(records.uuid) & related.uuid_b.isin(records.uuid)
    records.to_parquet(paths["processed"] / "binders.parquet", index=False)
    records.to_csv(paths["processed"] / "binders.csv", index=False)
    audit.to_csv(paths["results"] / "exclusions.csv", index=False)
    related.to_csv(paths["results"] / "related_sequences.csv", index=False)
    selected.groupby(["target", "stoichiometry", "target_form"], dropna=False).size().rename("rows").to_csv(paths["results"] / "prediction_groups.csv")
    selected.groupby("uuid")[list(METRICS)].count().to_csv(paths["results"] / "metric_counts.csv")
    records.head(5).to_json(paths["results"] / "sample_records.json", orient="records", indent=2, double_precision=15)
    # Preprocessing-only historical comparator: never used as the main feature table.
    _, _, original_counts, _ = prepare(designs, predictions, "exclude", "author_pooled")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source": source,
              "config": config, "counts": counts, "author_pooled_comparison": original_counts,
              "features": list(FEATURES), "related_sequence_pairs": len(related),
              "exact_sequence_pairs": int(related.exact.sum()),
              "retained_related_pairs": int(related.both_retained.sum()),
              "python": platform.python_version(),
              "versions": {p: version(p) for p in ("datasets", "huggingface-hub", "pandas", "numpy", "pyarrow", "PyYAML")},
              "code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
              "processed_sha256": digest(paths["processed"] / "binders.parquet")}
    write_json(paths["results"] / "preprocessing_report.json", report)
    write_json(paths["processed"] / "feature_columns.json", list(FEATURES))
    print(json.dumps(counts, indent=2))
    return report
