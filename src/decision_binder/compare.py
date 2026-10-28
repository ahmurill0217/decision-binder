"""Reproduce base/adapter comparisons from saved probabilities; no training."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve, roc_curve

from .data import digest, write_json
from .metrics import choose_threshold, classification_metrics


def verify_cohort(predictions, source):
    if predictions.uuid.duplicated().any() or source.uuid.duplicated().any():
        raise ValueError("Duplicate evaluation UUID")
    left = predictions.set_index("uuid").sort_index()
    right = source.set_index("uuid").sort_index()
    if not left.index.equals(right.index):
        raise ValueError("Evaluation UUID cohort differs")
    label = "label" if "label" in right else "ground_truth_label"
    if not np.array_equal(left.ground_truth_label, right[label]) or not np.array_equal(left.target, right.target):
        raise ValueError("Evaluation labels/targets differ")


def verify_adapter(adapter, config, frames, root):
    metadata_path = adapter.parent / "metadata.json"
    meta = json.loads(metadata_path.read_text())
    ac = json.loads((adapter / "adapter_config.json").read_text())
    if meta["status"] != "passed" or meta["mode"] != "full" or not meta["adapter_reload_verified"]:
        raise ValueError("Require a completed full-training adapter with verified reload")
    if meta["model_identifier"] != config["model"]["id"] or meta["model_revision"] != config["model"]["revision"]:
        raise ValueError("Adapter base model mismatch")
    if ac["base_model_name_or_path"] != config["model"]["id"] or ac["revision"] != config["model"]["revision"]:
        raise ValueError("PEFT adapter base configuration mismatch")
    if meta["test_split_loaded"] or meta["config"]["sft"] != config["sft"]:
        raise ValueError("Training/test or prompt provenance mismatch")
    for key in ("dataset", "preprocessing", "splits", "model"):
        if meta["config"][key] != config[key]:
            raise ValueError(f"Adapter {key} configuration mismatch")
    for split in ("train", "validation"):
        ids = meta["run_uuids"][split]
        if len(ids) != len(set(ids)) or set(ids) != set(frames[split].uuid):
            raise ValueError("Adapter trained on a different split cohort")
    mode = meta["config"]["training"]["experiment"]
    report = root / config["paths"]["results"] / "sft" / mode / "report.json"
    local_hashes = {s: digest(root / "data/sft" / mode / f"{s}.parquet") for s in ("train", "validation")}
    source_matches = local_hashes == meta["source"]["input_sha256"]
    warnings = [] if source_matches else [
        "Colab training SFT file hashes differ from local SFT files. UUIDs and dataset/preprocessing/split/model/prompt-seed configurations match; exact training prompt byte equality is unverified. Evaluation uses the original local baseline cohort."]
    for warning in warnings:
        print(f"PROVENANCE WARNING: {warning}", flush=True)
    hashes = {p.name: digest(p) for p in adapter.iterdir() if p.is_file()}
    if hashes != meta["adapter_sha256"] or "adapter_model.safetensors" not in hashes:
        raise ValueError("Adapter checksum mismatch")
    return {"files_sha256": hashes, "training_metadata_sha256": digest(metadata_path),
            "training_sft_source": meta["source"], "local_sft_input_sha256": local_hashes,
            "local_sft_report_sha256": digest(report), "training_sft_files_match": source_matches,
            "warnings": warnings}


def comparison(root, mode="primary", training_metadata=None, config_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    config = yaml.safe_load((Path(config_path) if config_path else root / "configs/experiment.yaml").read_text())
    results = root / config["paths"]["results"]
    directories = [results / kind / mode for kind in ("zero_shot", "fine_tuned")]
    names = ["Base d1-3B", "Fine-tuned d1-3B"]
    tabular = results / "logistic_regression" / mode
    if (tabular / "metrics.json").exists():
        directories.append(tabular)
        names.append("Logistic regression")
    tables = [pd.read_parquet(p / "test_predictions.parquet") for p in directories]
    verify_cohort(tables[0], tables[1])
    train = pd.read_parquet(root / config["paths"]["processed"] / "splits" / mode / "train.parquet")
    split_dir = root / config["paths"]["processed"] / "splits" / mode
    source_test = pd.read_parquet(split_dir / "test.parquet")
    source_validation = pd.read_parquet(split_dir / "validation.parquet")
    majority = int(train.label.value_counts().sort_index().idxmax())
    reports = []
    for directory, table in zip(directories, tables):
        verify_cohort(table, source_test)
        validation = pd.read_parquet(directory / "validation_predictions.parquet")
        verify_cohort(validation, source_validation)
        threshold = json.loads((directory / "threshold.json").read_text())
        if threshold["selection_split"] != "validation" or threshold["test_used_for_selection"]:
            raise ValueError("Threshold must be selected on validation")
        expected = choose_threshold(validation.ground_truth_label, validation.binder_probability)
        if expected["threshold"] != threshold["threshold"]:
            raise ValueError("Saved threshold differs from validation selection rule")
        reports.append(classification_metrics(table.ground_truth_label, table.binder_probability, threshold["threshold"], majority))
    out = results / "comparison" / mode
    figures = results / "figures" / mode
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    metrics = ["roc_auc", "pr_auc_average_precision", "precision", "recall", "f1", "balanced_accuracy", "accuracy", "majority_class_accuracy", "threshold"]
    table = pd.DataFrame({"Metric": metrics, **{name: [r[m] for m in metrics] for name, r in zip(names, reports)}})
    table.to_csv(out / "metrics.csv", index=False)
    labels = ["ROC-AUC", "PR-AUC (average precision)", "Precision", "Recall", "F1",
              "Balanced accuracy", "Accuracy", "Majority-class accuracy", "Validation-selected threshold"]
    lines = ["# Generated held-out comparison", "", "Generated from saved predictions; do not edit numbers manually.", "",
             "| Metric | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for label, metric in zip(labels, metrics):
        values = ["undefined" if r[metric] is None else f"{r[metric]:.6f}" for r in reports]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    (out / "metrics.md").write_text("\n".join(lines) + "\n")
    manifests = [json.loads((d / "run_manifest.json").read_text()) for d in directories]
    report = {"metrics": dict(zip(names, reports)), "cohort_verified": True,
              "prediction_sha256": {name: digest(d / "test_predictions.parquet") for name, d in zip(names, directories)},
              "inference_hardware": {name: {k: m["provenance"][k] for k in ("device", "dtype")} for name, m in zip(names, manifests)},
              "provenance_warnings": manifests[1]["provenance"].get("adapter", {}).get("warnings", []),
              "model_protocol_notes": {name: m.get("protocol_note") for name, m in zip(names, manifests)
                                       if m.get("protocol_note")},
              "note": "Each model uses its own validation-selected threshold; PR-AUC means average precision. Hardware/precision differences may affect comparison."}
    write_json(out / "metrics.json", report)

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(figures / f"{name}.{ext}", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots()
    for name, frame, metric in zip(names, tables, reports):
        fpr, tpr, _ = roc_curve(frame.ground_truth_label, frame.binder_probability)
        ax.plot(fpr, tpr, label=f"{name}: AUC {metric['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out ROC comparison")
    ax.legend(); save(fig, "roc_comparison")
    fig, ax = plt.subplots()
    for name, frame, metric in zip(names, tables, reports):
        precision, recall, _ = precision_recall_curve(frame.ground_truth_label, frame.binder_probability)
        ax.step(recall, precision, where="post", label=f"{name}: AP {metric['pr_auc_average_precision']:.3f}")
    ax.axhline(tables[0].ground_truth_label.mean(), linestyle="--", color="gray", label="Binder prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Held-out precision–recall comparison")
    ax.legend(); save(fig, "pr_comparison")
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
    vmax = max(max(row) for r in reports for row in r["confusion_matrix"])
    for ax, name, r in zip(axes, names, reports):
        matrix = np.array(r["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > vmax/2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-binder", "Binder"], yticklabels=["Non-binder", "Binder"], xlabel="Predicted", ylabel="Actual", title=f"{name}\nThreshold {r['threshold']:.4f}")
    save(fig, "confusion_matrices")
    fig, ax = plt.subplots()
    for index, split in enumerate(("train", "validation", "test")):
        frame = pd.read_parquet(root / config["paths"]["processed"] / "splits" / mode / f"{split}.parquet")
        counts = [(frame.label == label).sum() for label in (0, 1)]
        for label, count, offset in zip(("Non-binder", "Binder"), counts, (-.2, .2)):
            bars = ax.bar(index + offset, count, width=.4, label=label if index == 0 else None, color="tab:blue" if label == "Non-binder" else "tab:orange")
            ax.bar_label(bars)
    ax.set(xticks=[0, 1, 2], xticklabels=["Train", "Validation", "Test"], ylabel="Candidates", title="Experimental class distribution")
    ax.legend(); save(fig, "class_distribution")
    if training_metadata:
        meta = json.loads(Path(training_metadata).read_text())
        if digest(Path(training_metadata)) != manifests[1]["provenance"]["adapter"]["training_metadata_sha256"]:
            raise ValueError("Loss history is not from the evaluated adapter's training run")
        fig, ax = plt.subplots()
        found = False
        for key, name in (("loss", "Training (logged averages)"), ("eval_loss", "Validation")):
            rows = [x for x in meta.get("log_history", []) if key in x and isinstance(x[key], (int, float)) and np.isfinite(x[key])]
            if rows:
                found = True
                ax.plot([x["step"] for x in rows], [x[key] for x in rows], marker="o", label=name)
        if found:
            ax.set(xlabel="Optimizer step attempts", ylabel="Completion-token cross-entropy", title="Training and validation loss")
            ax.legend(); save(fig, "training_loss")
        else:
            plt.close(fig)
    print(table.to_string(index=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--mode", choices=("primary", "target_holdout"), default="primary")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    from .evaluate import run
    if not args.plots_only:
        run(args.config, args.mode, args.download_model, args.device, args.adapter)
    comparison(Path(args.config).resolve().parent.parent, args.mode, Path(args.adapter).resolve().parent / "metadata.json",
               config_path=args.config)
