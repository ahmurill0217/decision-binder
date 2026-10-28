"""Local base-model decision evaluation, validation threshold first, test second."""
import argparse
import importlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import time

import pandas as pd
import yaml

from .data import digest, write_json
from .metrics import choose_threshold, classification_metrics
from .prompts import ASSET_FILES, QUESTION, build_prompt, load_prompt_assets
from .splits import NAMES, validate_assignment

MODEL_FILES = (*ASSET_FILES, "config.json", "modeling_d1.py", "hybrid.py", "lfm2_vl.py", "runner.py", "api.py", "model.safetensors")


def binder_probability(response):
    answer = response["answers"]["binding"]
    probs = answer["probabilities"]
    if answer["type"] != "choice" or set(probs) != {"Binder", "Non-binder"}:
        raise ValueError("Unexpected native decision response")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()) or not math.isclose(sum(probs.values()), 1, abs_tol=1e-6):
        raise ValueError("Invalid native probabilities")
    if response["usage"]["output_tokens"] != 0:
        raise ValueError("Expected decision readout, not generated text")
    return float(probs["Binder"])


def load_base(config, root, download=False, device="auto"):
    import torch
    from transformers import AutoModel
    directory = root / config["paths"]["raw"] / "d1_base"
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(config["model"]["id"], revision=config["model"]["revision"],
                          allow_patterns=list(MODEL_FILES), local_dir=str(directory))
    if any(not (directory / name).exists() for name in MODEL_FILES):
        raise FileNotFoundError("Missing pinned base checkpoint; run with --download-model")
    tokenizer, utility = load_prompt_assets(config["model"], directory)
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    torch.manual_seed(config["splits"]["seed"])
    # Loading local files guarantees all code and weights come from one snapshot.
    model = AutoModel.from_pretrained(str(directory), trust_remote_code=True, local_files_only=True, dtype=dtype).to(device).eval()
    model.requires_grad_(False)
    # Use the model's supported API engine; replace only its tokenizer with the
    # already hash-verified tokenizer. No calibration or prompt overrides.
    engine = model.engine
    engine.tokenizer = tokenizer
    if engine.calibration is not None:
        raise ValueError("Base evaluation must not fit or load a calibration")
    native_prompt = importlib.import_module(type(engine).__module__.rsplit(".", 1)[0] + ".prompt")
    provenance = {"device": device, "dtype": str(dtype), "api": "model.system_one",
                  "revision": config["model"]["revision"], "batch_size": 1,
                  "model_files_sha256": {name: digest(directory / name) for name in MODEL_FILES},
                  "versions": {name: version(name) for name in ("torch", "transformers", "tokenizers", "scikit-learn")}}
    return model, tokenizer, utility, native_prompt, provenance


def score_frame(frame, split, model, tokenizer, utility, native_prompt, seed, model_id, revision, output):
    """Checkpoint after each decision; resume only inside an identical run manifest."""
    expected = frame.set_index("uuid")
    rows = []
    if output.exists():
        for line in output.read_text().splitlines():
            rows.append(json.loads(line))
    seen = set()
    for row in rows:
        uuid = row["uuid"]
        if uuid in seen or uuid not in expected.index:
            raise ValueError("Duplicate or unknown UUID in inference checkpoint")
        if row["ground_truth_label"] != int(expected.loc[uuid, "label"]) or row["target"] != expected.loc[uuid, "target"]:
            raise ValueError("Checkpoint labels/targets differ from source")
        if row["split"] != split or row["model_identifier"] != model_id or row["model_revision"] != revision:
            raise ValueError("Checkpoint provenance differs from requested run")
        if not math.isfinite(row["binder_probability"]) or not 0 <= row["binder_probability"] <= 1:
            raise ValueError("Invalid checkpoint probability")
        seen.add(uuid)
    started = time.monotonic()
    with output.open("a") as file:
        for record in frame.sort_values("uuid").to_dict(orient="records"):
            if record["uuid"] in seen:
                continue
            prompt, state, encoding = build_prompt(record, tokenizer, utility, seed)
            question = {"type": "choice", "instructions": QUESTION, "criteria": dict.fromkeys(encoding)}
            if model.engine.render(state, native_prompt.as_question(question)) != prompt:
                raise ValueError("Native inference prompt differs from SFT formatter")
            response = model.system_one(state, {"binding": question})
            p = binder_probability(response)
            row = {"uuid": record["uuid"], "ground_truth_label": int(record["label"]),
                   "binder_probability": p, "target": record["target"], "model_identifier": model_id,
                   "model_revision": revision, "split": split,
                   "native_choice": response["answers"]["binding"]["choice"],
                   "input_tokens": response["usage"]["input_tokens"]}
            rows.append(row)
            file.write(json.dumps(row, allow_nan=False) + "\n")
            file.flush()
            if len(rows) % 10 == 0 or len(rows) == len(frame):
                print(f"{split}: {len(rows)}/{len(frame)} decisions; {time.monotonic() - started:.1f}s this run", flush=True)
    return pd.DataFrame(rows).sort_values("uuid").reset_index(drop=True)


def run(config_path, mode="primary", download=False, device="auto"):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    root = config_path.parent.parent
    results = root / config["paths"]["results"]
    processed = root / config["paths"]["processed"]
    split_dir = processed / "splits" / mode
    report_path = results / "splits" / mode / "report.json"
    split_report = json.loads(report_path.read_text())
    if split_report["mode"] != mode or split_report["dataset"] != config["dataset"] or split_report["options"] != config["splits"]:
        raise ValueError("Split provenance/config mismatch")
    if digest(processed / "binders.parquet") != split_report["input_sha256"]:
        raise ValueError("Processed cohort differs from split source")
    for name in (*[f"{s}.parquet" for s in NAMES], "assignments.parquet"):
        if digest(split_dir / name) != split_report["output_sha256"][name]:
            raise ValueError("Split hash mismatch")
    frames = {s: pd.read_parquet(split_dir / f"{s}.parquet") for s in NAMES}
    validate_assignment(pd.concat(frames.values()), pd.read_parquet(split_dir / "assignments.parquet"), mode)
    # Training contributes only the trivial majority class, never model updates.
    majority = int(frames["train"].label.value_counts().sort_index().idxmax())
    model, tokenizer, utility, native_prompt, provenance = load_base(config, root, download, device)
    directory = results / "zero_shot" / mode
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"model_identifier": config["model"]["id"], "mode": mode,
                "provenance": provenance, "split_report_sha256": digest(report_path),
                "sft_seed": config["sft"]["seed"], "majority_class_from_training": majority,
                "code_sha256": {name: digest(Path(__file__).with_name(name)) for name in ("evaluate.py", "prompts.py", "metrics.py")}}
    manifest_path = directory / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Run configuration changed; move previous zero_shot output before starting a new run")
    write_json(manifest_path, manifest)

    def score(split):
        return score_frame(frames[split], split, model, tokenizer, utility, native_prompt,
                           config["sft"]["seed"], config["model"]["id"], config["model"]["revision"],
                           directory / f"{split}_decisions.jsonl")

    validation = score("validation")
    selection = choose_threshold(validation.ground_truth_label, validation.binder_probability)
    selection["validation_decisions_sha256"] = digest(directory / "validation_decisions.jsonl")
    selection["test_used_for_selection"] = False
    # This artifact is committed to disk BEFORE test inference or test metrics.
    write_json(directory / "threshold.json", selection)
    print(f"Locked validation threshold: {selection['threshold']:.9g}", flush=True)
    test = score("test")
    metrics = {}
    for split, frame in (("validation", validation), ("test", test)):
        frame["predicted_class"] = (frame.binder_probability >= selection["threshold"]).astype(int)
        frame["threshold"] = selection["threshold"]
        frame.to_parquet(directory / f"{split}_predictions.parquet", index=False)
        frame.to_csv(directory / f"{split}_predictions.csv", index=False)
        metrics[split] = classification_metrics(frame.ground_truth_label, frame.binder_probability, selection["threshold"], majority)
        metrics[split]["fixed_0_5_comparison"] = classification_metrics(frame.ground_truth_label, frame.binder_probability, .5, majority)
    report = {"model_identifier": config["model"]["id"], "model_revision": config["model"]["revision"],
              "mode": mode, "threshold_selection": selection, "metrics": metrics,
              "pr_auc_definition": "Average precision is primary; trapezoidal PR integral is separately reported",
              "probability_readout": "Native choice readout: max over valid token forms per option, softmax across options; no fitted calibration",
              "test_target_metrics": {target: classification_metrics(g.ground_truth_label, g.binder_probability, selection["threshold"], majority)
                                      for target, g in test.groupby("target")},
              "run_manifest_sha256": digest(manifest_path)}
    write_json(directory / "metrics.json", report)
    print(json.dumps(metrics["test"], indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--mode", choices=("primary", "target_holdout"), default="primary")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run(args.config, args.mode, args.download_model, args.device)
