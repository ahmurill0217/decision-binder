"""CPU-only d1 prompt/completion conversion; no model or trainer imports."""
import argparse
import importlib.util
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import re
import sys

from datasets import Dataset, DatasetDict, Features, Value
import pandas as pd
import yaml

from .data import digest, write_json
from .features import FEATURES, model_inputs
from .splits import NAMES, validate_assignment

QUESTION = "Is this candidate expected to bind its intended protein target?"
CHOICES = ("Binder", "Non-binder")
LABEL_CHOICES = {1: "Binder", 0: "Non-binder"}
ASSET_FILES = ("prompt.py", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE")
SCHEMA = Features({"uuid": Value("string"), "prompt": Value("string"), "completion": Value("string")})


def load_prompt_assets(model_config, directory, download=False):
    """Download only enumerated lightweight assets, hash-check, then import.

    Default offline: network is allowed only with the explicit download flag.
    No AutoModel, model config, weight files, or unreviewed remote imports.
    """
    directory = Path(directory)
    if not re.fullmatch(r"[a-f0-9]{40}", model_config["revision"]):
        raise ValueError("Pin model revision to an immutable commit SHA")
    if set(model_config["assets_sha256"]) != set(ASSET_FILES):
        raise ValueError("Asset manifest must match the lightweight file allowlist")
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(model_config["id"], revision=model_config["revision"],
                          allow_patterns=list(ASSET_FILES), local_dir=str(directory))
    for name in ASSET_FILES:
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run with --download-tokenizer once")
        if digest(path) != model_config["assets_sha256"][name]:
            raise ValueError(f"Pinned asset checksum mismatch: {name}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    module_name = "_decision_binder_d1_" + model_config["assets_sha256"]["prompt.py"]
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, directory / "prompt.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # dataclass definitions require registration
        spec.loader.exec_module(module)
    utility = sys.modules[module_name]
    if tokenizer.eos_token != utility.IM_END or not tokenizer.bos_token:
        raise ValueError("Unexpected d1 BOS/EOS convention")
    for token in (tokenizer.bos_token, utility.IM_START, utility.IM_END):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != token:
            raise ValueError(f"Invalid special token: {token}")
    return tokenizer, utility


def feature_state(record):
    """Keep original names, order, and numeric precision; exclude all other keys."""
    state = {}
    for name in FEATURES:
        value = record[name]
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError(f"Unsupported feature value: {name}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Nonfinite feature: {name}")
        if isinstance(value, str) and ("<|" in value or "|>" in value):
            raise ValueError(f"Special-token markup in feature: {name}")
        state[name] = value
    # json roundtrip retains IEEE float values; no rounding, scaling or imputation.
    if json.loads(json.dumps(state, allow_nan=False)) != state:
        raise ValueError("State JSON roundtrip failed")
    return state


def build_prompt(record, tokenizer, utility, seed=1337):
    """Shared training/inference formatter: never reads an experimental label.

    UUID seeds option order but is not included in the state or rendered text.
    Both options always exist; a label-independent shuffle avoids answer leakage.
    """
    if not isinstance(record["uuid"], str) or not record["uuid"]:
        raise ValueError("A nonempty UUID is required for reproducible choice order")
    options = list(CHOICES)
    random.Random(f"decision-binder-v1:{seed}:{record['uuid']}").shuffle(options)
    question = utility.as_question({"type": "choice", "instructions": QUESTION,
                                    "criteria": {choice: None for choice in options}})
    aliases = utility.aliases(tokenizer, options)
    mapping = {choice: {"code": code, "token_id": token_id}
               for choice, (code, token_id) in zip(options, aliases)}
    for choice, entry in mapping.items():
        ids = tokenizer.encode(entry["code"], add_special_tokens=False)
        if ids != [entry["token_id"]] or tokenizer.decode(ids) != entry["code"]:
            raise ValueError(f"Answer alias is not a roundtripping single token: {choice}")
    if len({entry["token_id"] for entry in mapping.values()}) != 2:
        raise ValueError("Answer choices must have distinct token IDs")
    state = feature_state(record)
    prompt = utility.render(tokenizer, state, question, bos=tokenizer.bos_token,
                            style="json_only", system="none", lead="", option_style="desc")
    return prompt, state, mapping


def convert_record(record, tokenizer, utility, seed=1337):
    prompt, state, mapping = build_prompt(record, tokenizer, utility, seed)
    label = record["label"]
    if isinstance(label, bool) or label not in LABEL_CHOICES:
        raise ValueError("Expected experimental label 0 or 1")
    choice = LABEL_CHOICES[label]
    answer = mapping[choice]
    completion = answer["code"] + tokenizer.eos_token
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if completion_ids != [answer["token_id"], tokenizer.eos_token_id]:
        raise ValueError("Completion must contain exactly answer token plus end-of-turn")
    if tokenizer.decode(completion_ids, skip_special_tokens=False) != completion:
        raise ValueError("Completion decode mismatch")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    joined_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
    if joined_ids != prompt_ids + completion_ids:
        raise ValueError("Tokenization changed at the prompt/completion boundary")
    if prompt_ids[0] != tokenizer.bos_token_id or prompt.count(tokenizer.bos_token) != 1:
        raise ValueError("Prompt must have exactly one leading BOS")
    if not prompt.endswith(utility.IM_START + "assistant\n"):
        raise ValueError("Prompt must end at the assistant answer slot")
    # Check default tokenization doesn't silently duplicate already-rendered BOS.
    if tokenizer.encode(prompt + completion) != joined_ids:
        raise ValueError("Default tokenizer adds special tokens; review later TRL handling")
    result = {"uuid": record["uuid"], "prompt": prompt, "completion": completion}
    audit = {"uuid": record["uuid"], "features": state, "label": int(label),
             "intended_answer": choice, "choice_encoding": mapping,
             "prompt": prompt, "completion": completion, "completion_token_ids": completion_ids,
             "prompt_tokens": len(prompt_ids), "total_tokens": len(joined_ids)}
    return result, audit


def convert_frame(frame, tokenizer, utility, seed):
    if frame.uuid.isna().any() or frame.uuid.duplicated().any():
        raise ValueError("Missing or duplicate UUID")
    model_inputs(frame)  # fail early on missing allowlisted feature columns
    rows, audits = [], []
    for record in frame.sort_values("uuid").to_dict(orient="records"):
        row, audit = convert_record(record, tokenizer, utility, seed)
        rows.append(row)
        audits.append(audit)
    return Dataset.from_list(rows, features=SCHEMA), audits


def run(config_path, mode="primary", download=False):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    root = config_path.parent.parent
    processed = root / config["paths"]["processed"]
    results = root / config["paths"]["results"]
    source_dir = processed / "splits" / mode
    split_report_path = results / "splits" / mode / "report.json"
    split_report = json.loads(split_report_path.read_text())
    if split_report["mode"] != mode or split_report["options"] != config["splits"] or split_report["dataset"] != config["dataset"]:
        raise ValueError("Split configuration differs from the saved report; regenerate splits")
    if digest(processed / "binders.parquet") != split_report["input_sha256"]:
        raise ValueError("Processed cohort differs from split source")
    for name in (*[f"{s}.parquet" for s in NAMES], "assignments.parquet"):
        if digest(source_dir / name) != split_report["output_sha256"][name]:
            raise ValueError(f"Split checksum mismatch: {name}")
    assignments = pd.read_parquet(source_dir / "assignments.parquet")
    cohort = pd.read_parquet(processed / "binders.parquet")
    validate_assignment(cohort, assignments, mode)
    frames = {name: pd.read_parquet(source_dir / f"{name}.parquet") for name in NAMES}
    pd.testing.assert_frame_equal(pd.concat(frames.values()).sort_values("uuid").reset_index(drop=True),
                                  cohort.sort_values("uuid").reset_index(drop=True))
    for name, frame in frames.items():
        if set(frame.uuid) != set(assignments.loc[assignments.split.eq(name), "uuid"]):
            raise ValueError("Split assignments differ from saved Parquet")
    tokenizer, utility = load_prompt_assets(config["model"], root / config["paths"]["raw"] / "d1_tokenizer", download)
    converted = DatasetDict()
    summary, examples = {}, []
    for name, frame in frames.items():
        converted[name], audits = convert_frame(frame, tokenizer, utility, config["sft"]["seed"])
        lengths = [a["total_tokens"] for a in audits]
        summary[name] = {"examples": len(audits), "min_tokens": min(lengths), "max_tokens": max(lengths),
                         "binder_examples": int(frame.label.sum()),
                         "first_choice_binder": sum(next(iter(a["choice_encoding"])) == "Binder" for a in audits)}
        # One inspectable example per split, guaranteeing both label types if present.
        desired = 0 if name == "validation" else 1
        sample = next((a for a in audits if a["label"] == desired), audits[0])
        examples.append({"split": name, **sample})
    destination = root / "data" / "sft" / mode
    report_dir = results / "sft" / mode
    report_dir.mkdir(parents=True, exist_ok=True)
    converted.save_to_disk(str(destination / "dataset"))
    for name in NAMES:
        converted[name].to_parquet(str(destination / f"{name}.parquet"))
    write_json(report_dir / "examples.json", examples)
    example_text = "# Three inspectable d1 examples\n\nGenerated from saved split records; features and prompts are unabridged.\n"
    for example in examples:
        example_text += f"\n## {example['split']}: {example['uuid']}\n\n"
        example_text += "Original allowed features:\n\n```json\n" + json.dumps(example["features"], indent=2, ensure_ascii=False) + "\n```\n"
        example_text += "\nRendered prompt:\n\n```text\n" + example["prompt"] + "\n```\n"
        example_text += "\nCompletion (literal end-of-turn token):\n\n```text\n" + example["completion"] + "\n```\n"
        example_text += f"\nExperimental label: {example['label']} ({example['intended_answer']}). Completion token IDs: {example['completion_token_ids']}. Full length: {example['total_tokens']} tokens.\n"
    (report_dir / "examples.md").write_text(example_text)
    report = {"mode": mode, "model": config["model"], "sft": config["sft"], "question": QUESTION,
              "feature_columns": list(FEATURES), "summary": summary,
              "split_report_sha256": digest(split_report_path), "split_sha256": split_report["output_sha256"],
              "code_sha256": {p: digest(Path(__file__).with_name(p)) for p in ("prompts.py", "features.py")},
              "versions": {p: version(p) for p in ("transformers", "tokenizers", "datasets", "pyarrow")},
              "output_sha256": {str(p.relative_to(destination)): digest(p) for p in sorted(destination.rglob("*")) if p.is_file()},
              "checks": {"all_answer_codes_single_token": True, "all_completions_answer_plus_eos": True,
                         "all_token_boundaries_stable": True, "no_truncation": True,
                         "split_partition_preserved": True},
              "training_status": "Not started; TRL completion-only collator verification deferred"}
    write_json(report_dir / "report.json", report)
    print(json.dumps(summary, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--mode", choices=("primary", "target_holdout"), default="primary")
    parser.add_argument("--download-tokenizer", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mode, args.download_tokenizer)
