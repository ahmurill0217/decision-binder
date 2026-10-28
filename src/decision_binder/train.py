"""TRL + LoRA training; CPU preflight, explicit CUDA smoke/full execution."""
import argparse
import gc
from importlib.metadata import version
import json
import math
from pathlib import Path

from datasets import load_dataset
import yaml

from .data import digest, write_json
from .prompts import ASSET_FILES, load_prompt_assets

MODEL_FILES = (*ASSET_FILES, "config.json", "modeling_d1.py", "hybrid.py", "lfm2_vl.py", "runner.py", "api.py", "model.safetensors")
SPLITS = ("train", "validation")  # Never open, tokenize, evaluate, or fit on test.


def diagnostic_json(value):
    """Keep nonfinite diagnostic metrics explicit without emitting invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {k: diagnostic_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_json(v) for v in value]
    return value


def record_progress(output, state, logs):
    # Save evidence before raising, including grad_norm when AMP skips an update.
    write_json(output / "progress.json", diagnostic_json(
        {"global_step": state.global_step, "logs": state.log_history}))
    invalid_losses = {k: diagnostic_json(v) for k, v in (logs or {}).items()
                      if "loss" in k and not math.isfinite(float(v))}
    if invalid_losses:
        raise FloatingPointError(f"Nonfinite training/validation loss: {invalid_losses}")


def load_training_data(root, config):
    mode = config["training"]["experiment"]
    if mode not in ("primary", "target_holdout"):
        raise ValueError("Unknown experiment")
    folder = root / "data/sft" / mode
    report_path = root / config["paths"]["results"] / "sft" / mode / "report.json"
    report = json.loads(report_path.read_text())
    if report["mode"] != mode or report["model"] != config["model"] or report["sft"] != config["sft"]:
        raise ValueError("SFT provenance/config mismatch")
    split_report_path = root / config["paths"]["results"] / "splits" / mode / "report.json"
    split_report = json.loads(split_report_path.read_text())
    if digest(split_report_path) != report["split_report_sha256"] or split_report["options"] != config["splits"]:
        raise ValueError("SFT data no longer matches the configured split manifest")
    result = {}
    for split in SPLITS:
        path = folder / f"{split}.parquet"
        if digest(path) != report["output_sha256"][path.name]:
            raise ValueError(f"SFT checksum mismatch: {split}")
        result[split] = load_dataset("parquet", data_files=str(path), split="train",
                                     cache_dir=str(root / config["paths"]["raw"] / "training_cache"))
        if set(result[split].column_names) != {"uuid", "prompt", "completion"}:
            raise ValueError("Expected UUID/prompt/completion schema")
        ids = result[split]["uuid"]
        if not ids or len(ids) != len(set(ids)) or any(not x for x in ids):
            raise ValueError("Empty or duplicate training/validation UUIDs")
    if set(result["train"]["uuid"]) & set(result["validation"]["uuid"]):
        raise ValueError("Train/validation overlap")
    return result, {"sft_report_sha256": digest(report_path),
                    "input_sha256": {s: digest(folder / f"{s}.parquet") for s in SPLITS}}


def verify_source_lengths(datasets, tokenizer, max_length):
    stats = {}
    for split, dataset in datasets.items():
        lengths = []
        for row in dataset:
            p = tokenizer.encode(row["prompt"], add_special_tokens=False)
            c = tokenizer.encode(row["completion"], add_special_tokens=False)
            full = tokenizer.encode(row["prompt"] + row["completion"], add_special_tokens=False)
            if full != p + c or len(c) != 2 or c[-1] != tokenizer.eos_token_id:
                raise ValueError("Prompt boundary or answer/EOS mismatch")
            if len(full) > max_length:
                raise ValueError(f"{split} example {row['uuid']} exceeds max_length; no silent truncation")
            lengths.append(len(full))
        stats[split] = {"records": len(dataset), "min_tokens": min(lengths), "max_tokens": max(lengths)}
    return stats


def verify_masking(trainer, original, tokenizer):
    """Audit TRL's actual prepared labels and collated padding for every example."""
    summary = {}
    for split, prepared in (("train", trainer.train_dataset), ("validation", trainer.eval_dataset)):
        source = original[split]
        if len(prepared) != len(source):
            raise ValueError("TRL dropped or added examples")
        for index, row in enumerate(source):
            prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
            completion = tokenizer.encode(row["completion"], add_special_tokens=False)
            item = prepared[index]
            if item["input_ids"] != prompt + completion:
                raise ValueError("TRL changed token sequence or truncated the example")
            if item["labels"] != [-100] * len(prompt) + completion:
                raise ValueError("Completion-only loss mask is incorrect")
        # Include shortest/longest rows and representatives of both answer codes.
        lengths = [len(x["input_ids"]) for x in prepared]
        indices = {lengths.index(min(lengths)), lengths.index(max(lengths))}
        for code in sorted(set(row["completion"] for row in source)):
            indices.add(next(i for i, row in enumerate(source) if row["completion"] == code))
        examples = [prepared[i] for i in sorted(indices)]
        batch = trainer.data_collator(examples)
        for index, item in enumerate(examples):
            n = len(item["input_ids"])
            labels = batch["labels"][index]
            if labels[:n].tolist() != item["labels"] or not (labels[n:] == -100).all():
                raise ValueError("Collator changed labels or supervised padding")
            if not batch["attention_mask"][index, :n].eq(1).all() or not batch["attention_mask"][index, n:].eq(0).all():
                raise ValueError("Invalid padding attention mask")
            if labels.ne(-100).sum().item() != 2:
                raise ValueError("Exactly answer and EOS must be supervised")
        summary[split] = {"rows_verified": len(source), "supervised_tokens_per_example": 2,
                          "collated_rows": len(examples), "padding_mask_verified": True}
    return summary


def verify_lora(model, targets):
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not names or any("lora_" not in name or "vision_tower" in name for name in names):
        raise ValueError("Only language-model LoRA parameters may be trainable")
    for target in targets:
        if not any(f".{target}.lora_" in name for name in names):
            raise ValueError(f"Missing expected adapted module: {target}")
    return {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()), "trainable_parameter_names": names}


def choose_precision(requested, bf16_supported):
    if requested not in ("auto", "bf16", "fp16"):
        raise ValueError("precision must be auto, bf16, or fp16")
    if requested == "bf16" and not bf16_supported:
        raise ValueError("Requested bf16 is not natively supported by this GPU")
    return ("bf16" if bf16_supported else "fp16") if requested == "auto" else requested


def configure_loss_scaler(accelerator, precision, initial_scale):
    """Configure the existing AMP scaler before Accelerate prepares the optimizer."""
    if precision != "fp16":
        return None
    if not math.isfinite(initial_scale) or initial_scale <= 0:
        raise ValueError("fp16_initial_scale must be finite and positive")
    scaler = accelerator.scaler
    if scaler is None or not scaler.is_enabled():
        raise RuntimeError("Expected an enabled fp16 AMP scaler")
    state = scaler.state_dict()
    state["scale"] = float(initial_scale)
    scaler.load_state_dict(state)
    return scaler.get_scale()


def smoke_subset(dataset, count, tokenizer):
    if not isinstance(count, int) or count < 1:
        raise ValueError("Smoke example count must be a positive integer")
    lengths = [len(tokenizer.encode(row["prompt"] + row["completion"], add_special_tokens=False)) for row in dataset]
    longest = max(range(len(lengths)), key=lengths.__getitem__)
    indices = [longest] + [i for i in range(len(dataset)) if i != longest]
    return dataset.select(sorted(indices[:min(count, len(dataset))]))


def gpu_environment(config):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for smoke/full runs. Use Colab; --mode preflight is CPU-only. MPS training is not enabled.")
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    if total / gib < config["minimum_gpu_memory_gib"]:
        raise RuntimeError("GPU memory below configured minimum; no automatic quantization or setting changes")
    if free / gib < 10:
        raise RuntimeError("Less than 10 GiB GPU memory free; restart the runtime or free other allocations")
    precision = choose_precision(config["precision"], torch.cuda.is_bf16_supported(including_emulation=False))
    return {"name": props.name, "total_memory_gib": total / gib, "free_memory_gib_before_load": free / gib,
            "precision": precision, "device": "cuda", "compute_capability": list(torch.cuda.get_device_capability())}


def trainer_args(config, output, mode, hardware):
    from trl import SFTConfig
    t = config["training"]
    fields = ("seed", "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps",
              "learning_rate", "num_train_epochs", "lr_scheduler_type", "warmup_steps", "max_length",
              "gradient_checkpointing", "gradient_checkpointing_kwargs", "optim", "max_grad_norm", "logging_steps",
              "eval_strategy", "eval_steps", "save_strategy", "save_steps", "save_total_limit", "load_best_model_at_end")
    args = {key: t[key] for key in fields}
    if mode == "smoke":
        args.update(max_steps=t["smoke"]["max_steps"], gradient_accumulation_steps=t["smoke"]["gradient_accumulation_steps"],
                    warmup_steps=0, logging_steps=1, eval_steps=1, save_strategy="no", load_best_model_at_end=False)
    if mode == "preflight":
        args.update(use_cpu=True, save_strategy="no", eval_strategy="no", load_best_model_at_end=False)
    return SFTConfig(**args, output_dir=str(output), data_seed=t["seed"],
                     bf16=hardware["precision"] == "bf16", fp16=hardware["precision"] == "fp16",
                     completion_only_loss=True, assistant_only_loss=False, packing=False, padding_free=False,
                     prediction_loss_only=True, report_to="none", logging_nan_inf_filter=False,
                     dataloader_pin_memory=mode != "preflight", remove_unused_columns=True, use_cache=False)


def run(config_path, mode="preflight", output_override=None, download=False, approve_full=False):
    if mode not in ("preflight", "smoke", "full"):
        raise ValueError("Unknown execution mode")
    if mode == "full" and not approve_full:
        raise ValueError("Full training requires the explicit --approve-full-run flag")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    from transformers import AutoModel, TrainerCallback, set_seed
    from trl import SFTTrainer

    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text())
    t = config["training"]
    if mode == "smoke" and not 1 <= t["smoke"]["max_steps"] <= 5:
        raise ValueError("Smoke tests are limited to 1–5 optimizer steps")
    hardware = {"device": "cpu", "precision": "none", "purpose": "no-forward/no-optimizer preflight"} if mode == "preflight" else gpu_environment(t)
    print(json.dumps(hardware, indent=2), flush=True)
    default_output = t["output_dir"] + ("_" + mode if mode != "full" else "")
    output = Path(output_override).resolve() if output_override else root / default_output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory; refusing to overwrite {output}")
    original, provenance = load_training_data(root, config)
    local_model = root / config["paths"]["raw"] / "d1_base"
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(config["model"]["id"], revision=config["model"]["revision"],
                          allow_patterns=list(MODEL_FILES), local_dir=str(local_model))
    if any(not (local_model / f).exists() for f in MODEL_FILES):
        raise FileNotFoundError("Base snapshot missing; run with --download-model")
    tokenizer, _ = load_prompt_assets(config["model"], local_model)
    tokenizer.padding_side = "right"  # Native hybrid backbone supports right padding.
    lengths = verify_source_lengths(original, tokenizer, t["max_length"])
    if mode == "smoke":
        original = {s: smoke_subset(ds, t["smoke"][f'{s}_examples'], tokenizer) for s, ds in original.items()}
    set_seed(t["seed"])
    dtype = torch.float16 if hardware["precision"] == "fp16" else torch.bfloat16
    model = AutoModel.from_pretrained(str(local_model), local_files_only=True, trust_remote_code=True, dtype=dtype)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(**t["lora"]))
    model.peft_config["default"].base_model_name_or_path = config["model"]["id"]
    model.peft_config["default"].revision = config["model"]["revision"]
    lora = verify_lora(model, t["lora"]["target_modules"])
    # Make the checkpointing behavior inspectable before the first forward.
    if t["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=t["gradient_checkpointing_kwargs"])
        if not any(getattr(m, "gradient_checkpointing", False) for m in model.modules()):
            raise ValueError("Gradient checkpointing did not enable on model layers")
    checkpoint_layers = sum(getattr(m, "gradient_checkpointing", False) for m in model.modules())
    output.mkdir(parents=True, exist_ok=True)
    args = trainer_args(config, output, mode, hardware)
    trainer = SFTTrainer(model=model, args=args, train_dataset=original["train"].remove_columns("uuid"),
                         eval_dataset=original["validation"].remove_columns("uuid"), processing_class=tokenizer)
    initial_scale = configure_loss_scaler(trainer.accelerator, hardware["precision"],
                                          t.get("fp16_initial_scale", 16.0))
    masking = verify_masking(trainer, original, tokenizer)
    report = {"mode": mode, "status": "preflight_passed", "hardware": hardware, "config": config,
              "model_identifier": config["model"]["id"], "model_revision": config["model"]["revision"],
              "base_files_sha256": {name: digest(local_model / name) for name in MODEL_FILES},
              "source": provenance, "source_lengths": lengths, "masking": masking, "lora": lora,
              "gradient_checkpointing_layers": checkpoint_layers,
              "parameter_storage_gib": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3,
              "run_uuids": {s: list(ds["uuid"]) for s, ds in original.items()},
              "trainer_args": args.to_dict(), "test_split_loaded": False,
              "versions": {p: version(p) for p in ("torch", "transformers", "trl", "peft", "accelerate", "datasets")},
              "code_sha256": digest(Path(__file__)), "initial_loss_scale": initial_scale,
              "optimizer_steps": []}
    write_json(output / "metadata.json", report)
    print(f"Preflight passed: {lora['trainable_parameters']:,} trainable LoRA parameters; masks verified.", flush=True)
    if mode == "preflight":
        return report

    class FiniteLoss(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            scaler = trainer.accelerator.scaler
            report["optimizer_steps"].append({
                "global_step": state.global_step,
                "skipped": bool(trainer.accelerator.optimizer_step_was_skipped),
                "loss_scale_after_step": scaler.get_scale() if scaler is not None else None,
            })

        def on_log(self, args, state, control, logs=None, **kwargs):
            record_progress(output, state, logs)

    trainer.add_callback(FiniteLoss())
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    torch.cuda.reset_peak_memory_stats()
    try:
        train_result = trainer.train()
        evaluation = trainer.evaluate()
        if not math.isfinite(train_result.training_loss) or not math.isfinite(evaluation["eval_loss"]):
            raise FloatingPointError("Nonfinite final loss")
        changed = [n for n, p in model.named_parameters() if n in before and not torch.equal(p.detach().cpu(), before[n])]
        if not all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
            raise FloatingPointError("LoRA parameters contain nonfinite values")
        if not changed:
            skipped = sum(s["skipped"] for s in report["optimizer_steps"])
            raise ValueError(f"LoRA parameters did not change; AMP skipped {skipped}/{len(report['optimizer_steps'])} optimizer steps")
        adapter = output / "adapter"
        trainer.save_model(str(adapter))
        trainer.save_state()
        expected_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
        report.update(status="trained_adapter_saved", global_steps=trainer.state.global_step,
                      train_metrics=train_result.metrics, validation_metrics=evaluation,
                      peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                      changed_lora_tensors=len(changed), log_history=trainer.state.log_history)
        write_json(output / "metadata.json", diagnostic_json(report))
        # Release training state before loading a fresh base, avoiding two GPU copies.
        del trainer, model, before
        gc.collect()
        torch.cuda.empty_cache()
        fresh = AutoModel.from_pretrained(str(local_model), local_files_only=True, trust_remote_code=True, dtype=dtype)
        reloaded = PeftModel.from_pretrained(fresh, str(adapter), is_trainable=False)
        actual_state = get_peft_model_state_dict(reloaded)
        if expected_state.keys() != actual_state.keys() or any(not torch.equal(v, actual_state[k].detach().cpu()) for k, v in expected_state.items()):
            raise ValueError("Adapter reload changed saved tensors")
        report.update(status="passed", adapter_reload_verified=True,
                      adapter_sha256={p.name: digest(p) for p in adapter.iterdir() if p.is_file()})
        write_json(output / "metadata.json", diagnostic_json(report))
        return report
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      log_history=trainer.state.log_history if "trainer" in locals() else report.get("log_history", []))
        write_json(output / "metadata.json", diagnostic_json(report))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"), default="preflight")
    parser.add_argument("--output-dir")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--approve-full-run", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mode, args.output_dir, args.download_model, args.approve_full_run)
