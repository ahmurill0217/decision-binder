import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
pytest.importorskip("trl", reason="Install training extra for trainer tests")
import torch
from datasets import Dataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from decision_binder.data import digest, write_json
from decision_binder.train import choose_precision, load_training_data, run, smoke_subset, verify_lora, verify_masking, verify_source_lengths
from decision_binder.train import diagnostic_json, record_progress
from decision_binder.train import configure_loss_scaler, trainer_args


def test_amp_scale_configuration_and_real_optimizer_update():
    # Exercise the real scaler on CPU; this does not certify CUDA numerics.
    scaler = torch.amp.GradScaler("cpu")
    accelerator = SimpleNamespace(scaler=scaler)
    assert configure_loss_scaler(accelerator, "fp16", 16.0) == 16.0
    p = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([p], lr=0.1)
    scaler.scale(p.square()).backward()
    scaler.step(optimizer)
    scaler.update()
    assert p.item() == pytest.approx(0.8)
    assert configure_loss_scaler(SimpleNamespace(scaler=None), "bf16", 16.0) is None
    with pytest.raises(ValueError, match="positive"):
        configure_loss_scaler(accelerator, "fp16", float("nan"))


def test_smoke_disables_warmup_but_full_preserves_it(tmp_path):
    import yaml
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/experiment.yaml").read_text())
    # CPU hardware avoids requiring CUDA for argument construction.
    smoke = trainer_args(config, tmp_path, "smoke", {"precision": "none"})
    full = trainer_args(config, tmp_path, "full", {"precision": "none"})
    assert smoke.warmup_steps == 0
    assert full.warmup_steps == config["training"]["warmup_steps"]


def test_nonfinite_diagnostics_remain_valid_json(tmp_path):
    logs = {"loss": 1.25, "grad_norm": float("nan")}
    state = SimpleNamespace(global_step=1, log_history=[logs])
    record_progress(tmp_path, state, logs)
    saved = json.loads((tmp_path / "progress.json").read_text())
    assert saved["logs"] == [{"loss": 1.25, "grad_norm": "NaN"}]
    assert math.isnan(logs["grad_norm"])
    assert diagnostic_json([float("inf"), float("-inf")]) == ["Infinity", "-Infinity"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_loss_still_fails_after_saving_evidence(tmp_path, value):
    logs = {"eval_loss": value}
    state = SimpleNamespace(global_step=1, log_history=[logs])
    with pytest.raises(FloatingPointError, match="eval_loss"):
        record_progress(tmp_path, state, logs)
    assert json.loads((tmp_path / "progress.json").read_text())["logs"][0]["eval_loss"] in {
        "NaN", "Infinity", "-Infinity"}


def test_precision_and_full_run_guard():
    assert choose_precision("auto", False) == "fp16"
    assert choose_precision("auto", True) == "bf16"
    with pytest.raises(ValueError, match="not natively"):
        choose_precision("bf16", False)
    with pytest.raises(ValueError, match="approve-full-run"):
        run("unused.yaml", mode="full")


def test_loader_works_without_test_data_and_rejects_tampering(tmp_path):
    folder = tmp_path / "data/sft/primary"
    folder.mkdir(parents=True)
    reports = tmp_path / "results/sft/primary"
    reports.mkdir(parents=True)
    splits = tmp_path / "results/splits/primary"
    splits.mkdir(parents=True)
    config = {"training": {"experiment": "primary"}, "paths": {"results": "results", "raw": "data/raw"},
              "model": {}, "sft": {}, "splits": {"seed": 1337}}
    write_json(splits / "report.json", {"options": config["splits"]})
    hashes = {}
    for split in ("train", "validation"):
        path = folder / f"{split}.parquet"
        Dataset.from_dict({"uuid": [split], "prompt": ["prompt"], "completion": ["A<|im_end|>"]}).to_parquet(str(path))
        hashes[path.name] = digest(path)
    write_json(reports / "report.json", {"mode": "primary", "model": {}, "sft": {}, "output_sha256": hashes,
                                         "split_report_sha256": digest(splits / "report.json")})
    result, _ = load_training_data(tmp_path, config)
    assert set(result) == {"train", "validation"}
    assert not (folder / "test.parquet").exists()
    with (folder / "train.parquet").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_training_data(tmp_path, config)


class TinyTokenizer:
    eos_token_id = 3
    def encode(self, text, **kwargs):
        return {"p": [7, 8], "long": [7, 8, 9], "A!": [1, 3], "B!": [2, 3],
                "pA!": [7, 8, 1, 3], "longB!": [7, 8, 9, 2, 3]}[text]


def mask_fixture():
    source = Dataset.from_dict({"uuid": ["a", "b"], "prompt": ["p", "long"], "completion": ["A!", "B!"]})
    prepared = [{"input_ids": [7, 8, 1, 3], "labels": [-100, -100, 1, 3]},
                {"input_ids": [7, 8, 9, 2, 3], "labels": [-100, -100, -100, 2, 3]}]
    trainer = SimpleNamespace(train_dataset=prepared, eval_dataset=prepared,
                              data_collator=DataCollatorForLanguageModeling(pad_token_id=0))
    return {"train": source, "validation": source}, trainer


def test_actual_trl_collator_masks_prompt_and_padding():
    source, trainer = mask_fixture()
    result = verify_masking(trainer, source, TinyTokenizer())
    assert result["train"]["supervised_tokens_per_example"] == 2
    assert result["validation"]["padding_mask_verified"]
    trainer.train_dataset[0]["labels"][0] = 7
    with pytest.raises(ValueError, match="mask is incorrect"):
        verify_masking(trainer, source, TinyTokenizer())


def test_truncation_fails_instead_of_dropping_answers():
    source, _ = mask_fixture()
    with pytest.raises(ValueError, match="exceeds max_length"):
        verify_source_lengths(source, TinyTokenizer(), 4)
    assert verify_source_lengths(source, TinyTokenizer(), 5)["train"]["max_tokens"] == 5


def test_smoke_includes_longest_example():
    source, _ = mask_fixture()
    assert smoke_subset(source["train"], 1, TinyTokenizer())["uuid"] == ["b"]


def test_trainable_non_lora_or_vision_parameters_rejected():
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="Only language-model LoRA"):
        verify_lora(model, ["q_proj"])


def test_colab_wrapper_has_safe_default_and_compiles():
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/finetune_colab.ipynb").read_text())
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for cell in code:
        compile(cell, "colab-cell", "exec")
    assert "RUN_FULL_TRAINING = False" in code[0]
    assert any('"--mode", "smoke"' in cell for cell in code)
    assert not any("SFTTrainer(" in cell for cell in code)
