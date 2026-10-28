import json
from pathlib import Path
from types import SimpleNamespace

import pytest
pytest.importorskip("trl", reason="Install training extra for trainer tests")
import torch
from datasets import Dataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from decision_binder.data import digest, write_json
from decision_binder.train import choose_precision, load_training_data, run, smoke_subset, verify_lora, verify_masking, verify_source_lengths


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
