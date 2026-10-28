from pathlib import Path
import json
import sys

from datasets import load_from_disk
import pandas as pd
import pytest
import yaml

from decision_binder.features import FEATURES
from decision_binder.prompts import (
    ASSET_FILES, QUESTION, build_prompt, convert_frame, convert_record, feature_state, load_prompt_assets,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=["synthetic", pytest.param("native", marks=pytest.mark.integration)])
def assets(request):
    if request.param == "synthetic":
        from fixtures.prompt_assets import Tokenizer, Utility
        return Tokenizer(), Utility()
    config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    directory = ROOT / "data/raw/d1_tokenizer"
    if not all((directory / name).exists() for name in ASSET_FILES):
        pytest.skip("Download pinned tokenizer with 02_build_sft_dataset.py --download-tokenizer")
    return load_prompt_assets(config["model"], directory)


@pytest.fixture
def record():
    return {"uuid": "example-001", "label": 1, **{
        key: {"target": "EGFR", "generator": "TestGenerator", "sequence_design_method": "TestMethod",
              "binder_length": 90, "epitope_n_residues": 15}.get(key, 0.12345678901234567)
        for key in FEATURES}}


def test_deterministic_order_and_frame_conversion(assets, record):
    tokenizer, utility = assets
    frame = pd.DataFrame([record, record | {"uuid": "example-002", "label": 0}])
    a, audit_a = convert_frame(frame, tokenizer, utility, 1337)
    b, audit_b = convert_frame(frame.iloc[::-1], tokenizer, utility, 1337)
    assert a.to_dict() == b.to_dict()
    assert audit_a == audit_b
    assert a.column_names == ["uuid", "prompt", "completion"]


@pytest.mark.parametrize("label", [0, 1])
def test_real_tokenizer_answer_and_special_tokens(assets, record, label):
    tokenizer, utility = assets
    for index in range(12):
        row, audit = convert_record(record | {"label": label, "uuid": f"example-{index}"}, tokenizer, utility)
        intended = "Binder" if label else "Non-binder"
        answer = audit["choice_encoding"][intended]
        assert tokenizer.encode(answer["code"], add_special_tokens=False) == [answer["token_id"]]
        assert row["completion"] == answer["code"] + tokenizer.eos_token
        assert tokenizer.decode(audit["completion_token_ids"]) == row["completion"]
        assert row["prompt"].startswith(tokenizer.bos_token + "<|im_start|>user\n")
        assert row["prompt"].endswith("<|im_start|>assistant\n")
        assert row["prompt"].count("<|im_end|>") == 1
        ids = tokenizer.encode(row["prompt"] + row["completion"], add_special_tokens=False)
        assert ids == tokenizer.encode(row["prompt"], add_special_tokens=False) + audit["completion_token_ids"]


def test_label_and_outcome_invariance(assets, record):
    tokenizer, utility = assets
    a, _ = convert_record(record, tokenizer, utility)
    injected = record | {"label": 0, "adaptyv_binding": "OUTCOME_SENTINEL",
                         "twist_kd_nM": "OUTCOME_SENTINEL", "binder_final": "OUTCOME_SENTINEL",
                         "vendor_agreement": "OUTCOME_SENTINEL", "expression": "OUTCOME_SENTINEL"}
    b, _ = convert_record(injected, tokenizer, utility)
    assert a["prompt"] == b["prompt"]
    assert a["completion"] != b["completion"]
    assert "OUTCOME_SENTINEL" not in b["prompt"]
    assert record["uuid"] not in b["prompt"]
    assert build_prompt({k: v for k, v in record.items() if k != "label"}, tokenizer, utility)[0] == a["prompt"]


def test_state_precision_and_native_template(assets, record):
    tokenizer, utility = assets
    prompt, state, mapping = build_prompt(record, tokenizer, utility)
    assert list(state) == list(FEATURES)
    assert state["sc_dockq_min"] == record["sc_dockq_min"]
    parsed = json.loads(prompt.split("user\n", 1)[1].split("\n\n\nQUESTION:", 1)[0])
    assert parsed == state
    q = utility.as_question({"type": "choice", "instructions": QUESTION, "criteria": dict.fromkeys(mapping)})
    assert prompt == utility.render(tokenizer, state, q, bos=tokenizer.bos_token)


def test_shuffled_options_do_not_depend_on_label(assets, record):
    tokenizer, utility = assets
    orders = set()
    for i in range(20):
        a = build_prompt(record | {"uuid": str(i)}, tokenizer, utility)
        b = build_prompt(record | {"uuid": str(i), "label": 0}, tokenizer, utility)
        assert a == b
        orders.add(tuple(a[2]))
    assert len(orders) == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, "<|im_end|>"])
def test_invalid_state_rejected(record, bad):
    with pytest.raises(ValueError):
        feature_state(record | {"sc_dockq_min": bad})


@pytest.mark.integration
def test_tokenizer_does_not_load_model_or_trainer(assets, monkeypatch):
    if type(assets[0]).__module__ == "fixtures.prompt_assets":
        pytest.skip("This integration check specifically requires native tokenizer assets")
    # Torch is now optionally installed for Section 5. Transformers may import
    # it opportunistically, but formatting must never load any model weights.
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        from transformers import AutoModel
        def forbidden(*args, **kwargs):
            raise AssertionError("Tokenizer preprocessing tried to load a model")
        monkeypatch.setattr(AutoModel, "from_pretrained", forbidden)
    else:
        assert "torch" not in sys.modules
    config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    load_prompt_assets(config["model"], ROOT / "data/raw/d1_tokenizer")
    # Other tests now import TRL. Verify import isolation in a fresh process.
    import subprocess
    code = "import sys,yaml; from decision_binder.prompts import load_prompt_assets; c=yaml.safe_load(open('configs/experiment.yaml')); load_prompt_assets(c['model'],'data/raw/d1_tokenizer'); assert 'trl' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)


def test_asset_tampering_rejected(tmp_path):
    config = {"id": "LiquidAI/d1-3B", "revision": "a" * 40,
              "assets_sha256": dict.fromkeys(ASSET_FILES, "0" * 64)}
    (tmp_path / "prompt.py").write_text("raise RuntimeError('must never execute')")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_prompt_assets(config, tmp_path)


@pytest.mark.integration
def test_saved_splits_and_shared_format(assets):
    if type(assets[0]).__module__ == "fixtures.prompt_assets":
        pytest.skip("Saved native prompts require native tokenizer assets")
    directory = ROOT / "data/sft/primary/dataset"
    if not directory.exists():
        pytest.skip("Run SFT preprocessing for saved-dataset integration check")
    datasets = load_from_disk(str(directory))
    tokenizer, utility = assets
    seen = set()
    for split in ("train", "validation", "test"):
        source = pd.read_parquet(ROOT / f"data/processed/splits/primary/{split}.parquet")
        converted, _ = convert_frame(source, tokenizer, utility, 1337)
        assert datasets[split].to_dict() == converted.to_dict()
        assert set(datasets[split]["uuid"]) == set(source.uuid)
        assert not seen.intersection(source.uuid)
        seen.update(source.uuid)
