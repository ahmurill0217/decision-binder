# decision-binder

Research question: can supervised fine-tuning of Liquid AI's d1-3B improve prediction of experimentally measured protein binding from computational structure metrics over the same model zero-shot?

**Sections 3–5 and the Section 6 training implementation are available:** data preparation, leakage audits, primary and target-holdout splits, deterministic SFT formatting, and the original d1-3B zero-shot baseline on the primary test set. No fine-tuning has been performed. The Python project is named `decision-binder`; the existing checkout directory remains `decision-bio`.

## Run

Python 3.12 and [uv](https://docs.astral.sh/uv/) are used for the tested environment.

```bash
uv sync --frozen --extra dev
uv run --frozen python scripts/01_prepare_data.py
uv run --frozen python scripts/02_create_splits.py
uv run --frozen python scripts/02_create_splits.py --mode target_holdout
uv run --frozen pytest -q
```

Run from the repository root. The first run needs internet access; later runs verify and reuse local snapshots. Only the `design_summary` and `insilico_cofold_predictions` Hugging Face table configurations are downloaded, not structures. The dataset commit is pinned in [configs/experiment.yaml](configs/experiment.yaml); `uv.lock` pins transitive dependencies. Alternatively install with `python -m pip install -e '.[dev]'` in a fresh virtual environment (direct dependencies pinned, without uv's transitive lock guarantee).

## Observed preprocessing results

Dataset revision: `9e1b81696da46835e9e9cde9a3da976e0abc92ab`.

| Stage | Records |
|---|---:|
| Source designs | 1,440 |
| Source prediction rows (all predictors) | 113,550 |
| Boltz-2 prediction rows | 11,400 |
| Primary target / campaign stoichiometry prediction rows | 7,200 |
| Aggregated designs (five seeds each) | 1,440 |
| Resolved Adaptyv binder/non-binder labels | 1,235 |
| Complete labeled feature records | 1,076 |
| Final binders / non-binders | 294 / 782 |

205 designs are excluded by label policy and another 159 by incomplete aggregated features. Raw labels: binder 336, non_binder 899, not_expressed 61, not_tested 100, inconclusive 44. No missing labels are converted to negatives.

The original notebook's pooling procedure reproduces **1,082 records (294 binders, 788 non-binders)** on this revision. Our main output deliberately separates primary predictions from alternate forms and extra stoichiometries; this changes feature values and removes six additional non-binders. The comparison is preprocessing only.

There are no duplicate design UUIDs, duplicate prediction keys, or identical binder sequences. The sequence audit finds 36 pairs with at least 90% position-wise identity at equal length; 24 pairs have both designs retained. They are reported, not removed. Step E keeps transitive related-sequence components together. This limited screen does not detect all alignment or lineage relationships.

## Five processed records (selected columns)

| UUID | Target | Length | Mean scDockQ | Mean minimum ipSAE | Label |
|---|---|---:|---:|---:|---:|
| 002a7326-6f58-51e3-95c8-67d38c8269fd | Cas9 | 70 | 0.004672 | 0.000000 | 1 |
| 002e2495-ef2a-56d4-8f2a-45ca6d32c5e0 | EGFR | 90 | 0.241634 | 0.718102 | 1 |
| 008df55d-b575-52b3-83ee-85f2ce3ba893 | BBF-14 | 89 | 0.757012 | 0.852300 | 0 |
| 015e2121-bbb8-5aa0-9b08-bb74b51f0e46 | TNFa | 102 | 0.398148 | 0.203571 | 0 |
| 019a827b-01f2-51ee-907d-9d0621662aae | Latent GDF-8 | 70 | 0.548022 | 0.121948 | 0 |

All 85 features for these records are in [results/sample_records.json](results/sample_records.json). The exact feature list is in [results/preprocessing_report.json](results/preprocessing_report.json) and described in [docs/data_dictionary.md](docs/data_dictionary.md).

## Layout and outputs

Core logic lives in `src/decision_binder/data.py`, `features.py`, `splits.py`, and `prompts.py`; numbered scripts are thin entry points. `tests/` covers preprocessing and leakage boundaries. Later milestone modules and numbered scripts are intentionally absent rather than empty implementations.

- `data/raw/`: checksum-verified table snapshots and Hugging Face cache; ignored by Git.
- `data/processed/binders.parquet` and `.csv`: UUID, 85 features, binary label; ignored by Git.
- `data/processed/feature_columns.json`: the explicit input columns. Use `model_inputs()` to select them; never infer inputs by dropping one label column.
- `results/preprocessing_report.json`: counts, config, revision, checksums, versions, and original-method comparator.
- `results/exclusions.csv`: one disposition per design, with original label (audit only).
- `results/related_sequences.csv`: potential duplicate pairs (audit only).
- `results/metric_counts.csv`, `prediction_groups.csv`: seed coverage and grouping audit.

Section 4 writes converted datasets under `data/sft/<mode>/`. The output directory is replaced on reruns; use separate `paths` in a copied config for sensitivity runs. Config-relative paths resolve from the config's grandparent (normally repository root).

See [methodology and limitations](docs/methodology.md) and [reference inspection](docs/references.md). The source dataset is [Anthropic/claude-protein-binder-design](https://huggingface.co/datasets/Anthropic/claude-protein-binder-design), released under CC BY 4.0; retain attribution when sharing derived data. 
## Step E: reproducible splits

Fixed seed **1337**. Primary grouped, label-stratified allocation produces:

| Split | Records | Binders | Non-binders | Binder share | Targets |
|---|---:|---:|---:|---:|---:|
| Train | 754 | 206 | 548 | 27.32% | 15 |
| Validation | 161 | 44 | 117 | 27.33% | 15 |
| Test | 161 | 44 | 117 | 27.33% | 15 |

Parquets are in `data/processed/splits/primary/`. The separate `target_holdout/` experiment uses 755/159/162 records: 218/47/29 binders and 537/112/133 non-binders. Test targets are **EGFR and TrkA**; validation targets are **15-PGDH, Cas9, and PD-L1**. Training uses the other ten retained targets. Holdout target selection uses sizes and the seed, not label balance. Do not combine metrics from these experiments.

[Split methodology and target counts](docs/splits.md) describes grouping, configuration, limitations, and all target counts. Machine-readable reports and UUID assignments are in `results/splits/<mode>/`. The assignment audit Parquet contains grouping columns; the train/validation/test Parquets preserve the preprocessing schema. No UUID or detected sequence component crosses splits; no target crosses splits in the secondary experiment.

## Section 4: CPU-only d1-3B SFT preprocessing

```bash
uv sync --frozen --extra dev
# First run downloads ONLY pinned tokenizer/prompt assets, never model weights:
uv run --frozen python scripts/02_build_sft_dataset.py --download-tokenizer
# Subsequent runs are offline:
uv run --frozen python scripts/02_build_sft_dataset.py
# Optional separate secondary experiment:
uv run --frozen python scripts/02_build_sft_dataset.py --mode target_holdout
uv run --frozen pytest -q
```

The shared formatter uses Liquid's `as_question()`, `aliases()`, and `render()`. It serializes the 85 allowlisted features without rounding, shuffles both choices deterministically by seed and UUID, and creates the correct single-token answer code followed by `<|im_end|>`. UUID and experimental labels never enter the input state. The pinned asset SHA-256 values are verified before importing the prompt utility. This formatting step requires no torch, GPU, model weights, trainer, or LLM service. The optional evaluation extra separately installs torch; Transformers can import it opportunistically when present. In a tokenizer-only environment, the warning about absent PyTorch is expected.

`data/sft/primary/dataset/` is a Hugging Face `DatasetDict` with **754 train / 161 validation / 161 test** examples, each containing `uuid`, `prompt`, and `completion`. Load it with `datasets.load_from_disk("data/sft/primary/dataset")`. Individual split Parquets are also saved. The secondary experiment remains separate under `data/sft/target_holdout/` with **755 / 159 / 162** examples. Only `train` may update model weights later; validation is for development and test stays reserved.

[Three full examples](results/sft/primary/examples.md) show original features, native prompts, and completions. [Formatting and future masking checks](docs/sft_format.md) describe the reference adaptation and unresolved training compatibility. Every primary example passed real-tokenizer single-token, EOS, and boundary checks; total lengths are 1,350–1,523 tokens with no truncation. No training has started.

## Section 5: zero-shot baseline

```bash
uv sync --frozen --extra dev --extra evaluation
uv run --frozen --extra evaluation python scripts/03_evaluate_base.py --download-model --device mps
```

Later runs can omit `--download-model`. The original pinned d1-3B checkpoint runs locally through `system_one()`; no unconstrained generation or training occurs. Validation selects a balanced-accuracy threshold **0.7431680086124811**, locked before test inference. Test results (161 designs): ROC-AUC **0.6222**, PR-AUC/average precision **0.3509**, precision **0.3571**, recall **0.4545**, F1 **0.4000**, balanced accuracy **0.5734**. Accuracy **62.73%** is below the always-non-binder baseline's **72.67%**. Confusion matrix: `[[81, 36], [24, 20]]` (true rows, predicted columns; non-binder first).

[Full baseline methodology and results](docs/zero_shot.md) explain threshold selection, metric definitions, native readout, reproducibility, and remaining risks. [Test predictions](results/zero_shot/primary/test_predictions.csv) and [metrics](results/zero_shot/primary/metrics.json) are saved with model provenance. The secondary target-holdout mode is supported separately but has not been evaluated. No fine-tuning has begun.

## Section 6: LoRA training and Colab wrapper

**CPU preflight passed; GPU smoke execution is pending in Colab. No training has run and no trained adapter exists yet.**

```bash
uv sync --frozen --extra dev --extra training
uv run --frozen --extra training python scripts/04_train_lora.py --mode preflight
# In an NVIDIA GPU runtime only:
uv run --frozen --extra training python scripts/04_train_lora.py --mode smoke --download-model
```

[Open the Colab wrapper](notebooks/finetune_colab.ipynb), fill in the published repository URL (this checkout has no remote), select a CUDA GPU, and leave `RUN_FULL_TRAINING = False`. The notebook installs the project, prepares missing data, runs the two-step smoke command, and optionally stores outputs on Google Drive. It delegates all training logic to Python.

LoRA uses rank 16, alpha 32, dropout .05 and the reference language attention/convolution/MLP projections, with the vision tower frozen: **24,461,312 trainable parameters**. Actual TRL completion-only labels and padding masks were verified on all **754 training and 161 validation examples**. Test data is never opened by the training implementation. Full training requires `--mode full --approve-full-run`; it is not automatically launched. CUDA smoke must still verify finite updates, actual memory use, and adapter save/reload.

[Training instructions, hardware requirements, and documented deviations](docs/training.md) explain precision selection, batch settings, persistent outputs, loss calculation, and the remaining GPU acceptance checks. [CPU preflight metadata](results/training/primary_preflight/metadata.json) records the checks actually executed. Use the `evaluation` extra as well as `training` and `dev` when running the complete test suite.
