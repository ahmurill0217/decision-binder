# Reproducibility and scientific integrity

## What was reproduced

This project reproduces aspects of Ford's preprocessing: the pooled comparator yields 1,082 records (294 binders / 788 non-binders). Our main preprocessing deliberately changes target-form/stoichiometry handling and yields 1,076 records (294 / 782). This is **not a reproduction of the original article's model performance**. Its zero-shot models/cohort are not this experiment's d1-3B and 161-example held-out test set. All base-versus-adapter comparison numbers here are independently measured on our own split.

Counts: 1,440 source designs; 205 excluded by label policy (61 not expressed, 100 not tested, 44 inconclusive); 1,235 resolved labels; 159 further exclusions for incomplete aggregate features; 1,076 retained. Primary train/validation/test: 754/161/161, with 206/44/44 binders. Full audit records are in `results/exclusions.csv` and `results/preprocessing_report.json`. Missing outcomes are never relabeled as negative. Per-metric missing seed observations are skipped independently; sample standard deviations require at least two observations. Complete-case filtering can bias the retained population.

## Environment and artifacts

Use Python 3.12 and `uv sync --frozen` with `uv.lock`, not an unlocked pip install, for the tested dependency resolution. Optional extras select dependency groups; the lock includes their transitive versions. The tested local uv version is 0.12.1. The lock is the dependency authority; upgrading it creates a new environment that must be revalidated.

Dataset revision: `9e1b81696da46835e9e9cde9a3da976e0abc92ab`. Model revision: `051bcc464b01b9f92942b364d9586b0ef5912432`. Split, prompt-option and training seeds: 1337. Configuration lives in `configs/experiment.yaml`; completed training includes the exact configuration, trainer arguments, versions, hardware, source checksums, optimizer step/skip records and losses in `results/training/primary_full/metadata.json`. This is a byte-for-byte copy of the supplied Colab record, not a rerun or invented metadata. Its preflight predecessor remains labeled CPU-only.

Evaluation manifests preserve source split hash, model assets, code hashes, seed, batch size, device and dtype. The historical manifests do not contain a full config snapshot; the audit adds explicitly retrospective snapshots linked to their original manifest hashes. This must not be mistaken for original run-time capture. Future evaluator runs save the full config and detailed hardware directly.

Ignored artifacts include raw tables, processed cohorts, SFT datasets, downloaded models, LoRA adapters, optimizer/checkpoints and `full/`. Small numerical results, prediction records, UUID split audits and figures are intentionally retained for independent reanalysis. `.gitignore` does not untrack already committed files; the repository hygiene test inspects the Git index and rejects tracked training data, weights/checkpoints or files over 10 MiB. No history rewriting occurs.

## CPU tests without datasets or model downloads

```bash
uv sync --frozen --extra dev --extra evaluation --extra training
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run --frozen --extra dev --extra evaluation --extra training pytest -m 'not integration' -q
```

Installing dependencies requires a package cache or network access; running these tests does not. Unit tests use small in-memory designs, predictions, splits, tokenizer doubles and native probability responses. Network connections are rejected in unit tests. The GitHub Actions workflow runs this command on CPU with no dataset/model download step. It is configured, not yet claimed to have passed remotely.

Coverage includes deterministic aggregation under shuffled rows, one record per UUID/duplicate rejection, label policy, split/sequence-component/target separation, outcome-column injection, deterministic option ordering, answer-code roundtrip, BOS/EOS and boundary handling, completion-only labels/padding, and mapping probabilities by choice name rather than position. Synthetic tokenizer tests validate application contracts, not the real d1 vocabulary. Native tokenizer and saved-dataset integration tests are explicitly marked `integration`, require pre-existing assets, and can be run with `pytest -m integration` after preparing those assets. No test loads full model weights or trains on GPU.

## Limits on claims

- The primary split groups detected near-duplicate sequences but shares targets across train/test. It measures performance within this retained mixture; **it does not demonstrate generalization to unseen targets**. The separate target-holdout split exists but has not been evaluated.
- Equal-length positional-identity screening can miss variable-length homology, shared scaffolds and design lineage. Campaign, generator and target metadata may be predictive confounders. Experimental assay/antigen context may differ from computational design folds.
- Failed expression and missing features are excluded, changing the endpoint/population. Report target/class representation rather than implying a balanced sample.
- Possible pretraining exposure to the public dataset is unknown. The 161-example test results are single-split point estimates, with no confidence intervals or replicated-seed generalization claim.
- Both evaluated models use the exact original local validation/test UUIDs, labels and features, identical native decision decoding, and MPS/bf16. Each threshold is selected on validation only. The test set is now evaluated and must not be reused for repeated model/threshold selection while still being described as untouched.
- Colab regenerated training SFT files whose hashes differ from local files. UUIDs and dataset/preprocessing/split/model/prompt-seed configs match, but exact training feature/prompt equality is unverified. Do not call this byte-identical cross-platform reproduction. Preserve the Colab source Parquets and compare by UUID/value/token sequence to resolve it; serialization and floating-point reduction differences are possibilities, not established causes.
- Seeds do not guarantee bitwise-identical GPU training across hardware/software. No deterministic CUDA-kernel guarantee is claimed. Saved probability records support exact downstream metric recalculation without retraining.

See [licensing and attribution](licensing.md), [methodology](methodology.md), and [fine-tuned evaluation](fine_tuned_evaluation.md).
