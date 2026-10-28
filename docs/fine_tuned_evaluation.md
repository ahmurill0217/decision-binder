# Fine-tuned decision evaluation

Run the completed full-training adapter against the original local baseline cohort:

```bash
uv sync --frozen --extra evaluation --extra training
uv run --frozen --extra evaluation --extra training python scripts/05_evaluate_lora.py \
  --adapter full/adapter --device mps
```

The adapter directory must retain its parent `metadata.json`. CUDA (including fp16 T4), MPS (bf16), and CPU (fp32) are supported. The completed run uses MPS/bf16, matching the original baseline. The CLI loads the pinned base snapshot, attaches the frozen PEFT adapter without merging weights, and calls the native D1Model `system_one` API through its adapted layers. No generation, calibration fitting, or training occurs.

Before inference, it checks adapter checksums, successful full-training/reload status, base revision, configuration, train/validation UUIDs, and exact validation/test UUID/label/target parity with saved baseline predictions. Native rendered prompts must equal the shared SFT formatter for every row. Training SFT hashes are recorded separately: the provided Colab run regenerated files with different hashes from the local SFT artifacts. Configurations and UUIDs match, but exact training feature/prompt equality is not established without the Colab source Parquets. This is a documented limitation, not a reason to replace the baseline evaluation cohort.

Each model independently uses the same validation-only balanced-accuracy threshold selection rule, including the original tie-break. The adapter threshold is saved before test inference. Both validation and test prediction records retain UUID, label, probability, class, target, base model identifier/revision, and adapter weight hash. Native answer codes keep their per-UUID shuffled meaning. Test labels are never used to select thresholds.

Outputs:

- `results/fine_tuned/primary/`: resumable decision JSONL, prediction CSV/Parquet, threshold, metrics and manifest.
- `results/comparison/primary/metrics.csv` and `metrics.json`: table, source hashes, hardware and provenance limitations.
- `results/figures/primary/`: ROC, precision–recall, confusion matrices, class distribution and training/validation loss in PNG and SVG.

PR-AUC in the comparison is average precision, matching the baseline; the trapezoidal integral remains separately available. Confusion matrices use true labels on rows and predicted labels on columns, non-binder first. Loss curves use training log averages and validation evaluations, rather than interpolated batch observations. The final `train_loss` aggregate is not a separate training step.

Rebuild figures and metrics without model loading or retraining:

```bash
uv run --frozen --extra evaluation python scripts/05_evaluate_lora.py \
  --adapter full/adapter --plots-only
```

To run in the existing Colab runtime after pushing these changes and pulling them there:

```python
subprocess.run(["uv", "sync", "--frozen", "--extra", "training", "--extra", "evaluation"], cwd=repo, check=True)
subprocess.run([python, "scripts/05_evaluate_lora.py", "--adapter", str(full_dir / "adapter"), "--device", "cuda"], cwd=repo, check=True)
```

A Colab rerun may fail the strict baseline split-file provenance check if its regenerated processed files differ from the original baseline. Use the original local processed splits/manifests for an exact comparison; do not silently overwrite baseline results. Evaluation on a different device/precision must also be labeled as such. Copy results to Drive before closing a temporary runtime.

## Completed primary experiment

Evaluated 161 validation and 161 test candidates on Apple MPS with bf16, matching baseline hardware/precision. The test set contains 44 binders and 117 non-binders. Adapter threshold 0.2689414213699951 and baseline threshold 0.7431680086124811 were each selected on validation only.

| Metric | Base d1-3B | Fine-tuned d1-3B |
|---|---:|---:|
| ROC-AUC | 0.622183 | 0.829254 |
| PR-AUC (average precision) | 0.350910 | 0.637967 |
| Precision | 0.357143 | 0.581818 |
| Recall | 0.454545 | 0.727273 |
| F1 | 0.400000 | 0.646465 |
| Balanced accuracy | 0.573427 | 0.765346 |
| Accuracy | 0.627329 | 0.782609 |

The training-derived always-non-binder baseline achieves accuracy 0.726708. Base confusion matrix is `[[81,36],[24,20]]`; adapter matrix is `[[94,23],[12,32]]`. These are point estimates for this held-out random/grouped split, not evidence of generalization to unseen targets. The separate target-holdout experiment has not been evaluated.

Validation: 58 unit/regression tests passed. All five PNG plots were visually inspected; corresponding SVG files were also saved. Exact base asset hashes, prompt/metric code hashes, baseline cohort identity, and adapter file hashes were checked. The Colab/local training SFT hash discrepancy remains documented above and in JSON; no assertion of byte-identical training inputs is made.
