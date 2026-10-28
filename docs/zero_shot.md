# Section 5: original d1-3B baseline

The primary grouped-random experiment was evaluated with the original `LiquidAI/d1-3B`, revision `051bcc464b01b9f92942b364d9586b0ef5912432`. No adapters, weight updates, few-shot demonstrations, or fitted probability calibration were used. Validation labels select a decision threshold; this is zero-shot model inference with a validation-selected classification rule, not an entirely label-free evaluation protocol.

## Run

```bash
uv sync --frozen --extra dev --extra evaluation
# First run: download the original checkpoint and its pinned inference code.
uv run --frozen --extra evaluation python scripts/03_evaluate_base.py --download-model --device mps
# Subsequent runs reuse the local snapshot and resume verified checkpoints.
uv run --frozen --extra evaluation python scripts/03_evaluate_base.py --device mps
uv run --frozen --extra dev --extra evaluation pytest -q
```

`--device auto` chooses MPS, then CUDA, then CPU; CPU uses float32 and accelerators use bfloat16, following the pinned model card's example. The completed run used Apple MPS, bfloat16, PyTorch 2.11.0, Transformers 5.19.0, and batch size one. This step needs the roughly 6 GB checkpoint; the earlier CPU-only formatting step still needs only tokenizer assets. Training dependencies TRL and PEFT are not installed. If an execution sandbox hides the Apple GPU, the host must permit local GPU access; the implementation does not silently switch an explicit `--device mps` request to another backend.

`--mode target_holdout` supports the secondary experiment in a separate output directory with its own validation-selected threshold. **Only the primary experiment has been evaluated in this milestone.** Do not combine the modes' predictions or thresholds.

## Native decision readout and prompt parity

Inspected the pinned model's `modeling_d1.py`, `api.py`, `runner.py`, `hybrid.py`, and `lfm2_vl.py`, together with its [model card](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/README.md). The supported [System One runner](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/runner.py) performs a forward pass at the assistant answer position. Its choice readout scores each option's accepted token forms, takes the highest score within each option, and normalizes across options. We use `model.system_one(state, {"binding": question})` and extract `answers.binding.probabilities["Binder"]` by name, never by option position. We do not call `generate()` or parse generated prose.

The shared Section 4 builder selects the same allowed state fields and seeded per-UUID option ordering. Before every model call, the actual engine-rendered string must exactly equal the SFT formatter's prompt. The response must contain finite binary probabilities summing to one and zero generated tokens. The final `predicted_class` uses the locked validation threshold; `native_choice` is preserved separately because the API's default winner need not agree with that threshold.

## Threshold policy established before test inference

Score all 161 validation records first. Evaluate thresholds consisting of their unique probabilities plus 0, 0.5, and 1. Select maximum balanced accuracy; scores tied to 12 decimals prefer the threshold closest to 0.5 and then the higher threshold. Predict binder when probability is **greater than or equal to** the chosen threshold. Fail if validation lacks either class.

The selected threshold is **0.7431680086124811**, with validation balanced accuracy **0.6231546** across 38 candidate thresholds. Its policy and validation probability checksum were saved in `threshold.json` before test inference began. No test observations affect selection. A fixed 0.5 comparison is reported for transparency, not chosen after looking at test performance. Validation performance at the selected threshold is development performance and is optimistically selected; test is the held-out estimate.

## Primary held-out results

Test set: **161 records, 44 binders and 117 non-binders**.

| Metric | Validation-selected threshold | Fixed 0.5 comparison |
|---|---:|---:|
| ROC-AUC | 0.622183 | 0.622183 |
| PR-AUC (average precision) | 0.350910 | 0.350910 |
| PR-AUC (trapezoidal) | 0.351004 | 0.351004 |
| Precision | 0.357143 | 0.293333 |
| Recall | 0.454545 | 1.000000 |
| F1 | 0.400000 | 0.453608 |
| Balanced accuracy | 0.573427 | 0.547009 |
| Accuracy | 0.627329 | 0.341615 |
| Always non-binder accuracy | 0.726708 | 0.726708 |

Confusion matrix at the selected threshold:

| Actual / Predicted | Non-binder | Binder |
|---|---:|---:|
| Non-binder | 81 | 36 |
| Binder | 24 | 20 |

The trivial class is determined from **training** labels (548 non-binders versus 206 binders) and then evaluated on test: 117/161 = 72.67% accuracy. The model has modest ranking discrimination and balanced accuracy above the trivial classifier's 0.5, but lower overall accuracy than that classifier. This is not evidence that SFT will improve it; no fine-tuned comparison exists yet.

Average precision is the primary PR-AUC summary (recall-increment-weighted precision); the trapezoidal precision-recall integral is separately labeled because the definitions differ. Rankings use probabilities independently of the classification threshold and handle ties. Undefined precision/recall/F1 denominators produce zero; single-class per-target groups report ROC-AUC, PR-AUC, and balanced accuracy as null with an explicit warning. Target-level estimates are sparse and exploratory.

## Artifacts and checks

All files are under `results/zero_shot/primary/`:

- `validation_predictions.parquet` / `.csv` and `test_predictions.parquet` / `.csv`: UUID, ground-truth label, predicted class (0/1), binder probability, target, model identifier/revision, threshold, native choice, split, and input-token count.
- `validation_decisions.jsonl`, `test_decisions.jsonl`: incremental native decisions for safe resumption. Invalid/duplicate/unknown records fail rather than silently re-score or overwrite. A partial final JSON line must be repaired explicitly before resume; it is not silently discarded.
- `threshold.json`: validation-only selection policy, threshold, objective, and source checksum.
- `metrics.json`: full validation/test metrics, fixed-0.5 comparisons, per-target metrics, and definitions.
- `run_manifest.json`: split/source reference, hashes of every local model file including weights, evaluation/formatting/metric code hashes, dtype, device, batch size, and library versions. Changed run identity refuses to mix with existing checkpoints; preserve/move the previous experiment output before running a changed configuration.

Tests cover metric calculations, positive-class extraction independent of option order, threshold selection and ties, rare/invalid labels and probabilities, native-response validation, exact prompt parity checks, and checkpoint resumption. Post-run artifact checks match every predicted UUID/label/target to its saved source split, recompute threshold from validation only and metrics from saved predictions, and verify completeness and source/code hashes. Accelerator floating-point output is not promised to be bit-identical across devices, package versions, or batch shapes; record provenance when comparing future runs. The deterministic SFT formatting guarantee is unaffected.

No Section 6 fine-tuning work has begun. The remaining scientific limits include the small held-out cohort, imperfect related-sequence screening, target/generator correlations, possible public-data pretraining exposure, and the lack of an independent external cohort. Native output probabilities have not been validated as calibrated probabilities of experimental binding in this domain.
