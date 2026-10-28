# Conventional tabular baseline

The CPU logistic-regression baseline tests whether d1 fine-tuning adds value beyond a conventional classifier on the same information. It uses the same 85 allowlisted inputs, 754 training records, 161 validation records and 161 test records as the primary d1 experiment. The baseline was added after d1 test results were known; this is an exploratory comparison, not a preregistered independent replication.

## Fixed protocol

- Numerical columns: `StandardScaler`, fit on training only. Includes binder length, epitope size and the 80 aggregated structural metrics.
- Categorical columns: target, generator and sequence-design method, one-hot encoded with training-only category discovery and `handle_unknown='ignore'`.
- Classifier: L2 logistic regression, `C=1`, LBFGS, `max_iter=2000`, tolerance `1e-4`, seed 1337. No class weighting, imputation, interaction engineering, feature selection or hyperparameter search. Convergence warnings fail the run rather than accepting an unverified fit.
- Only training labels are used for fitting. Numerical/categorical transformations are not refit on validation or test. Inputs pass through the existing explicit feature allowlist.
- Positive probability is selected by fitted class label `1`, not an assumed column position.
- The same balanced-accuracy threshold rule used by d1 is applied to validation predictions. Its threshold is saved before test probabilities are computed. Fixed-0.5 metrics are also saved for context.
- Exact UUID/label/target cohort parity and source checksums are checked. Split-overlap/group checks reuse the existing split validator.

Settings are in the `tabular` section of `configs/experiment.yaml`. All dependencies are already in the locked evaluation environment; no new model download or GPU is required.

## Run and reproduce

After creating the splits and baseline predictions:

```bash
.venv/bin/python scripts/07_evaluate_tabular.py
.venv/bin/python scripts/06_compare_results.py \
  --training-metadata results/training/primary_full/metadata.json
```

Both commands support `--config` and `--mode`. If using `configs/quickstart.local.yaml`, regenerate that local configuration from the updated experiment config or add the identical `tabular` section before running. Target-holdout evaluation requires its own baseline records and remains separate.

`results/logistic_regression/primary/` contains validation/test CSV and Parquet predictions, threshold, metrics, and a run manifest with parameters, source hashes, versions, hardware and protocol notes. The fitted scaler, encoder and classifier are saved together in ignored `artifacts/tabular/primary/pipeline.joblib`; the saved model is reloaded and probabilities checked for exact equality. Only load trusted joblib artifacts. Nonempty results and an existing fitted pipeline are protected from overwrite; preserve or move both before rerunning.

The comparison generator automatically includes logistic regression when its completed metrics exist. The [generated table](../results/comparison/primary/metrics.md), ROC/PR curves and confusion matrices then include all three models. Rebuilding those comparisons uses only saved records and does not refit any model.

## Interpretation

On the completed primary experiment, logistic regression has higher ROC-AUC and average precision than the fine-tuned d1 model, while their F1 and balanced accuracy are close. The adapter has higher recall at the respective validation-selected thresholds; logistic regression has higher precision. Exact values are in the generated artifacts.

Thus the evidence supports improvement from d1 fine-tuning over d1 zero-shot, but **does not establish added predictive value over conventional tabular ML**. We have not tested statistical significance, tuned an optimized tabular baseline, evaluated unseen-target generalization, or performed an independent replication. No XGBoost search or repeated test-guided tuning was performed. The training-file provenance limitation of the d1 comparison remains in force.
