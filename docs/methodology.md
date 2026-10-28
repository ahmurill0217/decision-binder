# Milestone 1 methodology

This document describes preprocessing, independently of subsequent model evaluation. The pooled comparator reproduces preprocessing counts only. Our original and fine-tuned d1-3B scores come from a new grouped random split; they do not reproduce the source article’s reported model performance.

## Cohort and labels

Load the two named tabular subsets with `datasets.load_dataset`, split `full`, at a fixed commit. Snapshot them locally and record SHA-256 checksums. Cache identity/checksum mismatch fails instead of silently reusing unrelated data. Adaptyv calls come from `design_summary`, consistent with the reference experiment; online vendor result tables can differ in details.

The default endpoint is binding among designs with resolved binder/non-binder calls, excluding failed expression. Including `not_expressed` instead changes the endpoint toward combined expression-and-binding success. Both policies are explicit; neither uses missing labels as negatives. No class balancing or subsampling occurs.

## Aggregation and deliberate correction

The reference notebook filters Boltz-2, discards grouping context, aggregates by UUID, inner-joins labels/metadata, and drops remaining NA values. It excludes `not_expressed`. The `author_pooled_comparison` report reproduces these rules but is not the main dataset.

The main calculation uses blank primary `target_form` and one prescribed stoichiometry per target: Cas9 `rnp`, TNFa `1to3`, VEGF-A/15-PGDH/Mature GDF-8/Latent GDF-8 `1to2`, other released targets `1to1`. These follow the release's bundled design-complex convention. Additional multibinder, counter-screen, and alternate-antigen folds cannot be pooled. Unknown targets fail for review. This is an explicit deviation required to avoid mixing biological contexts; it is not claimed as an exact reproduction of Ford's features.

Aggregate the five seeds using pandas min/max/mean/median and sample standard deviation (`ddof=1`). Null metric observations are skipped independently; singleton standard deviation stays null. Keep only complete aggregated feature records, with no imputation or normalization. The current main cohort has five selected rows per design. Record per-metric observation counts so missingness cannot masquerade as five measurements.

No model outcome, affinity, or expression field participates in selecting forms, seeds, or features. DockQ compares computational predictions to computational design models; it is not agreement with an experimentally solved structure.

## Validation and remaining limitations

Tests cover aggregation values, sample standard deviation, alternate-context exclusion, label states, missing values, singleton groups, deterministic row ordering, duplicate keys, target consistency, nonfinite metrics, and an injected-outcome leakage test. Local snapshot hashes and package versions are written to the run manifest. `model_inputs()` selects exactly the 85 allowed columns.

Exact UUID and prediction-key duplicates fail. Sequence screening covers every source design, across targets, using at least 90% identity at equal length without alignment. It finds 36 related pairs, 24 fully retained, and no identical sequences. Step E keeps connected components of these pairs together, including links through excluded designs. Variable-length homologs, shared scaffolds, and common design ancestry may be missed. The implemented screen is not an alignment/lineage-aware clusterer. Step E also provides a separate target-holdout experiment; see [split methodology](splits.md).

Complete-case selection preferentially removes designs lacking a comparable design structure. Metadata can encode target, generator, and campaign biases; class proportions differ strongly by target. Primary folds may differ from actual assay antigen forms, especially Nipah-G and RBX1. Treat this as a computational feature experiment, not a universal assay predictor. Potential pretraining exposure of a general model to this public dataset is unmeasured. These caveats remain even though direct experimental-input leakage is blocked.

See the dataset [data notes](https://huggingface.co/datasets/Anthropic/claude-protein-binder-design/blob/9e1b81696da46835e9e9cde9a3da976e0abc92ab/data/docs/DATA_NOTES.md) and [in-silico documentation](https://huggingface.co/datasets/Anthropic/claude-protein-binder-design/blob/9e1b81696da46835e9e9cde9a3da976e0abc92ab/data/docs/INSILICO.md).
