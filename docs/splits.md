# Step E: split protocol

The primary experiment uses grouped random allocation at seed 1337, targeting 70% training, 15% validation, and 15% test. The separate secondary experiment holds entire targets out. Splitting never changes features or labels and does not train a model.

## Primary allocation

Compute sequence links at the configured identity threshold (default 90% position-wise identity for equal-length sequences). Build transitive connected components across all 1,440 source designs before selecting the 1,076 retained records. Thus A and C stay together if an excluded design B links them. Links are not restricted to matching targets. Every component is indivisible; singleton designs are their own components. Stable UUID sorting and canonical component IDs make input row order irrelevant.

For 64 seeded random group orderings, greedily assign each group to reduce the sum of squared deviations from requested class counts. Deviations for each class are normalized by its full-cohort count. Prefer candidates with both classes in every split, then the smallest count deviation; ties retain the first candidate. This uses labels only for stratification, never model performance or feature values. Group preservation and nonempty splits take priority over exact proportions. This is a heuristic, not a proof of globally optimal allocation. Missing classes are explicitly reported if encountered. Fewer than three independent components fails rather than splitting relatives.

The resulting primary allocation is 754/161/161 records, with 206/44/44 binders. All 15 retained targets appear in each split. Label stratification is global, not target-by-label stratification: targets with only one positive cannot have positives in all splits. Mature GDF-8 has no retained records and is absent throughout.

## Separate unseen-target experiment

`--mode target_holdout` writes an independent experiment with disjoint targets across all three partitions. The default seeded allocator balances total row counts only; it does not use binding outcomes to choose held-out targets. Targets connected by a detected sequence component are assigned together. Consequently target and related-sequence separation both hold. With seed 1337, validation contains 15-PGDH, Cas9, and PD-L1; test contains EGFR and TrkA. Training contains the remaining ten targets. Counts are 755/159/162, with 218/47/29 positives. Class rates differ naturally across target sets.

To specify targets explicitly, edit both `splits.target_holdout.validation_targets` and `test_targets` in the YAML, for example:

```yaml
target_holdout:
  validation_targets: [15-PGDH, Cas9, PD-L1]
  test_targets: [EGFR, TrkA]
```

Both lists must be nonempty, known, and disjoint; remaining targets form training. A selection that separates a sequence component or leaves training empty fails. Explicit targets override ratio optimization. Empty lists use automatic selection. Seed, fractions, and number of search trials live in the same config. Freeze these choices before model experiments; do not choose another seed or held-out target set after seeing model test scores.

## Files and validation

Run `python scripts/02_create_splits.py` for primary and append `--mode target_holdout` for secondary. Source Parquet hashes, dataset identity, and preprocessing settings are checked against the preprocessing manifest before splitting. Each mode writes:

- `data/processed/splits/<mode>/{train,validation,test}.parquet`: original UUID/features/label schema, no extra model inputs.
- `data/processed/splits/<mode>/assignments.parquet`: audit-only UUID, target, label, sequence group, and partition.
- `results/splits/<mode>/assignments.csv`, `class_balance.csv`, and `target_representation.csv`.
- `results/splits/<mode>/report.json`: configuration, input/output/code hashes, grouping summary, class/target counts, and checks.

Rerunning replaces files only in the selected mode. Primary and secondary can contain the same designs but serve different experimental questions; they must have separate model artifacts and metrics later. Tests cover reproducibility under input reordering, transitive links through excluded designs, group and UUID isolation, target disjointness, label-independent target selection, invalid explicit targets, insufficient groups, rare labels, and overlap detection. Integration validation verifies saved partitions reconstruct the source without row changes and have identical hashes on rerun.

## Remaining leakage and interpretation risks

Same-length identity is a practical initial screen. It can miss indels, shorter shared motifs, remote homology, common scaffolds, and shared design ancestry; no full sequence alignment or provenance clustering is performed. Related targets may also share structural families, so holding out a target does not imply holding out its protein family. Sequence components cannot guarantee biological independence.

The primary experiment shares targets and design generators across partitions. It measures performance on more designs from the represented target distribution, not unseen-target generalization. Its per-target positives are sparse, and some target/split combinations have none. Complete-case preprocessing bias and possible base-model exposure to public data remain. No scaling or learned preprocessing is fit at this stage; any later learned transformation must be fit on training alone. Test labels are used here only for the requested partition balance reporting/primary stratification, not for hyperparameter tuning.

## Primary target representation

Cells below are records (binders). Generated from the saved split audit.

| Target | Train | Validation | Test |
|---|---:|---:|---:|
| 15-PGDH | 17 (1) | 6 (0) | 6 (0) |
| BBF-14 | 61 (1) | 8 (0) | 9 (0) |
| BHRF1 | 42 (11) | 10 (0) | 7 (1) |
| Cas9 | 44 (4) | 10 (2) | 5 (1) |
| EGFR | 55 (6) | 10 (1) | 17 (2) |
| IL-7Ra | 52 (30) | 14 (7) | 12 (8) |
| Latent GDF-8 | 34 (7) | 8 (3) | 10 (3) |
| MBP | 61 (0) | 9 (0) | 16 (0) |
| Nipah-G | 44 (13) | 6 (1) | 13 (3) |
| PD-L1 | 54 (29) | 12 (7) | 5 (3) |
| RBX1 | 51 (16) | 14 (4) | 13 (3) |
| TNFa | 88 (2) | 17 (0) | 13 (2) |
| TREM2 | 47 (39) | 12 (10) | 8 (7) |
| TrkA | 55 (15) | 12 (2) | 13 (3) |
| VEGF-A | 49 (32) | 13 (7) | 14 (8) |
