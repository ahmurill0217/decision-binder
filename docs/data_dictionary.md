# Data dictionary

Source: [rendered column dictionary](https://huggingface.co/datasets/Anthropic/claude-protein-binder-design/blob/9e1b81696da46835e9e9cde9a3da976e0abc92ab/data/docs/COLUMNS.md). Exact names are validated before processing.

## Model inputs: 85 columns

Five metadata fields are used: `target` (categorical target code), `binder_length` (residues), `generator` (design generator), `sequence_design_method` (sequence designer), and `epitope_n_residues` (computational epitope size). The epitope field may derive from a co-fold fallback rather than exclusively design-time coordinates.

Each metric below produces five columns named `<metric>_<stat>` where `stat` is exactly `min`, `max`, `mean`, `std`, or `median`, giving 80 numerical features. The expanded list is persisted with every run.

| Metric | Meaning / units |
|---|---|
| `sc_dockq` | Predicted/design complex agreement |
| `dockq_fnat` | Recovered reference contact fraction |
| `dockq_irms` | Interface RMSD, Å |
| `dockq_lrms` | Binder RMSD after target alignment, Å |
| `ipsae_min` | Lower directional ipSAE |
| `ipsae_max` | Higher directional ipSAE in selected primary folds |
| `ipsae_binder_to_target` | Directional ipSAE |
| `ipsae_target_to_binder` | Reverse directional ipSAE |
| `iptm_pae` | Predictor interface confidence |
| `ptm_pae` | Predictor global confidence |
| `lis` | Local interaction score |
| `plddt_binder` | Binder confidence, 0–100 |
| `plddt_target` | Target confidence, 0–100 |
| `pae_interface_mean` | Mean interface PAE, Å |
| `pae_interface_min` | Minimum interface PAE, Å |
| `n_interface_contacts` | Cα-based interface residue-pair count |

## Identity, labels, and audit-only fields

`uuid` joins the tables and identifies each output row; it is not a model feature. `label` is 1 for `binder` and 0 for `non_binder`. `not_expressed` defaults to exclusion; setting `preprocessing.not_expressed: non_binder` explicitly maps it to 0. Missing, unknown, `not_tested`, and `inconclusive` labels are excluded under either setting.

`sequence` is read only to verify lengths and audit related designs. `cofolding_model`, `stoichiometry`, `target_form`, and `seed` select and validate prediction groups. None is passed through implicitly. Other design metadata remains in the raw snapshot only.

All experimental affinity, kinetics, expression, vendor agreement, combined final assessments, free-text notes, paths, and unknown future columns are excluded by the input allowlist. Audit files contain outcomes and must never be treated as model input tables.
