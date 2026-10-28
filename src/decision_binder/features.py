"""Explicit computational input allowlist and seed aggregation."""
import numpy as np
import pandas as pd

METRICS = (
    "sc_dockq", "dockq_fnat", "dockq_irms", "dockq_lrms", "ipsae_min",
    "ipsae_max", "ipsae_binder_to_target", "ipsae_target_to_binder",
    "iptm_pae", "ptm_pae", "lis", "plddt_binder", "plddt_target",
    "pae_interface_mean", "pae_interface_min", "n_interface_contacts",
)
STATS = ("min", "max", "mean", "std", "median")
METADATA = ("target", "binder_length", "generator", "sequence_design_method", "epitope_n_residues")
FEATURES = (*METADATA, *(f"{metric}_{stat}" for metric in METRICS for stat in STATS))
STOICHIOMETRY = dict.fromkeys(
    ("BBF-14", "BHRF1", "EGFR", "IL-7Ra", "MBP", "Nipah-G", "PD-L1", "RBX1", "TREM2", "TrkA"), "1to1"
) | {"Cas9": "rnp", "TNFa": "1to3", "VEGF-A": "1to2", "15-PGDH": "1to2", "Mature GDF-8": "1to2", "Latent GDF-8": "1to2"}


def require_columns(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def select_predictions(predictions, designs, scope="primary"):
    """Never choose a target form or seed using experimental outcomes."""
    require_columns(predictions, ["uuid", "target", "cofolding_model", "stoichiometry", "target_form", "seed", *METRICS])
    p = predictions.loc[predictions.cofolding_model.eq("boltz2")].copy()
    if p.empty:
        raise ValueError("No Boltz-2 predictions")
    if p[["uuid", "target", "stoichiometry", "seed"]].isna().any().any():
        raise ValueError("Null prediction key")
    p["target_form"] = p.target_form.fillna("").str.strip()
    keys = ["uuid", "cofolding_model", "stoichiometry", "target_form", "seed"]
    if p.duplicated(keys).any():
        raise ValueError("Duplicate prediction keys would reweight seeds")
    expected = designs.set_index("uuid").target
    if not p.target.eq(p.uuid.map(expected)).all():
        raise ValueError("Unknown UUID or inconsistent target in predictions")
    if scope == "primary":
        wanted = p.target.map(STOICHIOMETRY)
        if wanted.isna().any():
            raise ValueError("Unmapped target: review stoichiometry")
        p = p.loc[p.target_form.eq("") & p.stoichiometry.eq(wanted)].copy()
    elif scope != "author_pooled":
        raise ValueError(f"Unknown prediction scope: {scope}")
    for metric in METRICS:
        p[metric] = pd.to_numeric(p[metric], errors="raise").astype(float)
    if np.isinf(p[list(METRICS)].to_numpy()).any():
        raise ValueError("Infinite structural metric")
    return p


def aggregate_predictions(selected):
    """Pandas skips nulls per metric; sample std uses ddof=1 (singleton => NA)."""
    grouped = selected.groupby("uuid", sort=True)[list(METRICS)].agg(list(STATS))
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def model_inputs(records):
    """The sole input selector: identifiers and outcomes cannot enter features."""
    require_columns(records, FEATURES)
    return records.loc[:, list(FEATURES)].copy()
