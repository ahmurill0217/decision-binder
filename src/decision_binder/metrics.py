"""Binary metrics and validation-only threshold selection."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)


def validate_scores(labels, probabilities):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError("Labels and probabilities must be equal-length nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary labels and finite probabilities in [0,1]")
    return y.astype(int), p


def choose_threshold(labels, probabilities):
    """Call with validation only; never inspect test labels for selection."""
    y, p = validate_scores(labels, probabilities)
    if len(np.unique(y)) < 2:
        raise ValueError("Validation requires both classes for balanced-accuracy selection")
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], p)))
    scored = [(float(balanced_accuracy_score(y, p >= t)), float(t)) for t in candidates]
    score, threshold = max(scored, key=lambda pair: (round(pair[0], 12), -abs(pair[1] - .5), pair[1]))
    return {"threshold": threshold, "selection_split": "validation",
            "objective": "balanced_accuracy", "validation_objective": score,
            "rule": "predict binder when probability >= threshold",
            "candidates": "unique validation probabilities plus 0, 0.5, 1",
            "tie_break": "scores equal to 12 decimals: closest to 0.5, then higher threshold",
            "candidates_evaluated": len(candidates)}


def classification_metrics(labels, probabilities, threshold, majority_class):
    y, p = validate_scores(labels, probabilities)
    if not 0 <= threshold <= 1 or majority_class not in (0, 1):
        raise ValueError("Invalid threshold or majority class")
    predicted = (p >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    precision, recall, _ = precision_recall_curve(y, p) if both else (None, None, None)
    return {
        "records": len(y), "binders": int(y.sum()), "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc_average_precision": float(average_precision_score(y, p)) if both else None,
        "pr_auc_trapezoidal": float(auc(recall, precision)) if both else None,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if both else None,
        "accuracy": float(accuracy_score(y, predicted)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "confusion_matrix_layout": "rows=true [non_binder,binder], columns=predicted [non_binder,binder]",
        "majority_class_from_training": int(majority_class),
        "majority_class_accuracy": float(np.mean(y == majority_class)),
        "warnings": [] if both else ["Single-class cohort: ROC/PR AUC and balanced accuracy reported as null"],
    }
