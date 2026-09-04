"""Scaffold-split cross-validation, metrics, and validation tests."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from qsar_screen.model import RANDOM_SEED, build_pipeline, estimator

N_SPLITS = 5


def scaffold_cv(X, y, groups, kind: str = "rf", binary_features: bool = False,
                n_splits: int = N_SPLITS, seed: int = RANDOM_SEED):
    """Out-of-fold probabilities with scaffold-disjoint folds.

    Grouping by scaffold keeps analogue series on one side of every split; a
    random split would place near-duplicates in both train and test.
    """
    X, y = np.asarray(X), np.asarray(y)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    widths = []
    for train_idx, test_idx in cv.split(X, y, groups):
        pipe = build_pipeline(kind, binary_features)
        if kind == "xgb":
            pipe.set_params(clf=estimator("xgb", n_pos=int(y[train_idx].sum()),
                                          n_neg=int((1 - y[train_idx]).sum())))
        pipe.fit(X[train_idx], y[train_idx])
        oof[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
        widths.append(int(pipe.steps[-1][1].n_features_in_))
    return oof, float(np.mean(widths))


def classification_metrics(y, scores) -> dict:
    """PR-AUC, ROC-AUC, and lift over the no-skill baseline."""
    y = np.asarray(y)
    baseline = float(y.mean())
    pr = float(average_precision_score(y, scores))
    return {"pr_auc": round(pr, 4),
            "roc_auc": round(float(roc_auc_score(y, scores)), 4),
            "baseline": round(baseline, 4),
            "lift": round(pr / baseline, 3)}


def enrichment_factor(y, scores, fraction: float = 0.10):
    """Enrichment and hit count in the top ``fraction`` of the ranking."""
    y = np.asarray(y)
    k = max(1, int(round(fraction * len(y))))
    top = np.argsort(scores)[::-1][:k]
    return float(y[top].mean() / y.mean()), int(y[top].sum()), k


def permutation_test(X, y, groups, kind: str = "rf", binary_features: bool = False,
                     n_permutations: int = 200, seed: int = RANDOM_SEED) -> dict:
    """Compare the real score against a shuffled-label null.

    The whole cross-validation is refitted per permutation, so the null absorbs
    any optimism in the pipeline itself.
    """
    y = np.asarray(y)
    real, _ = scaffold_cv(X, y, groups, kind, binary_features)
    real_pr = average_precision_score(y, real)
    real_roc = roc_auc_score(y, real)

    rng = np.random.RandomState(seed)
    null_pr, null_roc = [], []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        oof, _ = scaffold_cv(X, y_perm, groups, kind, binary_features)
        null_pr.append(average_precision_score(y_perm, oof))
        null_roc.append(roc_auc_score(y_perm, oof))
    null_pr, null_roc = np.asarray(null_pr), np.asarray(null_roc)

    return {"real_pr_auc": round(float(real_pr), 4),
            "null_pr_median": round(float(np.median(null_pr)), 4),
            "p_pr_auc": round(float((np.sum(null_pr >= real_pr) + 1)
                                    / (n_permutations + 1)), 4),
            "real_roc_auc": round(float(real_roc), 4),
            "null_roc_median": round(float(np.median(null_roc)), 4),
            "p_roc_auc": round(float((np.sum(null_roc >= real_roc) + 1)
                                     / (n_permutations + 1)), 4),
            "n_permutations": n_permutations}
