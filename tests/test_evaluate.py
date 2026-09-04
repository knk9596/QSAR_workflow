import numpy as np

from qsar_screen.evaluate import (
    classification_metrics, enrichment_factor, scaffold_cv,
)


def test_metrics_on_a_perfect_ranking():
    y = np.array([0, 0, 0, 1, 1])
    m = classification_metrics(y, np.array([0.1, 0.2, 0.3, 0.9, 0.95]))
    assert m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0
    assert m["baseline"] == 0.4 and m["lift"] == 2.5


def test_random_scores_sit_near_the_baseline():
    rng = np.random.RandomState(0)
    y = (rng.rand(400) < 0.3).astype(int)
    m = classification_metrics(y, rng.rand(400))
    assert abs(m["pr_auc"] - y.mean()) < 0.08
    assert abs(m["roc_auc"] - 0.5) < 0.08


def test_enrichment_of_a_perfect_and_random_ranking():
    y = np.array([1] * 10 + [0] * 90)
    ef, hits, k = enrichment_factor(y, np.linspace(1, 0, 100), 0.10)
    assert k == 10 and hits == 10 and ef == 10.0
    ef_flat, _, _ = enrichment_factor(y, np.zeros(100) + 0.5, 1.0)
    assert ef_flat == 1.0


def test_scaffold_cv_predicts_every_row_and_finds_signal():
    rng = np.random.RandomState(1)
    n = 120
    groups = np.array([f"scaffold_{i % 20}" for i in range(n)])
    y = (rng.rand(n) < 0.3).astype(int)
    X = rng.normal(size=(n, 12))
    X[y == 1] += 1.4                       # separable signal
    oof, width = scaffold_cv(X, y, groups, "rf")
    assert oof.shape == (n,)
    assert np.all((oof >= 0) & (oof <= 1))
    assert width <= 12
    assert classification_metrics(y, oof)["roc_auc"] > 0.7


def test_scaffold_folds_are_disjoint_by_group():
    from sklearn.model_selection import StratifiedGroupKFold
    rng = np.random.RandomState(2)
    n = 100
    groups = np.array([f"s{i % 10}" for i in range(n)])
    y = (rng.rand(n) < 0.4).astype(int)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in cv.split(np.zeros((n, 1)), y, groups):
        assert not set(groups[train_idx]) & set(groups[test_idx])
