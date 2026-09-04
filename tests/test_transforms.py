import numpy as np
import pytest

from qsar_screen import CorrPrune


def test_drops_perfectly_correlated_duplicate():
    x = np.arange(20, dtype=float).reshape(-1, 1)
    X = np.hstack([x, x * 2.0, np.random.RandomState(0).normal(size=(20, 1))])
    pruner = CorrPrune(0.95).fit(X)
    assert pruner.transform(X).shape[1] == 2
    assert 0 in pruner.keep_ and 1 not in pruner.keep_   # keeps lower index


def test_keeps_independent_columns():
    X = np.random.RandomState(1).normal(size=(200, 5))
    assert CorrPrune(0.95).fit(X).transform(X).shape[1] == 5


def test_is_deterministic():
    X = np.random.RandomState(2).normal(size=(50, 8))
    X = np.hstack([X, X[:, :3] + 1e-9])
    first = CorrPrune(0.95).fit(X).keep_
    second = CorrPrune(0.95).fit(X).keep_
    np.testing.assert_array_equal(first, second)


def test_transform_before_fit_raises():
    with pytest.raises(AttributeError):
        CorrPrune().transform(np.zeros((3, 3)))


def test_threshold_is_respected():
    rng = np.random.RandomState(3)
    base = rng.normal(size=(300, 1))
    X = np.hstack([base, base * 0.9 + rng.normal(scale=0.5, size=(300, 1))])
    assert CorrPrune(0.99).fit(X).transform(X).shape[1] == 2
    assert CorrPrune(0.50).fit(X).transform(X).shape[1] == 1
