import numpy as np
import pytest

from qsar_screen import build_pipeline, train
from qsar_screen.model import ad_tier, load_scorer, save_scorer, score_smiles


def test_binary_pipeline_omits_scaling_and_pruning():
    steps = dict(build_pipeline("rf", binary_features=True).steps)
    assert "prune" not in steps and "scale" not in steps
    dense = dict(build_pipeline("rf", binary_features=False).steps)
    assert "prune" in dense and "scale" in dense


def test_no_univariate_selection_step():
    assert "select" not in dict(build_pipeline("rf").steps)


def test_train_produces_a_usable_scorer(smiles, activity):
    scorer = train(smiles, activity)
    assert scorer.metrics["n_train"] == len(smiles)
    assert scorer.metrics["n_active"] == int((activity < 65).sum())
    assert 0 < scorer.n_features <= len(scorer.descriptor_names)
    probs = scorer.predict_proba(smiles)
    assert probs.shape == (len(smiles),)
    assert np.all((probs >= 0) & (probs <= 1))


def test_training_molecules_are_in_domain(smiles, activity):
    scorer = train(smiles, activity)
    np.testing.assert_allclose(scorer.applicability(smiles), 1.0)


def test_ad_tier_boundaries():
    assert ad_tier(0.9) == "in_domain"
    assert ad_tier(0.50) == "in_domain"
    assert ad_tier(0.40) == "edge"
    assert ad_tier(0.29) == "out_of_domain"
    assert ad_tier(float("nan")) == "unparseable"


def test_round_trip_through_disk(tmp_path, smiles, activity):
    scorer = train(smiles, activity)
    save_scorer(scorer, str(tmp_path))
    reloaded = load_scorer(str(tmp_path))
    np.testing.assert_allclose(scorer.predict_proba(smiles),
                               reloaded.predict_proba(smiles))
    assert reloaded.n_features == scorer.n_features


def test_missing_model_raises_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/train.py"):
        load_scorer(str(tmp_path / "empty"))


def test_score_smiles_is_sorted_and_annotated(tmp_path, smiles, activity):
    save_scorer(train(smiles, activity), str(tmp_path))
    frame = score_smiles(smiles, model_dir=str(tmp_path))
    assert list(frame.columns) == ["smiles", "prob_active", "nn_tanimoto", "ad_tier"]
    assert frame.prob_active.is_monotonic_decreasing
    assert (frame.ad_tier == "in_domain").all()


def test_single_class_training_is_rejected(smiles, activity):
    """A one-class fit succeeds silently but breaks at scoring time."""
    all_inactive = np.full(len(smiles), 90.0)
    with pytest.raises(ValueError, match="need both classes"):
        train(smiles, all_inactive)
    all_active = np.full(len(smiles), 10.0)
    with pytest.raises(ValueError, match="need both classes"):
        train(smiles, all_active)


def test_threshold_shifts_the_label_boundary(smiles, activity):
    strict = train(smiles, activity, threshold=50.0)
    relaxed = train(smiles, activity, threshold=70.0)
    assert strict.metrics["n_active"] < relaxed.metrics["n_active"]
    assert strict.metrics["activity_threshold"] == 50.0
