import numpy as np
import pytest

from qsar_screen.model import save_scorer, train
from qsar_screen.reward import components, reward

ANALOGUE = "O=C(NCCCn1cnc2ccccc21)c1cc(-c2ccc(F)cc2)[nH]n1"
GREASY = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
POLYCYCLE = "c1ccc2c(c1)ccc1c2ccc2c1ccc1c2cccc1"


@pytest.fixture
def model_dir(tmp_path, smiles, activity):
    path = tmp_path / "model"
    save_scorer(train(smiles, activity), str(path))
    return str(path)


def test_reward_is_bounded(model_dir, smiles):
    r = reward(smiles + [ANALOGUE, GREASY], model_dir=model_dir)
    assert np.all((r >= 0) & (r <= 1))


def test_invalid_smiles_scores_zero(model_dir):
    assert reward(["!!!not a molecule"], model_dir=model_dir)[0] == 0.0


def test_guardrails_suppress_adversarial_molecules(model_dir):
    r = reward([ANALOGUE, GREASY, POLYCYCLE], model_dir=model_dir)
    assert r[0] > r[1] and r[0] > r[2]


def test_training_compound_keeps_a_nonzero_gradient(model_dir, smiles):
    """A raw 1 - similarity novelty term would zero this out entirely."""
    assert reward([smiles[0]], model_dir=model_dir)[0] > 0.0


def test_disabling_novelty_favours_known_chemistry(model_dir, smiles):
    known = smiles[0]
    balanced = reward([known], model_dir=model_dir)[0]
    analogue_mode = reward([known], novelty_weight=0.0, model_dir=model_dir)[0]
    assert analogue_mode > balanced


def test_weights_are_exponents(model_dir):
    full = reward([ANALOGUE], model_dir=model_dir)[0]
    no_guardrails = reward([ANALOGUE], ad_weight=0.0, novelty_weight=0.0,
                           quality_weight=0.0, model_dir=model_dir)[0]
    parts = components([ANALOGUE], model_dir)
    np.testing.assert_allclose(no_guardrails, parts["prob_active"][0])
    assert full < no_guardrails


def test_components_are_in_range(model_dir, smiles):
    parts = components(smiles, model_dir)
    for key in ("prob_active", "nn_tanimoto", "qed", "valid"):
        assert np.all((parts[key] >= 0) & (parts[key] <= 1))
    assert np.all((parts["sa"] >= 1) & (parts["sa"] <= 10))


def test_return_components_matches_scalar_call(model_dir, smiles):
    parts = reward(smiles, model_dir=model_dir, return_components=True)
    np.testing.assert_allclose(parts["reward"],
                               reward(smiles, model_dir=model_dir))
