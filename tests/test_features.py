import numpy as np

from qsar_screen import ecfp4, maccs, murcko_scaffolds, nn_tanimoto, rdkit_descriptors
from qsar_screen.features import DESCRIPTOR_NAMES


def test_fingerprint_shapes_and_binarity(smiles):
    fp = ecfp4(smiles)
    assert fp.shape == (len(smiles), 2048)
    assert set(np.unique(fp)).issubset({0, 1})
    assert maccs(smiles).shape == (len(smiles), 167)


def test_feature_fingerprints_differ_from_ecfp(smiles):
    assert not np.array_equal(ecfp4(smiles), ecfp4(smiles, use_features=True))


def test_descriptors_are_finite_and_ordered(smiles):
    X, kept = rdkit_descriptors(smiles)
    assert X.shape == (len(smiles), len(DESCRIPTOR_NAMES))
    assert np.isfinite(X).all()
    assert kept == smiles
    subset = ["MolWt", "MolLogP"]
    X2, _ = rdkit_descriptors(smiles, subset)
    np.testing.assert_allclose(X2[:, 0], X[:, DESCRIPTOR_NAMES.index("MolWt")])


def test_invalid_smiles_handling():
    mixed = ["CCO", "not_a_molecule", "c1ccccc1"]
    X_zero, kept_zero = rdkit_descriptors(mixed)
    assert len(kept_zero) == 3
    assert np.all(X_zero[1] == 0.0)
    X_skip, kept_skip = rdkit_descriptors(mixed, skip_invalid=True)
    assert len(kept_skip) == 2 and X_skip.shape[0] == 2


def test_scaffold_grouping_collapses_analogues():
    analogues = ["O=c1ccnc2nc(NCc3ccccc3)[nH]n12",
                 "O=c1ccnc2nc(NCc3ccc(F)cc3)[nH]n12"]
    scaffolds = murcko_scaffolds(analogues)
    assert len(scaffolds) == 2
    assert murcko_scaffolds(["CCO"])[0] == ""   # acyclic has no Murcko scaffold


def test_self_similarity_is_one(smiles):
    ref = ecfp4(smiles)
    nn = nn_tanimoto(smiles, ref)
    np.testing.assert_allclose(nn, 1.0)


def test_unrelated_molecule_is_dissimilar(smiles):
    nn = nn_tanimoto(["CCCCCCCCCCCCCCCCCC"], ecfp4(smiles[:4]))
    assert nn[0] < 0.3


def test_similarity_of_invalid_is_nan(smiles):
    assert np.isnan(nn_tanimoto(["!!!"], ecfp4(smiles))[0])
