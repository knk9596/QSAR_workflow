import numpy as np
import pandas as pd
import pytest

from qsar_screen.structural import (
    align_to_dataset, interaction_summary, load_fingerprints,
)


@pytest.fixture
def fingerprint_csv(tmp_path):
    frame = pd.DataFrame({
        "compound": ["a1", "a2", "a3", "old_name"],
        "PHE100.A:Hydrophobic": [1, 1, 0, 1],
        "ASP200.A:HBDonor": [0, 1, 0, 1],
        "TYR300.A:PiStacking": [1, 0, 0, 0],
    })
    path = tmp_path / "fp.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_load_requires_the_id_column(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"name": ["x"], "PHE1.A:Hydrophobic": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="no 'compound' column"):
        load_fingerprints(str(bad))


def test_alignment_follows_the_requested_order(fingerprint_csv):
    table = load_fingerprints(fingerprint_csv)
    X, mask = align_to_dataset(table, ["a3", "a1"])
    assert mask.all() and X.shape == (2, 3)
    np.testing.assert_array_equal(X[0], [0, 0, 0])   # a3
    np.testing.assert_array_equal(X[1], [1, 0, 1])   # a1


def test_missing_compounds_are_masked_not_dropped_silently(fingerprint_csv):
    table = load_fingerprints(fingerprint_csv)
    X, mask = align_to_dataset(table, ["a1", "absent", "a2"])
    assert mask.tolist() == [True, False, True]
    assert X.shape == (2, 3)          # only matched rows are returned
    assert mask.sum() == X.shape[0]   # caller can subset labels with mask


def test_aliases_recover_a_renamed_compound(fingerprint_csv):
    """Pose files outlive id corrections; without the alias the row is lost."""
    table = load_fingerprints(fingerprint_csv)
    _, without = align_to_dataset(table, ["new_name"])
    assert not without.any()
    X, with_alias = align_to_dataset(table, ["new_name"],
                                     aliases={"old_name": "new_name"})
    assert with_alias.all()
    np.testing.assert_array_equal(X[0], [1, 1, 0])


def test_no_overlap_yields_an_all_false_mask(fingerprint_csv):
    table = load_fingerprints(fingerprint_csv)
    X, mask = align_to_dataset(table, ["zzz", "yyy"])
    assert not mask.any() and X.shape[0] == 0


def test_interaction_summary_splits_residue_and_type(fingerprint_csv):
    summary = interaction_summary(load_fingerprints(fingerprint_csv))
    assert len(summary) == 3
    assert summary.iloc[0]["column"] == "PHE100.A:Hydrophobic"   # highest occupancy
    assert summary.iloc[0]["residue"] == "PHE100.A"
    assert summary.iloc[0]["interaction"] == "Hydrophobic"
    np.testing.assert_allclose(summary.iloc[0]["occupancy"], 0.75)
    assert summary.occupancy.is_monotonic_decreasing
