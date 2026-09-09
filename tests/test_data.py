import numpy as np
import pandas as pd
import pytest

from qsar_screen.data import ACTIVITY_CEILING, canonical_smiles, load_assay


@pytest.fixture
def raw_csv(tmp_path):
    """Mimics the raw plate export: legacy column name, an artefact, a bad SMILES."""
    frame = pd.DataFrame({
        "ID": ["a1", "a2", "a3", "a4", "bad"],
        "smiles": ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "not_a_molecule"],
        "avg_inhibition": [20.0, 90.0, 105.3, 50.0, 40.0],
        "notes": ["s1", "s1", "s2", "s2", "s1"],
    })
    path = tmp_path / "raw.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_legacy_activity_column_is_found(raw_csv):
    """The raw export calls it avg_inhibition, not activity_remaining."""
    frame = load_assay(raw_csv, verbose=False)
    assert "activity_remaining" in frame.columns
    assert set(frame.columns) >= {"id", "smiles", "activity_remaining"}


def test_above_ceiling_rows_are_dropped_not_clipped(raw_csv):
    frame = load_assay(raw_csv, verbose=False)
    assert "a3" not in set(frame.id)                      # 105.3 removed
    assert frame.activity_remaining.max() <= ACTIVITY_CEILING
    kept = load_assay(raw_csv, ceiling=None, verbose=False)
    assert "a3" in set(kept.id)                           # opt out works


def test_unparseable_smiles_are_dropped(raw_csv):
    frame = load_assay(raw_csv, verbose=False)
    assert "bad" not in set(frame.id)
    assert frame.smiles.notna().all()


def test_smiles_are_canonicalised(raw_csv):
    frame = load_assay(raw_csv, verbose=False)
    assert list(frame.smiles) == canonical_smiles(list(frame.smiles))


def test_missing_column_names_what_it_looked_for(tmp_path):
    path = tmp_path / "nocol.csv"
    pd.DataFrame({"structure": ["CCO"], "readout": [10.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="could not find a SMILES column"):
        load_assay(str(path), verbose=False)


def test_explicit_column_that_does_not_exist_is_reported(raw_csv):
    with pytest.raises(ValueError, match="no activity column 'nope'"):
        load_assay(raw_csv, activity_col="nope", verbose=False)


def test_canonical_smiles_returns_none_for_junk():
    out = canonical_smiles(["CCO", "not_a_molecule", None, 42])
    assert out[0] == "CCO" and out[1] is None and out[2] is None and out[3] is None
