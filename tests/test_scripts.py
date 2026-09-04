"""Tests for the standalone scripts in scripts/."""

import pandas as pd


def test_example_generator_terminates_on_over_request(tmp_path):
    """Requesting more molecules than the enumeration yields must not loop."""
    import runpy
    import sys
    out = tmp_path / "ex.csv"
    argv = sys.argv
    sys.argv = ["make_example_data.py", "--n", "10000", "--out", str(out)]
    try:
        runpy.run_path(
            str(__import__("pathlib").Path(__file__).parent.parent
                / "scripts" / "make_example_data.py"),
            run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0
    finally:
        sys.argv = argv
    frame = pd.read_csv(out)
    assert 0 < len(frame) < 10000
    assert frame.smiles.is_unique
    assert frame.activity_remaining.between(5, 100).all()
