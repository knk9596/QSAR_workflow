import gzip

import numpy as np
import pandas as pd
import pytest

from qsar_screen.model import save_scorer, score_smiles, train
from qsar_screen.screen import expand_inputs, screen, sniff_format, stream_library


@pytest.fixture
def model_dir(tmp_path, smiles, activity):
    path = tmp_path / "model"
    save_scorer(train(smiles, activity), str(path))
    return str(path)


@pytest.fixture
def library(tmp_path, smiles):
    root = tmp_path / "lib"
    root.mkdir()
    pd.DataFrame({"zinc_id": [f"Z{i}" for i in range(len(smiles))],
                  "smiles": smiles}).to_csv(root / "a.csv", index=False)
    (root / "b.smi").write_text(
        "".join(f"{s}\tS{i}\n" for i, s in enumerate(smiles)))
    with gzip.open(root / "c.smi.gz", "wt") as handle:
        handle.write("".join(f"{s} G{i}\n" for i, s in enumerate(smiles[:4])))
        handle.write("this_is_not_a_molecule BAD1\n")
    return root


def test_sniffs_headed_csv_and_headerless_smi(library):
    delim, s_idx, i_idx, header = sniff_format(str(library / "a.csv"))
    assert delim == "," and header and s_idx == 1 and i_idx == 0
    delim, s_idx, i_idx, header = sniff_format(str(library / "b.smi"))
    assert delim == "\t" and not header and s_idx == 0 and i_idx == 1


def test_expand_inputs_handles_dirs_and_globs(library):
    assert len(expand_inputs([str(library)])) == 3
    assert len(expand_inputs([str(library / "*.csv")])) == 1


def test_stream_yields_every_row_with_ids(library, smiles):
    chunks = list(stream_library([str(library / "b.smi")], chunk_size=3))
    rows = [row for chunk in chunks for row in chunk]
    assert len(rows) == len(smiles)
    assert rows[0] == (smiles[0], "S0")
    assert max(len(c) for c in chunks) <= 3


def test_gzip_is_read_transparently(library):
    rows = [r for c in stream_library([str(library / "c.smi.gz")]) for r in c]
    assert len(rows) == 5 and rows[-1][1] == "BAD1"


def test_screen_matches_the_reference_scorer(library, model_dir, tmp_path):
    out = tmp_path / "hits.csv"
    hits = screen([str(library / "a.csv")], str(out), top_k=100, workers=1,
                  chunk_size=4, model_dir=model_dir, progress=False)
    assert out.exists()
    reference = score_smiles(hits.smiles.tolist(), model_dir=model_dir)
    merged = hits.merge(reference, on="smiles", suffixes=("", "_ref"))
    np.testing.assert_allclose(merged.prob_active, merged.prob_active_ref, atol=1e-12)
    np.testing.assert_allclose(merged.nn_tanimoto, merged.nn_tanimoto_ref, atol=1e-12)


def test_top_k_is_respected_and_sorted(library, model_dir, tmp_path):
    hits = screen([str(library)], str(tmp_path / "h.csv"), top_k=3, workers=1,
                  chunk_size=5, model_dir=model_dir, progress=False)
    assert len(hits) <= 3
    assert hits.prob_active.is_monotonic_decreasing


def test_duplicates_across_files_are_collapsed(library, model_dir, tmp_path):
    kept = screen([str(library)], str(tmp_path / "d.csv"), top_k=1000, workers=1,
                  model_dir=model_dir, progress=False)
    assert kept.smiles.is_unique
    raw = screen([str(library)], str(tmp_path / "r.csv"), top_k=1000, workers=1,
                 model_dir=model_dir, dedup=False, progress=False)
    assert len(raw) > len(kept)


def test_invalid_smiles_are_skipped_not_scored(library, model_dir, tmp_path):
    hits = screen([str(library / "c.smi.gz")], str(tmp_path / "i.csv"), top_k=100,
                  workers=1, model_dir=model_dir, progress=False)
    assert "BAD1" not in set(hits.id)
    assert len(hits) == 4


def test_ad_prefilter_drops_dissimilar_molecules(model_dir, tmp_path):
    # neither molecule is in the training fixture, so both are far from it
    path = tmp_path / "far.smi"
    path.write_text("CCCCCCCCCCCCCCCCCCCC\tFAR1\n[Na+].[Cl-]\tFAR2\n")
    kept = screen([str(path)], str(tmp_path / "n.csv"), top_k=10, workers=1,
                  model_dir=model_dir, ad_prefilter=True, ad_min=0.9, progress=False)
    assert len(kept) == 0
    passed = screen([str(path)], str(tmp_path / "p.csv"), top_k=10, workers=1,
                    model_dir=model_dir, ad_prefilter=False, progress=False)
    assert len(passed) == 2   # without the prefilter both are scored


def test_run_resumes_from_a_checkpoint(library, model_dir, tmp_path, capsys):
    out = str(tmp_path / "resume.csv")
    screen([str(library / "b.smi")], out, top_k=50, workers=1, chunk_size=2,
           checkpoint_every=1, limit=4, model_dir=model_dir, progress=False)
    screen([str(library / "b.smi")], out, top_k=50, workers=1, chunk_size=2,
           checkpoint_every=1, model_dir=model_dir, progress=True)
    assert "resuming" in capsys.readouterr().out


def test_missing_input_raises(model_dir, tmp_path):
    with pytest.raises(FileNotFoundError):
        screen([str(tmp_path / "nope*.smi")], str(tmp_path / "x.csv"),
               workers=1, model_dir=model_dir, progress=False)
