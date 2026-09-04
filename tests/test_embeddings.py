import os

import pytest

from qsar_screen.embeddings import (
    ATOM_FEATURIZER_VERSION, CHECKPOINT_ENV, EMBEDDING_DIM, default_checkpoint,
    load_encoder,
)

torch = pytest.importorskip("torch", reason="embeddings need torch/chemprop")


def test_checkpoint_path_respects_the_env_override(monkeypatch):
    monkeypatch.setenv(CHECKPOINT_ENV, "/custom/ckpt.pt")
    assert default_checkpoint() == "/custom/ckpt.pt"
    monkeypatch.delenv(CHECKPOINT_ENV)
    assert default_checkpoint().endswith(os.path.join("models", "chemeleon_mp.pt"))


def test_missing_checkpoint_names_the_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="Zenodo"):
        load_encoder(str(tmp_path / "absent.pt"))


def test_pretraining_featurizer_version_is_pinned():
    """v1 silently yields a different graph encoding and a useless embedding."""
    assert ATOM_FEATURIZER_VERSION == "v2"


@pytest.mark.skipif(not os.path.exists(default_checkpoint()),
                    reason="checkpoint not present")
def test_embeddings_have_the_expected_shape():
    from qsar_screen.embeddings import chemeleon_embeddings
    X = chemeleon_embeddings(["CCO", "c1ccccc1"])
    assert X.shape == (2, EMBEDDING_DIM)
    assert (X != 0).any()


@pytest.mark.skipif(not os.path.exists(default_checkpoint()),
                    reason="checkpoint not present")
def test_invalid_smiles_leaves_a_zero_row():
    from qsar_screen.embeddings import chemeleon_embeddings
    X = chemeleon_embeddings(["CCO", "not_a_molecule"])
    assert (X[1] == 0).all() and (X[0] != 0).any()
