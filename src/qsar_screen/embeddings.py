"""Learned molecular embeddings from a pretrained message-passing encoder.

Wraps the CheMeleon foundation-model checkpoint (Burns et al.). Only the frozen
encoder is used: a forward pass gives a 2048-dim embedding per molecule in
seconds on CPU. Fine-tuning the whole network is the GPU path and is not
appropriate at n = 163 -- a frozen encoder with a shallow head is.
"""

from __future__ import annotations

import os

import numpy as np

CHECKPOINT_ENV = "CHEMELEON_CHECKPOINT"
EMBEDDING_DIM = 2048
_ENCODER: dict = {}

# CheMeleon was pretrained with the v2 atom featuriser; v1 silently produces a
# different graph encoding and a useless embedding.
ATOM_FEATURIZER_VERSION = "v2"


def default_checkpoint() -> str:
    """Checkpoint path from ``CHEMELEON_CHECKPOINT``, else ``models/chemeleon_mp.pt``."""
    if os.environ.get(CHECKPOINT_ENV):
        return os.environ[CHECKPOINT_ENV]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "models", "chemeleon_mp.pt")


def load_encoder(checkpoint: str | None = None):
    """Load and cache the frozen message-passing encoder."""
    path = checkpoint or default_checkpoint()
    if path in _ENCODER:
        return _ENCODER[path]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no CheMeleon checkpoint at {path}. Download chemeleon_mp.pt "
            f"(Zenodo record 15460715) or set ${CHECKPOINT_ENV}.")

    import torch
    from chemprop.nn.message_passing import BondMessagePassing

    blob = torch.load(path, map_location="cpu", weights_only=False)
    keys = ("d_v", "d_e", "d_h", "bias", "depth", "dropout", "activation",
            "undirected", "d_vd")
    params = {k: v for k, v in blob["hyper_parameters"].items() if k in keys}
    encoder = BondMessagePassing(**params)
    encoder.load_state_dict(blob["state_dict"])
    encoder.eval()
    _ENCODER[path] = encoder
    return encoder


def chemeleon_embeddings(smiles, checkpoint: str | None = None,
                         batch_size: int = 64, n_threads: int = 1) -> np.ndarray:
    """Embed SMILES as an ``(n, 2048)`` matrix; unparseable rows are left zero."""
    import torch
    from chemprop import featurizers
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch
    from chemprop.nn.agg import MeanAggregation
    from torch.utils.data import DataLoader

    torch.set_num_threads(n_threads)
    encoder = load_encoder(checkpoint)
    smiles = list(smiles)

    from rdkit import Chem
    valid = [i for i, s in enumerate(smiles)
             if isinstance(s, str) and Chem.MolFromSmiles(s) is not None]
    out = np.zeros((len(smiles), EMBEDDING_DIM), dtype=np.float32)
    if not valid:
        return out

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    dataset = MoleculeDataset(
        [MoleculeDatapoint.from_smi(smiles[i]) for i in valid],
        featurizer=featurizer)
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=collate_batch, shuffle=False)

    aggregate = MeanAggregation()
    chunks = []
    with torch.no_grad():
        for batch in loader:
            chunks.append(aggregate(encoder(batch.bmg), batch.bmg.batch).cpu().numpy())
    out[valid] = np.vstack(chunks)
    return out
