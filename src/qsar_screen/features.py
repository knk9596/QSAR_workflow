"""Molecular featurisation: fingerprints, descriptors, scaffolds, similarity."""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ECFP_RADIUS = 2
ECFP_BITS = 2048

DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]
_DESCRIPTOR_FN = dict(Descriptors.descList)


def _mols(smiles):
    return [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in smiles]


def ecfp4(smiles, radius: int = ECFP_RADIUS, n_bits: int = ECFP_BITS,
          use_features: bool = False) -> np.ndarray:
    """Morgan fingerprints as an ``(n, n_bits)`` int8 matrix."""
    out = np.zeros((len(smiles), n_bits), dtype=np.int8)
    for i, mol in enumerate(_mols(smiles)):
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits, useFeatures=use_features)
        DataStructs.ConvertToNumpyArray(fp, out[i])
    return out


def maccs(smiles) -> np.ndarray:
    """MACCS structural keys as an ``(n, 167)`` int8 matrix."""
    out = np.zeros((len(smiles), 167), dtype=np.int8)
    for i, mol in enumerate(_mols(smiles)):
        if mol is None:
            continue
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), out[i])
    return out


def rdkit_descriptors(smiles, names=None, skip_invalid: bool = False):
    """Full RDKit descriptor block, no pre-selection.

    Returns ``(X, kept_smiles)``. Non-finite values become 0.0. With
    ``skip_invalid`` unparseable SMILES are dropped; otherwise they give zero rows.
    """
    names = DESCRIPTOR_NAMES if names is None else list(names)
    rows, kept = [], []
    for smi, mol in zip(smiles, _mols(smiles)):
        if mol is None:
            if skip_invalid:
                continue
            rows.append([0.0] * len(names))
            kept.append(smi)
            continue
        values = []
        for name in names:
            try:
                v = _DESCRIPTOR_FN[name](mol)
            except Exception:
                v = 0.0
            values.append(0.0 if v is None or not np.isfinite(v) else float(v))
        rows.append(values)
        kept.append(smi)
    X = np.asarray(rows, dtype=float).reshape(len(rows), len(names))
    return X, kept


def murcko_scaffolds(smiles) -> np.ndarray:
    """Bemis-Murcko scaffold SMILES, used as the cross-validation grouping."""
    out = []
    for smi, mol in zip(smiles, _mols(smiles)):
        out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else smi)
    return np.asarray(out, dtype=object)


def nn_tanimoto(smiles, reference_fps) -> np.ndarray:
    """Max ECFP4 Tanimoto of each molecule to a reference bit-vector matrix."""
    ref = [DataStructs.CreateFromBitString("".join(map(str, row)))
           for row in np.asarray(reference_fps, dtype=np.int8)]
    out = []
    for mol in _mols(smiles):
        if mol is None:
            out.append(np.nan)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, ECFP_RADIUS, nBits=ECFP_BITS)
        out.append(max(DataStructs.BulkTanimotoSimilarity(fp, ref)))
    return np.asarray(out, dtype=float)
