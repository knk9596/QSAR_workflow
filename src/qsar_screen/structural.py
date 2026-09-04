"""Protein-ligand interaction fingerprints from docked poses.

The strongest feature family in the benchmark, but it needs a 3D pose per
molecule, so it cannot run at library scale. Its role is rescoring a shortlist
that the ligand-based classifier has already reduced.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ProLIF's default interaction types, fixed here so a fingerprint built today
# has the same column semantics as one built for the training set.
INTERACTIONS = [
    "HBDonor", "HBAcceptor", "Hydrophobic", "PiStacking", "EdgeToFace",
    "FaceToFace", "PiCation", "CationPi", "Anionic", "Cationic",
    "XBDonor", "XBAcceptor", "MetalDonor", "MetalAcceptor", "VdWContact",
]

ID_COLUMN = "compound"


def load_fingerprints(csv_path: str, id_column: str = ID_COLUMN) -> pd.DataFrame:
    """Read an interaction-fingerprint table written by ``generate_prolif_fingerprints``.

    One row per compound, one binary column per ``RESIDUE.chain:interaction``.
    """
    frame = pd.read_csv(csv_path)
    if id_column not in frame.columns:
        raise ValueError(
            f"{csv_path} has no {id_column!r} column; found {list(frame.columns[:5])}")
    frame[id_column] = frame[id_column].astype(str)
    return frame


def align_to_dataset(fingerprints: pd.DataFrame, ids, id_column: str = ID_COLUMN,
                     aliases: dict[str, str] | None = None):
    """Return ``(X, mask)`` with fingerprint rows in the order of ``ids``.

    ``mask`` marks which requested ids were present. Poses routinely cover a
    subset of the assay set (docking fails, or the pose file predates a QC
    drop), so callers must subset their labels with the same mask rather than
    assume alignment.

    ``aliases`` maps a name as it appears in the fingerprint file to the dataset
    id. Pose files are generated once and then outlive later ID corrections, so
    without a map a renamed compound is dropped silently -- which shifts both
    the sample size and the metric.
    """
    ids = [str(i) for i in ids]
    aliases = aliases or {}
    lookup: dict[str, int] = {}
    for i, name in enumerate(fingerprints[id_column].astype(str)):
        lookup[aliases.get(name, name)] = i

    feature_cols = [c for c in fingerprints.columns if c != id_column]
    values = fingerprints[feature_cols].to_numpy(dtype=float)

    mask = np.array([i in lookup for i in ids], dtype=bool)
    rows = [lookup[i] for i in ids if i in lookup]
    return values[rows], mask


def interaction_summary(fingerprints: pd.DataFrame,
                        id_column: str = ID_COLUMN) -> pd.DataFrame:
    """Per-column occupancy, for spotting residues that never or always contact."""
    feature_cols = [c for c in fingerprints.columns if c != id_column]
    occupancy = fingerprints[feature_cols].to_numpy(dtype=float).mean(axis=0)
    parts = [c.rsplit(":", 1) for c in feature_cols]
    return pd.DataFrame({
        "column": feature_cols,
        "residue": [p[0] for p in parts],
        "interaction": [p[1] if len(p) > 1 else "" for p in parts],
        "occupancy": occupancy,
    }).sort_values("occupancy", ascending=False).reset_index(drop=True)


def generate_prolif_fingerprints(protein_pdb: str, poses_sdf: str, out_csv: str,
                                 interactions=None, n_jobs: int = 1) -> pd.DataFrame:
    """Compute interaction fingerprints for an SDF of docked poses.

    Requires ``prolif`` and ``MDAnalysis``. One row per pose in the SDF; pass a
    best-pose-per-compound file, because averaging over all poses measurably
    dilutes the signal (pose-ensemble variants ranked last in the benchmark).
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(n_jobs))

    import prolif as plf
    from rdkit import Chem

    interactions = list(interactions or INTERACTIONS)

    protein = plf.Molecule(Chem.MolFromPDBFile(protein_pdb, removeHs=False))
    supplier = Chem.SDMolSupplier(poses_sdf, removeHs=False)
    poses, names = [], []
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        poses.append(plf.Molecule.from_rdkit(mol))
        names.append(mol.GetProp("_Name").strip() if mol.HasProp("_Name")
                     else f"compound_{i}")

    fingerprint = plf.Fingerprint(interactions=interactions, count=False)
    fingerprint.run_from_iterable(poses, protein, progress=False)
    table = fingerprint.to_dataframe()
    table.columns = [f"{res}:{inter}" for _, res, inter in table.columns]
    table.insert(0, ID_COLUMN, names)
    table.to_csv(out_csv, index=False)
    return table
