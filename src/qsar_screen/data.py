"""Load a raw assay table into the modelling set.

The raw plate export is not the modelling set. Three things have to happen
first, and none of them is inferable from the file itself:

1. Readings above the DMSO control ceiling are assay artefacts, not
   super-inactive compounds, and must be dropped rather than clipped.
2. SMILES must be canonicalised before anything is joined or de-duplicated
   on structure.
Fix identifier problems in the source CSV before calling this - the loader
deliberately does not rewrite ids, so what you read is what you passed in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

ACTIVITY_CEILING = 100.0   # vehicle control = 100% activity remaining

# columns we will look for, in order of preference
SMILES_CANDIDATES = ("smiles_canon", "smiles", "SMILES", "canonical_smiles")
ACTIVITY_CANDIDATES = ("activity_remaining", "avg_inhibition", "activity",
                       "pct_activity_remaining")
ID_CANDIDATES = ("ID", "id", "compound", "compound_id", "name")


def _pick(frame: pd.DataFrame, candidates, kind: str, explicit=None) -> str:
    if explicit:
        if explicit not in frame.columns:
            raise ValueError(f"no {kind} column {explicit!r}; "
                             f"file has {list(frame.columns)}")
        return explicit
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(
        f"could not find a {kind} column. Looked for {list(candidates)}; "
        f"file has {list(frame.columns)}. Pass it explicitly.")


def canonical_smiles(smiles):
    """Canonicalise, returning None for anything RDKit cannot parse."""
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        out.append(Chem.MolToSmiles(mol) if mol is not None else None)
    return out


def load_assay(path: str, smiles_col: str | None = None,
               activity_col: str | None = None, id_col: str | None = None,
               ceiling: float = ACTIVITY_CEILING,
               verbose: bool = True) -> pd.DataFrame:
    """Read a raw assay CSV and return the QC'd modelling set.

    Returns a frame with `id`, `smiles` (canonical) and `activity_remaining`.
    Rows are dropped when the SMILES will not parse or the reading exceeds
    `ceiling`; pass `ceiling=None` to keep everything.
    """
    raw = pd.read_csv(path)
    s_col = _pick(raw, SMILES_CANDIDATES, "SMILES", smiles_col)
    a_col = _pick(raw, ACTIVITY_CANDIDATES, "activity", activity_col)
    try:
        i_col = _pick(raw, ID_CANDIDATES, "id", id_col)
    except ValueError:
        i_col = None

    frame = pd.DataFrame({
        "id": (raw[i_col].astype(str) if i_col
               else [f"row{i}" for i in range(len(raw))]),
        "smiles": canonical_smiles(raw[s_col]),
        "activity_remaining": pd.to_numeric(raw[a_col], errors="coerce"),
    })
    for extra in ("notes", "series", "batch"):
        if extra in raw.columns:
            frame[extra] = raw[extra].values

    n_raw = len(frame)
    unparsed = frame.smiles.isna().sum()
    frame = frame[frame.smiles.notna()]
    missing = frame.activity_remaining.isna().sum()
    frame = frame[frame.activity_remaining.notna()]

    n_artefact = 0
    if ceiling is not None:
        artefacts = frame.activity_remaining > ceiling
        n_artefact = int(artefacts.sum())
        frame = frame[~artefacts]

    frame = frame.reset_index(drop=True)
    if verbose:
        print(f"{path}: {n_raw} rows -> {len(frame)} after QC")
        print(f"  columns used: id={i_col}, smiles={s_col}, activity={a_col}")
        if unparsed:
            print(f"  dropped {unparsed} unparseable SMILES")
        if missing:
            print(f"  dropped {missing} rows with no activity value")
        if n_artefact:
            print(f"  dropped {n_artefact} readings above the {ceiling:g}% "
                  f"ceiling (assay artefacts, not inactives)")
        if frame.smiles.duplicated().any():
            print(f"  NOTE {int(frame.smiles.duplicated().sum())} duplicate "
                  f"structures remain - de-duplicate before scaffold splitting")
    return frame
