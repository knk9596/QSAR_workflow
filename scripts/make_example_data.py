#!/usr/bin/env python3
"""Generate a synthetic assay table so the pipeline runs without real data.

Enumerates core-substituent combinations exhaustively, so the requested size is
capped by the combinatorics rather than sampled until it happens to be reached.
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

CORES = [
    "O=C(NCCCn1cnc2ccccc21)c1cc(-c2ccc{r}cc2)[nH]n1",
    "O=C(NCCCc1nnc2ccccn12)c1cc(-c2ccc{r}cc2)[nH]n1",
    "O=c1ccnc2nc(NCc3ccc{r}cc3)[nH]n12",
    "c1ccc(CNc2nc3ccc{r}cc3o2)cc1",
    "O=C(Nc1nc2nccc(=O)n2[nH]1)c1ccc{r}cc1",
    "O=C(NCCn1cnc2ccccc21)c1cc(-c2ccc{r}cc2)[nH]n1",
]
SUBSTITUENTS = [
    "(C)", "(CC)", "(F)", "(Cl)", "(Br)", "(O)", "(OC)", "(OCC)", "(N)",
    "(C(F)(F)F)", "(S)", "(SC)", "(CO)", "(C#N)", "(C(C)C)", "(N(C)C)",
    "(CCO)", "(OCCO)", "", "(CN)",
]


def enumerate_molecules():
    """Yield unique canonical SMILES over the full core x substituent grid."""
    seen = set()
    for core, sub in itertools.product(CORES, SUBSTITUENTS):
        mol = Chem.MolFromSmiles(core.format(r=sub))
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical not in seen:
            seen.add(canonical)
            yield canonical, mol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=120,
                    help="molecules to write; capped by the enumeration size")
    ap.add_argument("--out", default="data/example/assay_example.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    records = []
    for canonical, mol in enumerate_molecules():
        if len(records) >= args.n:
            break
        # activity loosely tracks heavy-atom count so a model has real signal
        # intercept/slope calibrated to give ~30% actives, matching the real set
        activity = 112.0 - 1.7 * mol.GetNumHeavyAtoms() + rng.normal(0, 11)
        records.append({"id": f"EX{len(records):04d}", "smiles": canonical,
                        "activity_remaining": round(float(np.clip(activity, 5, 100)), 2)})

    if not records:
        raise RuntimeError("enumeration produced no valid molecules")
    frame = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frame.to_csv(args.out, index=False)

    n_active = int((frame.activity_remaining < 65).sum())
    if len(frame) < args.n:
        print(f"note: enumeration yields {len(frame)} unique molecules "
              f"(requested {args.n})")
    print(f"wrote {args.out}: {len(frame)} molecules, {n_active} active "
          f"({n_active / len(frame):.1%}), "
          f"{frame.smiles.nunique()} unique structures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
