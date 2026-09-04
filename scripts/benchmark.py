#!/usr/bin/env python3
"""Scaffold-split benchmark across feature families and classifiers.

    python scripts/benchmark.py --data data/assay.csv --out benchmark.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from qsar_screen.evaluate import (
    classification_metrics, enrichment_factor, permutation_test, scaffold_cv,
)
from qsar_screen.features import ecfp4, maccs, murcko_scaffolds, rdkit_descriptors
from qsar_screen.model import ACTIVITY_THRESHOLD

BINARY = {"ECFP4": True, "FCFP4": True, "MACCS": True, "RDKit-Desc": False}


def feature_blocks(smiles):
    X_desc, _ = rdkit_descriptors(smiles)
    return {"ECFP4": ecfp4(smiles),
            "FCFP4": ecfp4(smiles, use_features=True),
            "MACCS": maccs(smiles),
            "RDKit-Desc": X_desc}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--activity-col", default="activity_remaining")
    ap.add_argument("--threshold", type=float, default=ACTIVITY_THRESHOLD)
    ap.add_argument("--models", nargs="+", default=["rf", "svm"])
    ap.add_argument("--permutations", type=int, default=0,
                    help="y-scrambling permutations for the best model (0 = skip)")
    ap.add_argument("--out", default="benchmark.csv")
    args = ap.parse_args()

    frame = pd.read_csv(args.data)
    smiles = frame[args.smiles_col].tolist()
    y = (frame[args.activity_col].to_numpy(float) < args.threshold).astype(int)
    groups = murcko_scaffolds(smiles)
    blocks = feature_blocks(smiles)

    print(f"n={len(y)}  active={int(y.sum())}  baseline={y.mean():.3f}  "
          f"scaffolds={len(set(groups))}\n")

    rows = []
    for name, X in blocks.items():
        for kind in args.models:
            oof, width = scaffold_cv(X, y, groups, kind, BINARY[name])
            ef10, hits10, _ = enrichment_factor(y, oof, 0.10)
            row = {"feature": name, "model": kind, "n_features": int(width),
                   **classification_metrics(y, oof),
                   "ef_top10": round(ef10, 3), "hits_top10": hits10}
            rows.append(row)
            print(f"  {name:12s} {kind:4s} PR-AUC {row['pr_auc']:.3f}  "
                  f"ROC {row['roc_auc']:.3f}  EF@10% {row['ef_top10']:.2f}")

    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    table.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    if args.permutations:
        best = table.iloc[0]
        print(f"\ny-scrambling {best.feature}/{best.model}, "
              f"{args.permutations} permutations")
        result = permutation_test(blocks[best.feature], y, groups, best.model,
                                  BINARY[best.feature], args.permutations)
        for key, value in result.items():
            print(f"  {key:18s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
