#!/usr/bin/env python3
"""Scaffold-split benchmark across feature families and classifiers.

    python scripts/benchmark.py --data data/assay.csv --out benchmark.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from qsar_screen.evaluate import (
    classification_metrics, enrichment_factor, permutation_test, scaffold_cv,
)
from qsar_screen.features import ecfp4, maccs, murcko_scaffolds, rdkit_descriptors
from qsar_screen.model import ACTIVITY_THRESHOLD

# whether a block is a bit vector (variance filter only) or dense (scale + prune)
BINARY = {"ECFP4": True, "FCFP4": True, "MACCS": True, "RDKit-Desc": False,
          "ProLIF": True, "CheMeleon": False}


def feature_blocks(smiles, ids=None, prolif_csv=None, chemeleon_ckpt=None,
                   chemeleon_npy=None, prolif_aliases=None):
    """Ligand-based blocks always; structural and learned blocks when available.

    Interaction fingerprints need docked poses and the embedding needs a
    checkpoint, so both are optional. A block that covers only a subset of the
    dataset also returns the row mask it applies to.
    """
    X_desc, _ = rdkit_descriptors(smiles)
    blocks = {"ECFP4": (ecfp4(smiles), None),
              "FCFP4": (ecfp4(smiles, use_features=True), None),
              "MACCS": (maccs(smiles), None),
              "RDKit-Desc": (X_desc, None)}

    if prolif_csv:
        from qsar_screen.structural import align_to_dataset, load_fingerprints
        table = load_fingerprints(prolif_csv)
        X_plf, mask = align_to_dataset(table, ids if ids is not None else smiles,
                                       aliases=prolif_aliases)
        if mask.sum() == 0:
            sample_ds = list(ids if ids is not None else smiles)[:3]
            sample_fp = table.iloc[:3, 0].tolist()
            raise SystemExit(
                "interaction fingerprints share no ids with the dataset.\n"
                f"  dataset ids look like: {sample_ds}\n"
                f"  fingerprint ids look like: {sample_fp}\n"
                "  pass --prolif-aliases old=new for renamed compounds.")
        if mask.sum() < len(mask):
            print(f"  NOTE {len(mask) - int(mask.sum())} compounds lack a pose and are "
                  f"excluded from the ProLIF row only")
        blocks["ProLIF"] = (X_plf, mask if not mask.all() else None)
        print(f"  interaction fingerprints: {X_plf.shape[1]} columns, "
              f"{int(mask.sum())}/{len(mask)} compounds matched")

    if chemeleon_npy:
        X_emb = np.load(chemeleon_npy)
        if len(X_emb) != len(smiles):
            raise ValueError(f"{chemeleon_npy} has {len(X_emb)} rows, "
                             f"dataset has {len(smiles)}")
        blocks["CheMeleon"] = (X_emb, None)
        print(f"  learned embedding: {X_emb.shape[1]} dims (precomputed)")
    elif chemeleon_ckpt:
        from qsar_screen.embeddings import chemeleon_embeddings
        blocks["CheMeleon"] = (chemeleon_embeddings(smiles, chemeleon_ckpt), None)
        print(f"  learned embedding: {blocks['CheMeleon'][0].shape[1]} dims")
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--activity-col", default="activity_remaining")
    ap.add_argument("--threshold", type=float, default=ACTIVITY_THRESHOLD)
    ap.add_argument("--models", nargs="+", default=["rf", "svm", "xgb"],
                    choices=["rf", "svm", "xgb"])
    ap.add_argument("--prolif-csv", default=None,
                    help="interaction-fingerprint CSV from docked poses")
    ap.add_argument("--chemeleon-ckpt", default=None,
                    help="CheMeleon checkpoint (needs torch/chemprop)")
    ap.add_argument("--chemeleon-npy", default=None,
                    help="precomputed embeddings from generate_embeddings.py")
    ap.add_argument("--prolif-aliases", default=None,
                    help="renamed ids as old=new pairs, e.g. cmpd29v=cmpd29")
    ap.add_argument("--permutations", type=int, default=0,
                    help="y-scrambling permutations for the best model (0 = skip)")
    ap.add_argument("--out", default="benchmark.csv")
    args = ap.parse_args()

    frame = pd.read_csv(args.data)
    smiles = frame[args.smiles_col].tolist()
    y = (frame[args.activity_col].to_numpy(float) < args.threshold).astype(int)
    id_col = next((c for c in frame.columns if c.lower() in ("id", "compound",
                   "compound_id", "name")), None)
    ids = frame[id_col] if id_col else None
    if args.prolif_csv and ids is None:
        raise SystemExit("--prolif-csv needs an id column in --data to join on "
                         f"(looked for id/compound/compound_id/name; "
                         f"found {list(frame.columns)})")
    groups = murcko_scaffolds(smiles)
    aliases = dict(pair.split("=", 1) for pair in
                   args.prolif_aliases.split(",")) if args.prolif_aliases else None
    blocks = feature_blocks(smiles, ids, args.prolif_csv, args.chemeleon_ckpt,
                            args.chemeleon_npy, aliases)

    print(f"n={len(y)}  active={int(y.sum())}  baseline={y.mean():.3f}  "
          f"scaffolds={len(set(groups))}\n")

    rows = []
    for name, (X, mask) in blocks.items():
        # a subset block is scored against its own labels and its own baseline
        y_block = y if mask is None else y[mask]
        groups_block = groups if mask is None else groups[mask]
        for kind in args.models:
            oof, width = scaffold_cv(X, y_block, groups_block, kind, BINARY[name])
            ef10, hits10, _ = enrichment_factor(y_block, oof, 0.10)
            row = {"feature": name, "model": kind, "n_features": int(width),
                   "n": len(y_block), "n_active": int(y_block.sum()),
                   **classification_metrics(y_block, oof),
                   "ef_top10": round(ef10, 3), "hits_top10": hits10}
            rows.append(row)
            print(f"  {name:12s} {kind:4s} PR-AUC {row['pr_auc']:.3f}  "
                  f"ROC {row['roc_auc']:.3f}  EF@10% {row['ef_top10']:.2f}"
                  + ("" if mask is None else f"  (n={len(y_block)})"))

    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    table.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    if args.permutations:
        best = table.iloc[0]
        print(f"\ny-scrambling {best.feature}/{best.model}, "
              f"{args.permutations} permutations")
        X_best, mask_best = blocks[best.feature]
        result = permutation_test(
            X_best, y if mask_best is None else y[mask_best],
            groups if mask_best is None else groups[mask_best],
            best.model, BINARY[best.feature], args.permutations)
        for key, value in result.items():
            print(f"  {key:18s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
