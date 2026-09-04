#!/usr/bin/env python3
"""Precompute frozen CheMeleon embeddings to a .npy file.

Separated from the benchmark so the benchmark does not need torch/chemprop:

    python scripts/generate_embeddings.py --data data/assay.csv \
        --ckpt models/chemeleon_mp.pt --out models/chemeleon_embeddings.npy
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from qsar_screen.embeddings import chemeleon_embeddings, default_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    frame = pd.read_csv(args.data)
    smiles = frame[args.smiles_col].astype(str).tolist()
    checkpoint = args.ckpt or default_checkpoint()

    start = time.time()
    X = chemeleon_embeddings(smiles, checkpoint, batch_size=args.batch_size)
    elapsed = time.time() - start

    np.save(args.out, X)
    n_valid = int((X != 0).any(axis=1).sum())
    print(f"embedded {n_valid}/{len(smiles)} molecules -> {X.shape} in "
          f"{elapsed:.1f}s ({len(smiles) / elapsed:.0f} mol/s)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
