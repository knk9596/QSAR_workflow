#!/usr/bin/env python3
"""Fit the hit classifier on an assay table and write it to models/.

    python scripts/train.py --data data/assay.csv --smiles-col smiles \
        --activity-col activity_remaining
"""
from __future__ import annotations

import argparse

import pandas as pd

from qsar_screen.model import ACTIVITY_THRESHOLD, save_scorer, train


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--activity-col", default="activity_remaining")
    ap.add_argument("--threshold", type=float, default=ACTIVITY_THRESHOLD)
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args()

    frame = pd.read_csv(args.data)
    scorer = train(frame[args.smiles_col].tolist(),
                   frame[args.activity_col].to_numpy(),
                   threshold=args.threshold)
    path = save_scorer(scorer) if args.model_dir is None else save_scorer(
        scorer, args.model_dir)

    print(f"trained on {scorer.metrics['n_train']} molecules "
          f"({scorer.metrics['n_active']} active, "
          f"baseline PR-AUC {scorer.metrics['baseline_pr_auc']})")
    print(f"features reaching the classifier: {scorer.n_features}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
