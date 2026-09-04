"""Command-line entry points."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from qsar_screen.model import DEFAULT_MODEL_DIR, score_smiles


def _read_smiles(path: str, column: str = "smiles") -> list[str]:
    if path.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path)[column].astype(str).tolist()
    with open(path) as handle:
        return [line.split()[0] for line in handle if line.strip()]


def cmd_score(argv=None) -> int:
    """Score a small SMILES list; for large libraries use ``qsar-screen``."""
    ap = argparse.ArgumentParser(prog="qsar-score", description=cmd_score.__doc__)
    ap.add_argument("--input", required=True, help="CSV or one-SMILES-per-line file")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--out", default=None, help="output CSV (default: stdout)")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    args = ap.parse_args(argv)

    frame = score_smiles(_read_smiles(args.input, args.smiles_col),
                         model_dir=args.model_dir)
    if args.out:
        frame.to_csv(args.out, index=False)
        print(f"scored {len(frame)} molecules -> {args.out}")
    else:
        print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(cmd_score())
