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


def cmd_screen(argv=None) -> int:
    """Stream a large library through the classifier and keep the top hits."""
    from qsar_screen.model import AD_EDGE
    from qsar_screen.screen import screen

    ap = argparse.ArgumentParser(
        prog="qsar-screen", description=cmd_screen.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", required=True, nargs="+",
                    help="files, directories, or globs (quote globs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--ad-prefilter", action="store_true",
                    help="skip molecules below --ad-min before computing "
                         "descriptors; much faster, but suppresses novel scaffolds")
    ap.add_argument("--ad-min", type=float, default=AD_EDGE)
    ap.add_argument("--limit", type=int, default=0, help="stop after N molecules")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    args = ap.parse_args(argv)

    screen(args.input, args.out, top_k=args.top_k, workers=args.workers,
           chunk_size=args.chunk_size, smiles_col=args.smiles_col,
           id_col=args.id_col, ad_prefilter=args.ad_prefilter, ad_min=args.ad_min,
           limit=args.limit, dedup=not args.no_dedup, model_dir=args.model_dir)
    return 0


def cmd_reward(argv=None) -> int:
    """Score generated SMILES with the guardrailed generative reward."""
    from qsar_screen.reward import reward

    ap = argparse.ArgumentParser(prog="qsar-reward", description=cmd_reward.__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ad-weight", type=float, default=1.0)
    ap.add_argument("--novelty-weight", type=float, default=1.0)
    ap.add_argument("--quality-weight", type=float, default=1.0)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    args = ap.parse_args(argv)

    smiles = _read_smiles(args.input)
    parts = reward(smiles, args.ad_weight, args.novelty_weight,
                   args.quality_weight, args.model_dir, return_components=True)
    frame = pd.DataFrame({"smiles": smiles, **{k: parts[k] for k in
                          ("reward", "prob_active", "nn_tanimoto", "ad",
                           "novelty", "quality", "qed", "sa", "valid")}})
    frame.to_csv(args.out, index=False)
    print(f"scored {len(frame)} SMILES ({int(frame.valid.sum())} valid) -> {args.out}")
    print(f"reward mean {frame.reward.mean():.4f}  max {frame.reward.max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(cmd_score())
