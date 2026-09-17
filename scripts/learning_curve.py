#!/usr/bin/env python3
"""Train-vs-held-out learning curves: SEE the overfitting instead of assuming it.

Tracks PR-AUC on the training fold and the held-out fold after every few epochs,
for both the full fine-tune and the frozen-encoder FFN head, out to chemprop's
default of 50 epochs. The gap between the two curves IS the overfitting; where the
held-out curve peaks is the epoch budget the data supports (here: well under 50).

Needs the `[embeddings]` extra (torch, chemprop, lightning).

    python scripts/learning_curve.py --data data/example/assay_example.csv \\
        --ckpt models/chemeleon_mp.pt --epochs 50 --out results/learning_curve.csv
"""
from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from chemprop import data as cdata
from chemprop import featurizers
from chemprop import nn as cnn
from chemprop.models import MPNN

from qsar_screen.data import load_assay
from qsar_screen.embeddings import default_checkpoint
from qsar_screen.features import murcko_scaffolds

SEED = 42
EVERY = 2


class Track(pl.Callback):
    def __init__(self, train_dl, test_dl, y_tr, y_te, freeze):
        self.train_dl, self.test_dl = train_dl, test_dl
        self.y_tr, self.y_te, self.freeze = y_tr, y_te, freeze
        self.rows = []

    def on_train_epoch_end(self, trainer, module):
        ep = trainer.current_epoch + 1
        if ep % EVERY and ep != 1:
            return
        module.eval()
        with torch.no_grad():
            tr = torch.cat([module(b.bmg) for b in self.train_dl]).squeeze(-1).numpy()
            te = torch.cat([module(b.bmg) for b in self.test_dl]).squeeze(-1).numpy()
        module.train()
        assert len(tr) == len(self.y_tr) and len(te) == len(self.y_te)
        self.rows.append(dict(
            condition="full fine-tune" if not self.freeze else "frozen+FFN",
            epoch=ep, train_pr=average_precision_score(self.y_tr, tr),
            test_pr=average_precision_score(self.y_te, te)))


def _model(ckpt, freeze):
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    mp = cnn.BondMessagePassing(**blob["hyper_parameters"])
    mp.load_state_dict(blob["state_dict"])
    model = MPNN(message_passing=mp, agg=cnn.MeanAggregation(),
                 predictor=cnn.BinaryClassificationFFN(input_dim=mp.output_dim),
                 batch_norm=False, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4,
                 warmup_epochs=2)
    if freeze:
        for p in model.message_passing.parameters():
            p.requires_grad = False
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default="results/learning_curve.csv")
    args = ap.parse_args()
    ckpt = args.ckpt or default_checkpoint()

    frame = load_assay(args.data, verbose=False)
    smiles = list(frame.smiles)
    y = (frame.activity_remaining.to_numpy() < 65.0).astype(int)
    groups = np.asarray(murcko_scaffolds(smiles))
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"n={len(y)} active={y.sum()} epochs={args.epochs} every={EVERY}")

    rows = []
    for freeze in (False, True):
        for tr, te in cv.split(smiles, y, groups):
            mk = lambda idx, shuf, bs=32: cdata.build_dataloader(  # noqa: E731
                cdata.MoleculeDataset(
                    [cdata.MoleculeDatapoint.from_smi(smiles[i], np.array([float(y[i])]))
                     for i in idx], feat), batch_size=bs, shuffle=shuf)
            cb = Track(mk(tr, False, 512), mk(te, False, 512), y[tr], y[te], freeze)
            pl.Trainer(max_epochs=args.epochs, accelerator="auto", devices=1,
                       enable_checkpointing=False, enable_progress_bar=False,
                       enable_model_summary=False, logger=False,
                       callbacks=[cb]).fit(_model(ckpt, freeze), mk(tr, True))
            rows += cb.rows

    agg = (pd.DataFrame(rows).groupby(["condition", "epoch"])[["train_pr", "test_pr"]]
              .mean().reset_index())
    agg["gap"] = (agg.train_pr - agg.test_pr).round(4)
    agg[["train_pr", "test_pr"]] = agg[["train_pr", "test_pr"]].round(4)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    agg.to_csv(args.out, index=False)
    for c in agg.condition.unique():
        s = agg[agg.condition == c]
        best = s.loc[s.test_pr.idxmax()]
        print(f"{c}: held-out peak epoch {int(best.epoch)} (test {best.test_pr:.4f}, "
              f"train {best.train_pr:.4f}, gap {best.gap:.4f})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
