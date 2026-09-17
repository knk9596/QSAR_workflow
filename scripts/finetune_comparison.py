#!/usr/bin/env python3
"""Frozen encoder vs full fine-tuning of CheMeleon, on identical scaffold folds.

Three ways to put the CheMeleon (Chemprop 2 D-MPNN) checkpoint to work at n~163:

  A  full fine-tune       encoder + head trainable    (~9.3M params)
  B  frozen encoder + FFN  head only                  (~0.6M params)
  C  frozen encoder + RF   no gradient step           (0 params)

Everything else is held equal — same StratifiedGroupKFold on Murcko scaffold,
same labels, same embedding geometry, same metric. The result at this data scale
is that C (no training) wins; fine-tuning overfits (train PR-AUC -> 1.0 while
held-out stalls). This script demonstrates the fine-tuning path end-to-end AND
the controlled comparison that shows why it is the wrong choice here.

Needs the `[embeddings]` extra (torch, chemprop, lightning).

    python scripts/finetune_comparison.py --data data/example/assay_example.csv \\
        --ckpt models/chemeleon_mp.pt --epochs 50 --out results/finetune_comparison.csv
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from chemprop import data as cdata
from chemprop import featurizers
from chemprop import nn as cnn
from chemprop.models import MPNN

from qsar_screen.data import load_assay
from qsar_screen.embeddings import chemeleon_embeddings, default_checkpoint
from qsar_screen.features import murcko_scaffolds

SEED = 42


def _encoder(ckpt):
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    mp = cnn.BondMessagePassing(**blob["hyper_parameters"])
    mp.load_state_dict(blob["state_dict"])
    return mp


def _model(ckpt, freeze):
    mp = _encoder(ckpt)
    model = MPNN(message_passing=mp, agg=cnn.MeanAggregation(),
                 predictor=cnn.BinaryClassificationFFN(input_dim=mp.output_dim),
                 batch_norm=False, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4,
                 warmup_epochs=2)
    if freeze:
        for p in model.message_passing.parameters():
            p.requires_grad = False
    return model


def _trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _dl(smiles, y, idx, feat, shuffle, bs=32):
    dp = [cdata.MoleculeDatapoint.from_smi(smiles[i], np.array([float(y[i])]))
          for i in idx]
    return cdata.build_dataloader(cdata.MoleculeDataset(dp, feat),
                                  batch_size=bs, shuffle=shuffle)


def _run_neural(smiles, y, groups, ckpt, epochs, freeze):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    oof = np.zeros(len(y))
    n_par = None
    t0 = time.time()
    for tr, te in cv.split(smiles, y, groups):
        # oversized eval batch: build_dataloader drops a trailing size-1 batch
        train_dl = _dl(smiles, y, tr, feat, True)
        test_dl = _dl(smiles, y, te, feat, False, bs=512)
        model = _model(ckpt, freeze)
        n_par = n_par or _trainable(model)
        trainer = pl.Trainer(max_epochs=epochs, accelerator="auto", devices=1,
                             enable_checkpointing=False, enable_progress_bar=False,
                             enable_model_summary=False, logger=False)
        trainer.fit(model, train_dl)
        preds = torch.cat(trainer.predict(model, test_dl)).squeeze(-1).numpy().ravel()
        assert len(preds) == len(te), f"{len(preds)} preds for {len(te)} labels"
        oof[te] = preds
    return oof, n_par, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default=None, help="chemeleon_mp.pt (else default)")
    ap.add_argument("--epochs", type=int, default=50, help="chemprop default is 50")
    ap.add_argument("--out", default="results/finetune_comparison.csv")
    args = ap.parse_args()
    ckpt = args.ckpt or default_checkpoint()

    frame = load_assay(args.data, verbose=False)
    smiles = list(frame.smiles)
    y = (frame.activity_remaining.to_numpy() < 65.0).astype(int)
    groups = np.asarray(murcko_scaffolds(smiles))
    print(f"n={len(y)} active={y.sum()} scaffolds={len(set(groups))} "
          f"baseline PR-AUC={y.mean():.4f} epochs={args.epochs}")

    rows = []
    for tag, freeze in [("B frozen+FFN", True), ("A full fine-tune", False)]:
        oof, n_par, secs = _run_neural(smiles, y, groups, ckpt, args.epochs, freeze)
        rows.append(dict(condition=tag, trainable_params=n_par,
                         pr_auc=round(average_precision_score(y, oof), 4),
                         roc_auc=round(roc_auc_score(y, oof), 4),
                         wall_seconds=round(secs, 1)))
        print(f"  {tag}: PR-AUC {rows[-1]['pr_auc']} ROC {rows[-1]['roc_auc']} "
              f"[{secs:.0f}s, {n_par:,} params]")

    # C: frozen encoder + RF
    t0 = time.time()
    X = chemeleon_embeddings(smiles, ckpt)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                    class_weight="balanced", random_state=SEED,
                                    n_jobs=1).fit(X[tr], y[tr])
        oof[te] = rf.predict_proba(X[te])[:, 1]
    rows.append(dict(condition="C frozen+RF", trainable_params=0,
                     pr_auc=round(average_precision_score(y, oof), 4),
                     roc_auc=round(roc_auc_score(y, oof), 4),
                     wall_seconds=round(time.time() - t0, 1)))

    out = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n=== scaffold-split OOF, n={len(y)} ===")
    print(out.to_string(index=False))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
