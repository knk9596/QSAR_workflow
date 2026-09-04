"""Model definition, training, packaging, and single-shot scoring."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qsar_screen.features import DESCRIPTOR_NAMES, ecfp4, nn_tanimoto, rdkit_descriptors
from qsar_screen.transforms import CorrPrune

ACTIVITY_THRESHOLD = 65.0    # active if activity_remaining < 65 %
CORR_CUTOFF = 0.95
RANDOM_SEED = 42
AD_IN_DOMAIN = 0.50
AD_EDGE = 0.30

DEFAULT_MODEL_DIR = os.environ.get(
    "QSAR_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "models"),
)


def estimator(kind: str = "rf", n_pos: int | None = None, n_neg: int | None = None):
    """Classifier with class-imbalance handling appropriate to `kind`."""
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1)
    if kind == "svm":
        return SVC(C=1.0, probability=True, class_weight="balanced",
                   random_state=RANDOM_SEED)
    if kind == "xgb":
        import xgboost as xgb
        ratio = (n_neg / n_pos) if (n_pos and n_neg) else 1.0
        return xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=ratio, random_state=RANDOM_SEED, n_jobs=1,
            eval_metric="logloss")
    raise ValueError(f"unknown estimator {kind!r}")


def build_pipeline(kind: str = "rf", binary_features: bool = False, **est_kw) -> Pipeline:
    """Reduction + classifier.

    Bit vectors get variance filtering only. Dense blocks are scaled and
    redundancy-pruned. There is no univariate selection step: a swept
    ``SelectKBest`` gave no significant gain over keeping all pruned columns.
    """
    steps = [("var", VarianceThreshold())]
    if not binary_features:
        steps += [("scale", StandardScaler()),
                  ("prune", CorrPrune(CORR_CUTOFF)),
                  ("rescale", StandardScaler())]
    steps.append(("clf", estimator(kind, **est_kw)))
    return Pipeline(steps)


@dataclass
class Scorer:
    """A fitted pipeline plus the reference data needed for domain checks."""

    pipeline: Pipeline
    descriptor_names: list[str]
    ad_reference: np.ndarray
    metrics: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.pipeline.steps[-1][1].n_features_in_)

    def predict_proba(self, smiles) -> np.ndarray:
        X, _ = rdkit_descriptors(smiles, self.descriptor_names)
        return self.pipeline.predict_proba(X)[:, 1]

    def applicability(self, smiles) -> np.ndarray:
        return nn_tanimoto(smiles, self.ad_reference)


def ad_tier(nn: float) -> str:
    """Bucket a nearest-neighbour similarity into a confidence tier."""
    if not np.isfinite(nn):
        return "unparseable"
    if nn >= AD_IN_DOMAIN:
        return "in_domain"
    if nn >= AD_EDGE:
        return "edge"
    return "out_of_domain"


def train(smiles, activity_remaining, threshold: float = ACTIVITY_THRESHOLD,
          kind: str = "rf") -> Scorer:
    """Fit a scorer on the full dataset.

    Raises if either class is empty: a single-class fit trains without error but
    yields a one-column ``predict_proba``, which fails only later at scoring time.
    """
    y = (np.asarray(activity_remaining, dtype=float) < threshold).astype(int)
    n_active = int(y.sum())
    if n_active == 0 or n_active == len(y):
        raise ValueError(
            f"need both classes to train: {n_active}/{len(y)} molecules are active "
            f"at threshold {threshold}. Adjust the threshold or use more data.")
    X, _ = rdkit_descriptors(smiles, DESCRIPTOR_NAMES)
    pipe = build_pipeline(kind, binary_features=False,
                          n_pos=int(y.sum()), n_neg=int((1 - y).sum()))
    pipe.fit(X, y)
    return Scorer(
        pipeline=pipe,
        descriptor_names=list(DESCRIPTOR_NAMES),
        ad_reference=ecfp4(smiles),
        metrics={"n_train": len(y), "n_active": int(y.sum()),
                 "baseline_pr_auc": round(float(y.mean()), 4),
                 "activity_threshold": threshold},
    )


def save_scorer(scorer: Scorer, model_dir: str = DEFAULT_MODEL_DIR) -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "hit_classifier.joblib")
    joblib.dump({"pipeline": scorer.pipeline,
                 "descriptor_names": scorer.descriptor_names,
                 "ad_reference": scorer.ad_reference,
                 "metrics": scorer.metrics}, path, compress=3)
    return path


def load_scorer(model_dir: str = DEFAULT_MODEL_DIR) -> Scorer:
    path = os.path.join(model_dir, "hit_classifier.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no model at {path}. Train one with `python scripts/train.py`.")
    blob = joblib.load(path)
    return Scorer(blob["pipeline"], blob["descriptor_names"],
                  blob["ad_reference"], blob.get("metrics", {}))


def score_smiles(smiles, scorer: Scorer | None = None,
                 model_dir: str = DEFAULT_MODEL_DIR) -> pd.DataFrame:
    """Score SMILES, best first. Probability is P(hit), not potency."""
    scorer = scorer or load_scorer(model_dir)
    smiles = list(smiles)
    probs = scorer.predict_proba(smiles)
    nn = scorer.applicability(smiles)
    out = pd.DataFrame({"smiles": smiles, "prob_active": probs,
                        "nn_tanimoto": nn,
                        "ad_tier": [ad_tier(v) for v in nn]})
    return out.sort_values("prob_active", ascending=False).reset_index(drop=True)
