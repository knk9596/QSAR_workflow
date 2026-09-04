"""Reward function for generative design loops (REINVENT, GB-GA, STONED)."""

from __future__ import annotations

import os
import sys

import numpy as np

from qsar_screen.model import AD_EDGE, AD_IN_DOMAIN, DEFAULT_MODEL_DIR, load_scorer

NOVELTY_SATURATION = 0.60   # similarity at or below which a molecule is fully novel
AD_FLOOR = 0.15             # keeps exploration from being multiplied to zero
NOVELTY_FLOOR = 0.05

_CACHE: dict = {}


def _scorer(model_dir: str):
    if model_dir not in _CACHE:
        _CACHE[model_dir] = load_scorer(model_dir)
    return _CACHE[model_dir]


def synthetic_accessibility(mol) -> float:
    """SA score, 1 (easy) to 10 (hard); ring/size proxy if the contrib module is absent."""
    try:
        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        return float(sascorer.calculateScore(mol))
    except Exception:
        from rdkit.Chem import Descriptors, rdMolDescriptors
        rings = rdMolDescriptors.CalcNumRings(mol)
        heavy = Descriptors.HeavyAtomCount(mol)
        return float(np.clip(1.5 + 0.35 * rings + 0.01 * heavy, 1.0, 10.0))


def components(smiles, model_dir: str = DEFAULT_MODEL_DIR) -> dict:
    """Unweighted reward terms, for inspection and diagnosis."""
    from rdkit import Chem
    from rdkit.Chem import QED

    scorer = _scorer(model_dir)
    smiles = list(smiles)
    n = len(smiles)
    out = {k: np.zeros(n) for k in
           ("prob_active", "nn_tanimoto", "qed", "sa", "valid")}

    valid_idx, valid_smiles, mols = [], [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        valid_idx.append(i)
        valid_smiles.append(smi)
        mols.append(mol)
        out["valid"][i] = 1.0
    if not mols:
        return out

    probs = scorer.predict_proba(valid_smiles)
    nn = scorer.applicability(valid_smiles)
    for slot, (i, mol) in enumerate(zip(valid_idx, mols)):
        out["prob_active"][i] = probs[slot]
        out["nn_tanimoto"][i] = nn[slot]
        try:
            out["qed"][i] = float(QED.qed(mol))
        except Exception:
            out["qed"][i] = 0.0
        out["sa"][i] = synthetic_accessibility(mol)
    return out


def reward(smiles, ad_weight: float = 1.0, novelty_weight: float = 1.0,
           quality_weight: float = 1.0, model_dir: str = DEFAULT_MODEL_DIR,
           return_components: bool = False):
    """Multiplicative reward in [0, 1]; invalid SMILES score 0.

    A model with ~2x enrichment is a weak optimisation target: maximising
    ``prob_active`` alone finds molecules that exploit the region where the
    model has no training data. Three guardrails multiply it down.

    Weights are exponents (0 disables a term):
      ``novelty_weight=0``    analogue expansion, stay near known chemistry
      ``ad_weight=0.5``       scaffold hopping with a soft competence floor
      ``ad_weight=0``         unconstrained, and readily gamed
    """
    parts = components(smiles, model_dir)
    nn = parts["nn_tanimoto"]

    ad = np.clip((nn - AD_EDGE) / (AD_IN_DOMAIN - AD_EDGE), 0.0, 1.0)
    ad = AD_FLOOR + (1.0 - AD_FLOOR) * ad

    # Saturating ramp, not a raw 1 - nn: that gives exactly zero for a training
    # compound and removes the gradient for near-duplicates.
    novelty = np.clip((1.0 - nn) / (1.0 - NOVELTY_SATURATION), 0.0, 1.0)
    novelty = NOVELTY_FLOOR + (1.0 - NOVELTY_FLOOR) * novelty

    quality = parts["qed"] * np.clip((10.0 - parts["sa"]) / 9.0, 0.0, 1.0)

    eps = 1e-9
    total = (parts["prob_active"]
             * np.power(np.clip(ad, eps, 1.0), ad_weight)
             * np.power(np.clip(novelty, eps, 1.0), novelty_weight)
             * np.power(np.clip(quality, eps, 1.0), quality_weight))
    total = np.where(parts["valid"] > 0, total, 0.0)

    if return_components:
        return {**parts, "ad": ad, "novelty": novelty,
                "quality": quality, "reward": total}
    return total
