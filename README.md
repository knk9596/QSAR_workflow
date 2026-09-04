# qsar-hit-screen

Ligand-based hit classification for SLC-family transport inhibitors, built to
triage ultra-large make-on-demand libraries and to serve as the reward inside a
generative design loop.

Trained on a 163-compound transport assay (normalised inhibition, pH-based
readout) from an iterative structure-based campaign against a single target
conformation. The interesting part of the problem is not the classifier — it is
that 163 compounds across 111 Murcko scaffolds is small enough that most
apparent performance differences are noise, so the pipeline is built around
measuring that honestly.

## Install

```bash
pip install -e ".[dev]"
pytest
```

Requires Python ≥ 3.10, RDKit, and scikit-learn. `xgboost` is optional
(`pip install -e ".[extra]"`).

## Quick start

```bash
# score a SMILES file with the shipped model
qsar-score --input molecules.smi

# screen a large library (ZINC22 tranches, Enamine, CSVs)
qsar-screen --input 'zinc22/**/*.smi.gz' --out hits.csv --top-k 100000 --workers 64

# reward for a generative loop
qsar-reward --input generated.smi --out scores.csv
```

```python
from qsar_screen import score_smiles
score_smiles(["O=C(NCCCn1cnc2ccccc21)c1cc(-c2ccccc2)[nH]n1"])
#   prob_active  nn_tanimoto    ad_tier
#      0.899047          1.0  in_domain
```

No dataset is required to try the code — `scripts/make_example_data.py`
enumerates a synthetic congeneric series and the whole pipeline runs on it.

```bash
python scripts/make_example_data.py
python scripts/train.py --data data/example/assay_example.csv
python scripts/benchmark.py --data data/example/assay_example.csv --out benchmark.csv
```

## Method

**Model.** Random forest (500 trees, `min_samples_leaf=3`,
`class_weight='balanced'`) on the full 217-descriptor RDKit block. Variance
filtering and correlation pruning at |r| > 0.95 reduce this to 154 columns,
both fitted inside each fold. Trees stay shallow (median depth 7) — the
small-data regularisation doing its work.

**Validation.** `StratifiedGroupKFold` grouped on Bemis-Murcko scaffold, so
analogue series never straddle a split. This matters here: a t-SNE on ECFP4
Tanimoto distances shows the compound set is organised by synthetic batch
(silhouette +0.075) far more strongly than by activity (+0.009), so a random
split places near-duplicates on both sides. An earlier random-split
configuration reported ROC 0.755; the scaffold-split value for the same
features is 0.738, and the gap was traced to descriptor selection performed
once on the full dataset rather than per fold.

**Significance.** 200-permutation y-scrambling with the whole
cross-validation refitted per permutation, so the null absorbs pipeline
optimism. Pairwise bootstrap over out-of-fold predictions for model
comparisons.

**Applicability domain.** Max ECFP4 Tanimoto to the training set, reported as
graded tiers (`in_domain` ≥ 0.50, `edge` ≥ 0.30, `out_of_domain` below) rather
than a hard gate. Binning held-out performance by similarity could not
identify a defensible cutoff — nearly all actives sit at high similarity, so
competence is only *demonstrable* near the training scaffolds and everything
below is unmeasured rather than known-bad.

## Results

Out-of-fold, scaffold-split, 163 compounds, 50 active (baseline PR-AUC 0.307):

| Features | Model | PR-AUC | ROC | Lift | EF@10% | y-scramble p |
|---|---|---|---|---|---|---|
| Interaction fingerprint | RF | 0.679 | 0.756 | 2.21× | 3.06 | 0.005 |
| Learned embedding (2048-d) | RF | 0.672 | 0.729 | 2.19× | 2.85 | 0.005 |
| FCFP4 | SVM | 0.654 | 0.778 | 2.13× | 2.45 | — |
| **RDKit descriptors (shipped)** | **RF** | **0.646** | **0.738** | **2.11×** | **2.24** | **0.005** |

**These are tied, not ranked.** Every pairwise bootstrap CI crosses zero — the
top five sit inside 0.034 PR-AUC at n = 163. The shipped model is the
descriptor RF because it needs only RDKit, exposes named interpretable
columns, and runs 6.4× faster per core than the learned embedding (240 vs 37
molecules/s), not because it scores highest. The interaction-fingerprint model
is stronger but requires a docked pose, so it cannot run at library scale — it
belongs downstream as a rescorer.

Two findings that shaped the design:

- **Univariate selection was dropped.** A `SelectKBest(k=30)` step inherited
  from an earlier pipeline turned out to be the post-hoc peak of a flat
  plateau: +0.035 PR-AUC over keeping all 154 pruned columns, CI
  [−0.007, +0.079], and nested cross-validation chose a different *k* per fold
  with a lower unbiased estimate (0.665). FDR-based selection performed worst,
  because at this sample size it is conservative enough to sometimes retain
  almost nothing — statistical significance and predictive utility are not the
  same criterion.
- **Pose-ensemble features hurt.** Averaging interaction fingerprints over all
  docked poses ranked at the bottom of the full grid; only the top-ranked pose
  carries signal. Composite docking scores themselves were flat against
  potency (ρ = −0.04, p = 0.62) while the top pose read through interaction
  fingerprints reached ρ = +0.44.

## Two limits worth stating

**The classifier ranks "worth assaying", not potency.** Within known actives,
predicted probability against potency is ρ = +0.14 (p = 0.33). Use it to
triage; do not use it to order candidates by expected potency.

**Recall is ~0.56 at a single threshold.** Roughly half the actives are missed
by any hard cut. Rank the library and take a top slice rather than treating the
output as a yes/no filter.

The generative reward is built around the first limit. Maximising
`prob_active` alone is trivially gamed by a model with 2× enrichment — a C30
straight-chain alkane scores 0.32, above aspirin at 0.19. The reward multiplies
in an applicability term, a saturating novelty term, and a QED × synthetic
accessibility term, which puts that alkane 23× below a plausible analogue.
Weights are exponents, so `novelty_weight=0` gives analogue expansion and
`ad_weight=0.5` gives softer scaffold hopping.

## Layout

```
src/qsar_screen/
  features.py     fingerprints, descriptors, Murcko scaffolds, similarity
  transforms.py   CorrPrune (deterministic correlation pruning)
  model.py        pipeline, training, packaging, scoring
  evaluate.py     scaffold CV, metrics, enrichment, permutation test
  screen.py       streaming parallel library screening, resumable
  reward.py       guardrailed generative reward
  cli.py          qsar-score / qsar-screen / qsar-reward
scripts/          train, benchmark, synthetic example data
tests/            45 tests
models/           trained classifier (332 KB)
```

`screen.py` streams input, so library size is bounded by disk rather than RAM;
only a top-K heap is held. State is checkpointed, so rerunning the same command
after a walltime kill resumes. Measured 240 molecules/s/core, of which RDKit
descriptors are 97% — so throughput scales with cores: 100M in ~1.8 h on 64
cores.

## Data availability

The underlying assay data is unpublished. This repository ships the trained
classifier and the full method; the compound structures and activity values are
withheld pending publication. The synthetic example dataset exists so that
every code path is runnable without it.

## License

MIT
