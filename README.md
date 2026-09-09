# qsar-hit-screen

Hit classification for SLC-family transport inhibitors. Benchmarks three
feature representations — ligand-based descriptors, protein-ligand interaction
fingerprints from docked poses, and a frozen molecular foundation-model
embedding — under one scaffold-split protocol, and packages the resulting
classifier.

Trained on a 163-compound transport assay (normalised inhibition, pH-based
readout) from an iterative structure-based campaign against a single target
conformation. The interesting part of the problem is not the classifier: 163
compounds across 111 Murcko scaffolds is small enough that most apparent
performance differences are noise, so the pipeline is built to measure that
honestly rather than to report a winner.

## Install

```bash
pip install -e ".[dev]"          # core + tests
pip install -e ".[all]"          # adds xgboost, chemprop/torch, prolif
pytest
```

Python ≥ 3.10. The core path needs only RDKit and scikit-learn; gradient
boosting, embeddings, and interaction fingerprints are optional extras, and
each is skipped cleanly when its dependency is absent.

## Quick start

```python
from qsar_screen import score_smiles
score_smiles(["O=C(NCCCn1cnc2ccccc21)c1cc(-c2ccccc2)[nH]n1"])
#   prob_active  nn_tanimoto    ad_tier
#      0.899047          1.0  in_domain
```

```bash
qsar-score --input molecules.smi           # score with the packaged model
```

No dataset is needed to try the code — `scripts/make_example_data.py`
enumerates a synthetic congeneric series and the whole pipeline runs on it:

```bash
python scripts/make_example_data.py
python scripts/train.py     --data data/example/assay_example.csv
python scripts/benchmark.py --data data/example/assay_example.csv --out benchmark.csv
```

To run it on a real assay table, see **Input format** below — the columns are
auto-detected, so `--data your_file.csv` is usually the whole command.

## Input format

`load_assay` reads a raw assay export and returns the QC'd modelling set. It
auto-detects the columns, so no flags are needed for a typical plate export:

| role | column names recognised (first match wins) |
|---|---|
| structure | `smiles_canon`, `smiles`, `SMILES`, `canonical_smiles` |
| activity | `activity_remaining`, `avg_inhibition`, `activity`, `pct_activity_remaining` |
| identifier | `ID`, `id`, `compound`, `compound_id`, `name` |

Extra columns are ignored, except `notes`/`series`/`batch`, which are carried
through for grouping.

```bash
python scripts/train.py --data path/to/your_assay.csv
#   your_assay.csv: 166 rows -> 163 after QC
#     columns used: id=ID, smiles=smiles, activity=avg_inhibition
#     dropped 3 readings above the 100% ceiling (assay artefacts, not inactives)
```

**The raw export is not the modelling set**, and the QC is not optional:

- **Readings above the vehicle-control ceiling (100%) are dropped, not
  clipped.** 
- **SMILES are canonicalised before any structural join or de-duplication**,


Activity direction: **lower means more potent** (vehicle control = 100% activity
remaining), and `active = activity_remaining < 65`. See the threshold note under
Method.

## Feature representations

| Block | Source | Needs |
|---|---|---|
| ECFP4 / FCFP4 / MACCS | SMILES | rdkit |
| RDKit descriptors| SMILES | rdkit |
| Interaction fingerprint | docked pose + receptor, via ProLIF | `[structural]` |
| CheMeleon embedding (2048-d) | frozen pretrained MPNN encoder | `[embeddings]` |


```bash
# interaction fingerprints from a best-pose SDF (one row per compound)
python -c "from qsar_screen.structural import generate_prolif_fingerprints as g; \
           g('receptor.pdb', 'best_poses.sdf', 'prolif.csv')"

# frozen embeddings, precomputed so the benchmark needs no torch
python scripts/generate_embeddings.py --data assay.csv \
    --ckpt models/chemeleon_mp.pt --out models/emb.npy

# full 6 x 3 grid
python scripts/benchmark.py --data assay.csv --smiles-col smiles \
    --prolif-csv prolif.csv --prolif-aliases cmpd29v=cmpd29 \
    --chemeleon-npy models/emb.npy --out benchmark.csv
```


## Method

**Model.** Random forest (500 trees, `min_samples_leaf=3`,
`class_weight='balanced'`) as the shipped classifier, with SVM and gradient
boosting as benchmark comparators. Bit vectors get variance filtering only;
dense blocks are standardised and correlation-pruned at |r| > 0.95, both fitted
per fold.

**Validation.** `StratifiedGroupKFold` grouped on Bemis-Murcko scaffold.

**Significance.** 200-permutation y-scrambling with the whole cross-validation
refitted per permutation, so the null absorbs pipeline optimism. Pairwise
bootstrap over out-of-fold predictions for model comparisons.

**Applicability domain.** Max ECFP4 Tanimoto to the training set, reported as
graded tiers (`in_domain` ≥ 0.50, `edge` ≥ 0.30, `out_of_domain` below) rather
than a hard gate. 



## Layout

```
src/qsar_screen/
  data.py         raw-assay loading + QC (column detection, ceiling, canonicalisation)
  features.py     fingerprints, descriptors, Murcko scaffolds, similarity
  structural.py   ProLIF interaction fingerprints, pose alignment
  embeddings.py   frozen CheMeleon encoder
  transforms.py   CorrPrune (deterministic correlation pruning)
  model.py        pipeline, training, packaging, scoring
  evaluate.py     scaffold CV, metrics, enrichment, permutation test
  cli.py          qsar-score
scripts/          train, benchmark, generate_embeddings, make_example_data
tests/            42 tests (5 more run when torch is installed)
models/           packaged classifier (332 KB)
results/          reference benchmark table
```

## Data availability

This repository ships the trained
classifier and the full method; compound structures and activity values are
withheld pending publication. The synthetic example dataset exists so every
code path is runnable without them.

The CheMeleon checkpoint (`chemeleon_mp.pt`, 35 MB) is third-party and not
vendored here — download it from Zenodo record 15460715 into `models/`, or
point `$CHEMELEON_CHECKPOINT` at it.

## License

MIT
