"""Streaming, parallel screening of large SMILES libraries."""

from __future__ import annotations

import glob
import gzip
import heapq
import json
import os
import sys
import time
from typing import Iterator

import numpy as np
import pandas as pd

from qsar_screen.model import AD_EDGE, AD_IN_DOMAIN, DEFAULT_MODEL_DIR, ad_tier

_STATE: dict = {}

ID_COLUMN_CANDIDATES = ("zinc_id", "id", "idnumber", "catalog_id", "name")
LIBRARY_EXTENSIONS = ("*.smi", "*.smi.gz", "*.csv", "*.csv.gz", "*.txt", "*.txt.gz")


def _init_worker(model_dir: str):
    from rdkit import DataStructs

    from qsar_screen.model import load_scorer

    scorer = load_scorer(model_dir)
    _STATE["scorer"] = scorer
    _STATE["ad_ref"] = [DataStructs.CreateFromBitString("".join(map(str, row)))
                        for row in np.asarray(scorer.ad_reference, dtype=np.int8)]


def _score_chunk(task):
    """Score one chunk; returns (rows, n_seen, n_invalid).

    Two passes: the applicability check parses and fingerprints (cheap), then
    descriptors run only on survivors. Descriptors dominate the cost, so the
    optional prefilter is where the speedup comes from.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    from qsar_screen.features import ECFP_BITS, ECFP_RADIUS, rdkit_descriptors

    pairs, prefilter, ad_min = task
    scorer, ad_ref = _STATE["scorer"], _STATE["ad_ref"]
    n_invalid = 0
    survivors = []

    for smi, cid in pairs:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid += 1
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, ECFP_RADIUS, nBits=ECFP_BITS)
        nn = max(DataStructs.BulkTanimotoSimilarity(fp, ad_ref))
        if prefilter and nn < ad_min:
            continue
        survivors.append((smi, cid, nn))

    if not survivors:
        return [], len(pairs), n_invalid

    X, _ = rdkit_descriptors([s for s, _, _ in survivors], scorer.descriptor_names)
    probs = scorer.pipeline.predict_proba(X)[:, 1]
    rows = [(float(probs[i]), survivors[i][1], survivors[i][0], float(survivors[i][2]))
            for i in range(len(survivors))]
    return rows, len(pairs), n_invalid


def expand_inputs(patterns) -> list[str]:
    """Resolve files, directories, and globs to a sorted list of file paths."""
    paths: list[str] = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            for ext in LIBRARY_EXTENSIONS:
                paths += sorted(glob.glob(os.path.join(pattern, "**", ext),
                                          recursive=True))
        else:
            hits = sorted(glob.glob(pattern, recursive=True))
            paths += hits if hits else ([pattern] if os.path.exists(pattern) else [])
    return [p for p in paths if os.path.isfile(p)]


def _open_any(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def sniff_format(path: str, smiles_col: str = "smiles", id_col: str | None = None):
    """Infer (delimiter, smiles_index, id_index, has_header) from the first line."""
    with _open_any(path) as handle:
        first = handle.readline().rstrip("\n")
    delim = "\t" if "\t" in first else ("," if "," in first else None)
    fields = first.split(delim) if delim else first.split()
    lowered = [f.strip().strip('"').lower() for f in fields]

    has_header = smiles_col.lower() in lowered or "smiles" in lowered
    if has_header:
        key = smiles_col.lower() if smiles_col.lower() in lowered else "smiles"
        s_idx = lowered.index(key)
        i_idx = lowered.index(id_col.lower()) if id_col and id_col.lower() in lowered else None
        if i_idx is None:
            for cand in ID_COLUMN_CANDIDATES:
                if cand in lowered:
                    i_idx = lowered.index(cand)
                    break
    else:
        s_idx, i_idx = 0, (1 if len(fields) > 1 else None)   # ZINC .smi convention
    return delim, s_idx, i_idx, has_header


def stream_library(paths, smiles_col: str = "smiles", id_col: str | None = None,
                   chunk_size: int = 5000) -> Iterator[list[tuple[str, str]]]:
    """Yield chunks of (smiles, id) without holding a library in memory."""
    for path in paths:
        delim, s_idx, i_idx, has_header = sniff_format(path, smiles_col, id_col)
        stem = os.path.basename(path)
        buffer: list[tuple[str, str]] = []
        with _open_any(path) as handle:
            if has_header:
                handle.readline()
            for line_no, line in enumerate(handle):
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(delim) if delim else line.split()
                if len(parts) <= s_idx:
                    continue
                smi = parts[s_idx].strip().strip('"')
                cid = (parts[i_idx].strip().strip('"')
                       if i_idx is not None and len(parts) > i_idx
                       else f"{stem}:{line_no}")
                buffer.append((smi, cid))
                if len(buffer) >= chunk_size:
                    yield buffer
                    buffer = []
        if buffer:
            yield buffer


def screen(inputs, out_csv: str, top_k: int = 100_000, workers: int = 4,
           chunk_size: int = 5000, smiles_col: str = "smiles",
           id_col: str | None = None, ad_prefilter: bool = False,
           ad_min: float = AD_EDGE, model_dir: str = DEFAULT_MODEL_DIR,
           checkpoint_every: int = 200, limit: int = 0, dedup: bool = True,
           progress: bool = True) -> pd.DataFrame:
    """Score a library and keep the best ``top_k`` molecules.

    Resumable: state is checkpointed to ``<out_csv>.ckpt.json``; rerunning the
    same command continues from the last checkpoint.
    """
    import multiprocessing as mp

    paths = expand_inputs(inputs)
    if not paths:
        raise FileNotFoundError(f"no input files matched {inputs!r}")

    ckpt_path = out_csv + ".ckpt.json"
    done, heap, seen, invalid, scored = 0, [], 0, 0, 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as handle:
            state = json.load(handle)
        done, seen = state["done_chunks"], state["n_seen"]
        invalid, scored = state["n_invalid"], state["n_scored"]
        heap = [tuple(row) for row in state["heap"]]
        heapq.heapify(heap)
        if progress:
            print(f"resuming after {done} chunks ({seen:,} molecules)", flush=True)

    def tasks():
        for i, chunk in enumerate(stream_library(paths, smiles_col, id_col, chunk_size)):
            if i >= done:
                yield (chunk, ad_prefilter, ad_min)

    def checkpoint(n_chunks):
        tmp = ckpt_path + ".tmp"
        with open(tmp, "w") as handle:
            json.dump({"done_chunks": n_chunks, "n_seen": seen, "n_invalid": invalid,
                       "n_scored": scored, "heap": [list(r) for r in heap]}, handle)
        os.replace(tmp, ckpt_path)

    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    start = time.time()
    with ctx.Pool(workers, initializer=_init_worker, initargs=(model_dir,)) as pool:
        n_chunks = done
        for rows, n_seen, n_bad in pool.imap_unordered(_score_chunk, tasks(), chunksize=1):
            n_chunks += 1
            seen += n_seen
            invalid += n_bad
            scored += len(rows)
            for row in rows:
                if len(heap) < top_k:
                    heapq.heappush(heap, row)
                elif row[0] > heap[0][0]:
                    heapq.heapreplace(heap, row)
            if progress and n_chunks % 20 == 0:
                rate = seen / max(time.time() - start, 1e-9)
                print(f"  {seen:>12,} seen | {scored:>12,} scored | {rate:8.0f} mol/s",
                      flush=True)
            if n_chunks % checkpoint_every == 0:
                checkpoint(n_chunks)
            if limit and seen >= limit:
                break
        checkpoint(n_chunks)

    hits = pd.DataFrame(sorted(heap, reverse=True),
                        columns=["prob_active", "id", "smiles", "nn_tanimoto"])
    if dedup:
        hits = hits.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)
    hits["ad_tier"] = [ad_tier(v) for v in hits.nn_tanimoto]
    hits = hits[["id", "smiles", "prob_active", "nn_tanimoto", "ad_tier"]]
    hits.to_csv(out_csv, index=False)

    if progress:
        elapsed = time.time() - start
        print(f"\nseen {seen:,} | invalid {invalid:,} | kept {len(hits):,} -> {out_csv}")
        print(f"{elapsed / 3600:.2f} h at {seen / max(elapsed, 1e-9):.0f} mol/s")
    return hits
