"""Reproduce CellOracle's Fig-S2 (AUROC/EPR) from their released inference_results.

Faithful re-implementation of the authors' GRN_benchmarking.py scoring (from the
janursa/CO_evaluation repo), so our numbers are directly comparable to Fig. S2a/b.
Candidate universe = permutations(genes_used ∩ ground-truth genes, 2), self-loops
excluded, and sources that are known TFs lacking ChIP data are dropped. AUROC over
that universe (absent edges score 0); EPR = top-k precision / positive-rate, k=#pos.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import CFG, CO_CHIP_GT_DIR, CO_INFERENCE_DIR
from .edges import CANON_COLS, _finalize_edges

# weight column used for ranking, by method (dir-name substring -> column).
CO_WEIGHT_COL = {
    "celloracle": "coef_abs",
    "genie3": "weight",
    "wgcna": "weight",
    "dcol": "weight",
    "scenic": "CoexWeight",
}


def load_all_tfs(path: Optional[str] = None) -> set:
    """gimme v5 mouse TF list used to decide which sources are 'known TFs'."""
    path = path or os.path.join(CO_CHIP_GT_DIR, "TFs_in_gimmev5_mouse.npy")
    return set(np.load(path, allow_pickle=True).tolist())


def load_chip_atlas_gt(tissue: str) -> pd.DataFrame:
    """ChIP-Atlas ground truth for a tissue: chip_GT_links.csv [tf, target]."""
    path = os.path.join(CO_CHIP_GT_DIR, tissue, "chip_GT_links.csv")
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns={"tf": "source", "target": "target"})
    df["weight"] = 1.0
    df["sign"] = 0
    return _finalize_edges(df[["source", "target", "weight", "sign"]])


def load_co_link(method_dir: str) -> pd.DataFrame:
    """One method's inferred network (link.csv) -> canonical edges.

    Column layout differs by method; the ranking score is picked per CO_WEIGHT_COL.
    CellOracle also carries a signed coef_mean, preserved as the sign.
    """
    name = os.path.basename(method_dir.rstrip("/")).lower()
    wcol = next((c for k, c in CO_WEIGHT_COL.items() if k in name), None)
    if wcol is None:
        raise ValueError(f"unknown method for {method_dir}")
    df = pd.read_csv(os.path.join(method_dir, "link.csv"), index_col=0)
    df = df.rename(columns={"regulatoryGene": "source", "TF": "source",
                            "targetGene": "target", "gene": "target"})
    df["score"] = df[wcol].abs()
    df["weight"] = df["coef_mean"] if "coef_mean" in df.columns else df["score"]
    df["sign"] = np.sign(df["weight"]).fillna(0).astype(int) if "coef_mean" in df.columns else 0
    df = df.dropna(subset=["source", "target"])
    df = df.drop_duplicates(subset=["source", "target"])
    return df[CANON_COLS].reset_index(drop=True)


def load_co_genes_used(method_dir: str) -> list:
    return pd.read_csv(os.path.join(method_dir, "genes_nonzero.csv"),
                       index_col=0).x.values.tolist()


def co_reproduce_metrics(link: pd.DataFrame, gt: pd.DataFrame, genes_used: list,
                         all_tfs: set) -> dict:
    """AUROC + EPR matching the CellOracle paper's benchmark exactly.

    AUROC is computed analytically over the full candidate universe (the bulk of
    which are absent edges scored 0) without materialising it: partition candidates
    into scored (present in link) and unscored (0), and combine via the Mann-Whitney
    identity so ties at 0 are handled the way roc_auc_score would.
    """
    gt_genes = set(gt["source"]) | set(gt["target"])
    all_genes = set(genes_used) & gt_genes
    gt_tfs = set(gt["source"])
    # sources that are known TFs but have no ChIP data are not judgeable -> dropped.
    kept_sources = {g for g in all_genes if not (g in all_tfs and g not in gt_tfs)}

    n_all = len(all_genes)
    n_universe = len(kept_sources) * (n_all - 1)          # ordered pairs, no self
    if n_universe == 0:
        return {"auroc": np.nan, "epr": np.nan, "n_gold": 0, "n_candidates": 0}

    gt_keys = {(s, t) for s, t in zip(gt["source"], gt["target"])
               if s in kept_sources and t in all_genes and s != t}
    P = len(gt_keys)
    N = n_universe - P
    if P == 0 or N == 0:
        return {"auroc": np.nan, "epr": np.nan, "n_gold": P, "n_candidates": n_universe}

    # present (scored) edges restricted to the candidate universe
    pres = link[[s in kept_sources and t in all_genes and s != t
                 for s, t in zip(link["source"], link["target"])]]
    pres_keys = list(zip(pres["source"], pres["target"]))
    pres_label = np.array([1 if k in gt_keys else 0 for k in pres_keys])
    pres_score = pres["score"].to_numpy()
    P_nz = int(pres_label.sum())
    N_nz = len(pres_label) - P_nz
    P_z, N_z = P - P_nz, N - N_nz

    # AUROC = [pairwise AUC among scored] * P_nz*N_nz + (scored pos > 0 neg) + 0.5*(0 pos vs 0 neg)
    auc_nz = (roc_auc_score(pres_label, pres_score)
              if P_nz > 0 and N_nz > 0 else 0.0)
    numerator = auc_nz * P_nz * N_nz + P_nz * N_z + 0.5 * P_z * N_z
    auroc = numerator / (P * N)

    # EPR: k = #positives; top-k by score. present edges dominate the ranking.
    k = P
    if len(pres_score) >= k:
        top_idx = np.argsort(-pres_score)[:k]
        ep = pres_label[top_idx].mean()
    else:
        # all present in top, remaining slots are tied-zero (expected positive rate)
        pad = k - len(pres_label)
        zero_rate = P_z / (P_z + N_z) if (P_z + N_z) else 0.0
        ep = (P_nz + pad * zero_rate) / k
    epr = ep / (P / n_universe)

    return {"auroc": auroc, "epr": epr, "n_gold": P, "n_candidates": n_universe}


def run_co_reproduction(sample: str, inference_dir: str = CO_INFERENCE_DIR) -> pd.DataFrame:
    """Score every method under one sample dir against its tissue ChIP-Atlas GT."""
    tissue = sample.split("-")[0]
    gt = load_chip_atlas_gt(tissue)
    all_tfs = load_all_tfs()
    sample_dir = os.path.join(inference_dir, sample)
    rows = {}
    for method in sorted(os.listdir(sample_dir)):
        mdir = os.path.join(sample_dir, method)
        if not os.path.isdir(mdir):
            continue
        try:
            link = load_co_link(mdir)
            genes = load_co_genes_used(mdir)
        except (ValueError, FileNotFoundError) as e:
            print(f"[skip] {sample}/{method}: {e}")
            continue
        rows[method] = co_reproduce_metrics(link, gt, genes, all_tfs)
    return pd.DataFrame(rows).T


def run_co_reproduction_all(inference_dir: str = CO_INFERENCE_DIR) -> pd.DataFrame:
    """Reproduce Fig. 7a/b across every sample present in inference_dir."""
    samples = sorted(s for s in os.listdir(inference_dir)
                     if os.path.isdir(os.path.join(inference_dir, s)))
    li = []
    for s in samples:
        df = run_co_reproduction(s, inference_dir)
        df["sample"] = s
        df["tissue"] = s.split("-")[0]
        li.append(df.reset_index().rename(columns={"index": "method"}))
        print(f"[co-repro] {s}: {len(df)} methods scored")
    out = pd.concat(li, ignore_index=True)
    os.makedirs(CFG.out_dir, exist_ok=True)
    out.to_csv(os.path.join(CFG.out_dir, "co_reproduction_scores.csv"), index=False)
    return out
