"""The three comparison points: network structure, edge weights (BEELINE), perturbation.

1. structure_metrics / run_structure_comparison / pairwise_jaccard  -- topology overlap
2. beeline_metrics / run_beeline_comparison                          -- AUROC / AUPRC / EPR
3. run_perturbation_comparison                                       -- stubbed
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import CFG
from .edges import edge_key, edge_set, edges_to_graph, restrict_to_nodes


# ---------------------------------------------------------------------------
# Comparison 1: NETWORK STRUCTURE (topology vs. reference)
# ---------------------------------------------------------------------------

def structure_metrics(
    pred: pd.DataFrame, ref: pd.DataFrame, directed: bool = CFG.directed
) -> dict:
    """Binary edge/topology agreement between a predicted network and a reference.

    Restrict both to their shared node universe first so absent-by-construction
    edges do not distort precision/recall.
    """
    shared = (set(pred.source) | set(pred.target)) & (set(ref.source) | set(ref.target))
    p = restrict_to_nodes(pred, shared)
    r = restrict_to_nodes(ref, shared)

    P = edge_set(p, directed)
    R = edge_set(r, directed)
    tp = len(P & R)
    precision = tp / len(P) if P else np.nan
    recall = tp / len(R) if R else np.nan
    jaccard = len(P & R) / len(P | R) if (P | R) else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if tp else 0.0

    # Degree-profile agreement across shared nodes.
    Gp, Gr = edges_to_graph(p, directed), edges_to_graph(r, directed)
    nodes = sorted(shared)
    dp = np.array([Gp.degree(n) if n in Gp else 0 for n in nodes], dtype=float)
    dr = np.array([Gr.degree(n) if n in Gr else 0 for n in nodes], dtype=float)
    # Spearman is undefined if either degree profile is constant (e.g. a uniform-random net).
    if len(nodes) > 2 and dp.std() > 0 and dr.std() > 0:
        degree_spearman = pd.Series(dp).corr(pd.Series(dr), method="spearman")
    else:
        degree_spearman = np.nan

    # Sign concordance over edges present in both (activator vs. repressor).
    if "sign" in p and "sign" in r:
        pm = {edge_key(s, t, directed): sg
              for s, t, sg in zip(p["source"], p["target"], p["sign"])}
        rm = {edge_key(s, t, directed): sg
              for s, t, sg in zip(r["source"], r["target"], r["sign"])}
        both = [k for k in (P & R) if pm.get(k) and rm.get(k)]
        sign_conc = np.mean([pm[k] == rm[k] for k in both]) if both else np.nan
    else:
        sign_conc = np.nan

    return {
        "n_shared_nodes": len(shared),
        "n_pred_edges": len(P),
        "n_ref_edges": len(R),
        "n_overlap": tp,
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "degree_spearman": degree_spearman,
        "sign_concordance": sign_conc,
    }


def run_structure_comparison(
    method_edges: dict[str, pd.DataFrame], ref: pd.DataFrame
) -> pd.DataFrame:
    rows = {name: structure_metrics(e, ref) for name, e in method_edges.items()}
    return pd.DataFrame(rows).T


def pairwise_jaccard(
    method_edges: dict[str, pd.DataFrame], directed: bool = CFG.directed
) -> pd.DataFrame:
    """Method-vs-method edge Jaccard (no reference needed) -- do methods agree?"""
    names = list(method_edges)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    sets = {n: edge_set(e, directed) for n, e in method_edges.items()}
    for a in names:
        for b in names:
            u = sets[a] | sets[b]
            M.loc[a, b] = len(sets[a] & sets[b]) / len(u) if u else np.nan
    return M


# ---------------------------------------------------------------------------
# Comparison 2: EDGE WEIGHTS (BEELINE-style ranked-edge scoring)
# ---------------------------------------------------------------------------

def _candidate_universe(nodes: set[str], allow_self: bool = True) -> list[tuple]:
    """All possible directed TF->target pairs over the node set."""
    pairs = [(a, b) for a in nodes for b in nodes if allow_self or a != b]
    return pairs


def beeline_metrics(
    pred: pd.DataFrame,
    gold: pd.DataFrame,
    nodes: Optional[set[str]] = None,
    directed: bool = True,
    allow_self: bool = False,
) -> dict:
    """AUROC / AUPRC / EPR for a ranked edge list vs. a gold-standard network.

    Every possible pair over `nodes` is a candidate; label = in gold standard;
    score = predicted edge score (0 for pairs the method did not predict).
    EPR = precision within the top-k predictions (k = #gold edges), divided by the
    random-baseline density (k / #candidates). EPR > 1 => better than random.

    BEELINE convention (Pratapa et al. 2020): the candidate universe is the set of
    ordered pairs over the genes present in the ground truth, excluding self-loops.
    Pass nodes=set(gold genes) and allow_self=False (the default) for parity.
    """
    if nodes is None:
        nodes = (set(pred.source) | set(pred.target)) & (set(gold.source) | set(gold.target))
    p = restrict_to_nodes(pred, nodes)
    g = restrict_to_nodes(gold, nodes)

    candidates = _candidate_universe(nodes, allow_self=allow_self)
    gold_set = edge_set(g, directed)
    score_map = {edge_key(s, t, directed): sc
                 for s, t, sc in zip(p["source"], p["target"], p["score"])}

    y_true, y_score = [], []
    for a, b in candidates:
        key = (a, b) if directed else tuple(sorted((a, b)))
        y_true.append(1 if key in gold_set else 0)
        y_score.append(float(score_map.get(key, 0.0)))
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    n_gold = int(y_true.sum())
    n_cand = len(y_true)
    if n_gold == 0 or n_gold == n_cand:
        return {"auroc": np.nan, "auprc": np.nan, "epr": np.nan,
                "n_gold": n_gold, "n_candidates": n_cand}

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    k = n_gold
    top_k = np.argsort(-y_score)[:k]
    early_precision = y_true[top_k].mean()
    random_precision = n_gold / n_cand
    epr = early_precision / random_precision if random_precision else np.nan

    return {"auroc": auroc, "auprc": auprc, "epr": epr,
            "early_precision": early_precision,
            "n_gold": n_gold, "n_candidates": n_cand}


def run_beeline_comparison(
    method_edges: dict[str, pd.DataFrame],
    gold: pd.DataFrame,
    nodes: Optional[set[str]] = None,
    allow_self: bool = False,
) -> pd.DataFrame:
    """Fig. 7a/b-style table: one row per method, AUROC + EPR columns.

    Candidate universe defaults to the ground-truth gene set (BEELINE convention).
    For full parity with the BEELINE toolkit (AUPRC ratio, motif-scrambled controls,
    cell-count robustness in Fig. 7c/d), export these ranked edge lists and run
    github.com/Murali-group/Beeline; this gives the same core metrics inline.
    """
    if nodes is None:
        nodes = set(gold["source"]) | set(gold["target"])
    rows = {name: beeline_metrics(e, gold, nodes, allow_self=allow_self)
            for name, e in method_edges.items()}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Comparison 3: IN SILICO PERTURBATION  (stub)
# ---------------------------------------------------------------------------

def run_perturbation_comparison(*args, **kwargs):
    """TODO: compare predicted vs. observed responses to TF perturbation.

    Planned:
      - CellOracle: oracle.simulate_shift(perturb_condition={TF: 0}); read
        per-cell/per-cluster expression-shift vectors and transition probabilities.
      - boba-T: attractor landscape / perturbation results (see 9999/perturbations
        and 9999/attractors) -- predicted destabilisation / phenotype shift per TF KO.
      - Ground truth: observed KO/OE DE genes or phenotype (if available).
      - Metrics: direction agreement of shift vectors (cosine / sign), correlation of
        predicted vs. observed DE, rank agreement of "most impactful" TFs.
    """
    raise NotImplementedError("In silico perturbation comparison not implemented yet.")
