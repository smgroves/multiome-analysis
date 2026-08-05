"""Orchestration: assemble methods + a reference and run the comparisons."""

from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np
import pandas as pd

from .config import BEELINE_GSD, CFG, CO_INFERENCE_DIR
from .co_reproduction import run_co_reproduction_all
from .edges import _finalize_edges
from .loaders import (
    load_all_methods,
    load_beeline_network,
    load_beeline_ranked,
    load_ground_truth,
)
from .metrics import (
    _candidate_universe,
    pairwise_jaccard,
    run_beeline_comparison,
    run_structure_comparison,
)


def run_benchmarks(
    methods: dict[str, pd.DataFrame],
    reference: pd.DataFrame,
    out_dir: str,
    tag: str = "",
    nodes: Optional[set[str]] = None,
    allow_self: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run points 1 (structure) and 2 (BEELINE) for a set of methods vs. a reference.

    Candidate universe defaults to the reference's gene set (BEELINE convention).
    Writes structure_metrics{tag}.csv and beeline_metrics{tag}.csv to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    if nodes is None:
        nodes = set(reference["source"]) | set(reference["target"])
    print(f"[universe] {len(nodes)} reference genes; methods: {list(methods)}")

    structure = run_structure_comparison(methods, reference)
    structure.to_csv(os.path.join(out_dir, f"structure_metrics{tag}.csv"))
    print(f"\n== structure vs. reference{tag} ==\n", structure.round(3))

    beeline = run_beeline_comparison(methods, reference, nodes, allow_self=allow_self)
    beeline.to_csv(os.path.join(out_dir, f"beeline_metrics{tag}.csv"))
    print(f"\n== BEELINE AUROC/EPR{tag} ==\n", beeline.round(3))

    return {"structure": structure, "beeline": beeline}


def run_beeline_dataset(dataset_dir: str, gt_filename: str = "GroundTruthNetwork.csv"):
    """Score every algorithm's rankedEdges against a BEELINE ground-truth network.

    Expects <dataset_dir>/<gt_filename> plus rankedEdges CSVs found recursively
    (BEELINE writes outputs/<dataset>/<Algorithm>/rankedEdges.csv). Each folder name
    becomes the method label. Returns None (writes tables) if no outputs are found.
    """
    gt = load_beeline_network(os.path.join(dataset_dir, gt_filename))
    ranked_files = glob.glob(os.path.join(dataset_dir, "**", "*rankedEdges*.csv"),
                             recursive=True)
    if not ranked_files:
        print(f"[beeline] no rankedEdges files under {dataset_dir}; "
              "run BEELINE algorithms first, or use run_beeline_selftest().")
        return None
    methods = {}
    for f in ranked_files:
        algo = os.path.basename(os.path.dirname(f)) or os.path.basename(f)
        methods[algo] = load_beeline_ranked(f)
        print(f"[load] {algo}: {len(methods[algo])} edges")
    return run_benchmarks(methods, gt, CFG.out_dir, tag=f"_{os.path.basename(dataset_dir)}")


def run_beeline_selftest(dataset_dir: str = BEELINE_GSD, seed: int = 0):
    """Validate the harness end-to-end on real BEELINE ground-truth data.

    With no algorithm outputs on disk yet, score two reference baselines so the
    metrics are demonstrably sane:
      - 'perfect'  = the ground truth itself  -> AUROC 1.0, EPR = max, F1 1.0
      - 'random'   = random scores over all candidate pairs -> AUROC ~0.5, EPR ~1
    Real methods (CellOracle, boba-T, SCENIC, ...) slot in exactly where these do.
    """
    gt = load_beeline_network(os.path.join(dataset_dir, "GroundTruthNetwork.csv"))
    nodes = set(gt["source"]) | set(gt["target"])
    rng = np.random.default_rng(seed)

    # 'random' baseline: every candidate pair gets a random score.
    cand = _candidate_universe(nodes, allow_self=False)
    random_pred = pd.DataFrame(
        {"source": [a for a, _ in cand], "target": [b for _, b in cand],
         "score": rng.random(len(cand)), "weight": np.nan, "sign": 0}
    )

    methods = {"perfect(=GT)": gt, "random": _finalize_edges(random_pred)}
    print(f"[selftest] BEELINE GSD: {len(gt)} ground-truth edges over {len(nodes)} genes")
    return run_benchmarks(methods, gt, CFG.out_dir, tag="_gsd_selftest", nodes=nodes)


def run_methods_vs_reference(reference_source: str = "beeline",
                             reference_path: Optional[str] = None):
    """Load every registered method and score it against a reference network.

    Use once real method loaders (CellOracle, SCENIC, ...) are wired up. For the
    boba-T-vs-CellOracle comparison on your own data, set the reference to the
    DIRECT-NET base or a curated gold standard.
    """
    os.makedirs(CFG.out_dir, exist_ok=True)
    methods = load_all_methods()
    if not methods:
        print("No methods loaded; wire up at least one loader.")
        return

    pw = pairwise_jaccard(methods)
    pw.to_csv(os.path.join(CFG.out_dir, "pairwise_edge_jaccard.csv"))
    print("\n== pairwise edge Jaccard ==\n", pw.round(3))

    try:
        ref = load_ground_truth(reference_source, reference_path)
        run_benchmarks(methods, ref, CFG.out_dir)
    except NotImplementedError as e:
        print(f"\n[reference not set] {e}")

    # run_perturbation_comparison(...)  # comparison 4, later


def main():
    # Milestone 1: validate the harness on real BEELINE ground-truth data.
    run_beeline_selftest()

    # Milestone 1a: reproduce CellOracle's Fig. S2a/b (AUROC/EPR) from their released
    # inference_results, scored against ChIP-Atlas. Matches the paper to 3 decimals,
    # which validates the scoring before boba-T's own edges are dropped in as a method.
    if os.path.isdir(CO_INFERENCE_DIR):
        run_co_reproduction_all()
    else:
        print(f"[co-repro] {CO_INFERENCE_DIR} not found; extract method results first.")

    # Later: drop BEELINE algorithm outputs into data/beeline/<dataset>/ and call
    #   run_beeline_dataset(BEELINE_GSD)
    # or wire up method loaders and call
    #   run_methods_vs_reference("beeline")  /  ("celloracle_base", "<base_grn.parquet>")
