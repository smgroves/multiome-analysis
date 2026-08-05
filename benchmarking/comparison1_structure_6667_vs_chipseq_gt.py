"""Comparison 1 (network structure, see benchmarking/README.md), real-ground-truth step:
score five 6667 networks -- boba-T, two CellOracle variants, GENIE3, and SCENIC -- against
three independent SCLC ChIP-seq ground truths, separately, not pooled, per the decision
recorded in the README.

Two different treatments are compared side by side here:
    - "CellOracle (DIRECT-NET base)" / "GENIE3 (DIRECT-NET-restricted)": both fit with
      their candidate regulators restricted to the same 228-edge DIRECT-NET network boba-T
      uses (comparison3_fit_celloracle_6667.py / comparison_genie3_fit_6667.py) -- isolates
      rule-fitting/pruning quality on a shared candidate set.
    - "CellOracle (from scratch)" / "SCENIC (from scratch)": both given ALL of 6667's
      cells and left to find their own structure -- CellOracle via its own human promoter
      base GRN (comparison_celloracle_fromscratch_fit_6667.py), SCENIC via full
      GRNBoost2 + RcisTarget motif pruning (comparison_scenic_fit_6667.py) -- neither
      ever sees the DIRECT-NET network at all.

Ground truths (see README's "Sourcing a real SCLC ground truth" for provenance):
    data/sclc_chipseq_gt/borromeo2016_ascl1_human_chip.csv  -- Borromeo et al. 2016 Table
        S9, ASCL1 ChIP-bound targets in human SCLC cell lines/tumors (620 edges, unsigned).
    data/sclc_chipseq_gt/borromeo2016_ascl1_mouse_chip.csv  -- Borromeo et al. 2016 Table
        S8, ASCL1 ChIP-seq peaks (GREAT-assigned genes) in the RPR2 mouse SCLC tumor
        (Trp53;Rb1;Rbl2), mouse gene symbols upper-cased to match human convention for
        matching (3,992 edges, unsigned). This is the actual RPR2-mouse ground truth.
    data/sclc_chipseq_gt/pozo2021_ascl1_direct.csv  -- Pozo et al. 2021 Table S6, ASCL1
        direct targets by ChIP + siASCL1-knockdown RNA-seq in NCI-H2107 (295 edges, signed).

All three are ASCL1-only (single source TF); two human, one RPR2-mouse -- see README for
the correction history (S9 was originally, incorrectly, thought to be the mouse data).

Requires network-inference-DIRECT-NET/6667/rules/{celloracle_coef_matrix,
celloracle_fromscratch_coef_matrix, genie3_importance_matrix, scenic_regulon_edges}.csv --
see each fitting script's own docstring for how to produce it. Runs in either conda env
(pandas/networkx only).
"""

import pandas as pd

from grn_benchmark.edges import _finalize_edges, matrix_to_edges
from grn_benchmark.loaders import load_bobat, load_sclc_chipseq_gt
from grn_benchmark.metrics import run_structure_comparison

DN = "../network-inference-DIRECT-NET"
GT_DIR = "data/sclc_chipseq_gt"
OUT_DIR = "benchmarking_out"

GROUND_TRUTHS = {
    "borromeo2016_ascl1_human": f"{GT_DIR}/borromeo2016_ascl1_human_chip.csv",
    "borromeo2016_ascl1_mouse": f"{GT_DIR}/borromeo2016_ascl1_mouse_chip.csv",
    "pozo2021_ascl1_direct": f"{GT_DIR}/pozo2021_ascl1_direct.csv",
}


def load_scenic_edges(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)  # already source,target,weight
    df["sign"] = 0  # regulon weights are unsigned correlation-derived importances
    return _finalize_edges(df[["source", "target", "weight", "sign"]])


def main():
    bobat_edges = load_bobat(run="6667")

    coef = pd.read_csv(f"{DN}/6667/rules/celloracle_coef_matrix.csv", index_col=0)
    celloracle_edges = matrix_to_edges(coef, regulators_on_columns=False)

    coef_scratch = pd.read_csv(f"{DN}/6667/rules/celloracle_fromscratch_coef_matrix.csv", index_col=0)
    celloracle_scratch_edges = matrix_to_edges(coef_scratch, regulators_on_columns=False)

    genie3_imp = pd.read_csv(f"{DN}/6667/rules/genie3_importance_matrix.csv", index_col=0)
    genie3_edges = matrix_to_edges(genie3_imp, regulators_on_columns=False)

    scenic_edges = load_scenic_edges(f"{DN}/6667/rules/scenic_regulon_edges.csv")

    methods = {
        "boba-T": bobat_edges,
        "CellOracle (DIRECT-NET base)": celloracle_edges,
        "CellOracle (from scratch)": celloracle_scratch_edges,
        "GENIE3 (DIRECT-NET-restricted)": genie3_edges,
        "SCENIC (from scratch)": scenic_edges,
    }

    for name, path in GROUND_TRUTHS.items():
        gt = load_sclc_chipseq_gt(path)
        result = run_structure_comparison(methods, gt)
        out_path = f"{OUT_DIR}/comparison1_structure_6667_vs_{name}.csv"
        result.to_csv(out_path)
        print(f"\n== {name} ({len(gt)} ASCL1-> edges) ==")
        print(result)
        print(f"-> {out_path}")


if __name__ == "__main__":
    main()
