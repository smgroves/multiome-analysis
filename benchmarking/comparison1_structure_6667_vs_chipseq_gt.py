"""Comparison 1 (network structure, see benchmarking/README.md), real-ground-truth step:
score boba-T, CellOracle, and GENIE3's 6667 networks against three independent SCLC
ChIP-seq ground truths -- separately, not pooled, per the decision recorded in the README.

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
genie3_importance_matrix}.csv, written by comparison3_fit_celloracle_6667.py and
comparison_genie3_fit_6667.py respectively. Runs in either conda env (pandas/networkx only).
"""

import pandas as pd

from grn_benchmark.loaders import load_bobat, load_sclc_chipseq_gt
from grn_benchmark.edges import matrix_to_edges
from grn_benchmark.metrics import run_structure_comparison

DN = "../network-inference-DIRECT-NET"
GT_DIR = "data/sclc_chipseq_gt"
OUT_DIR = "benchmarking_out"

GROUND_TRUTHS = {
    "borromeo2016_ascl1_human": f"{GT_DIR}/borromeo2016_ascl1_human_chip.csv",
    "borromeo2016_ascl1_mouse": f"{GT_DIR}/borromeo2016_ascl1_mouse_chip.csv",
    "pozo2021_ascl1_direct": f"{GT_DIR}/pozo2021_ascl1_direct.csv",
}


def main():
    bobat_edges = load_bobat(run="6667")
    coef = pd.read_csv(f"{DN}/6667/rules/celloracle_coef_matrix.csv", index_col=0)
    celloracle_edges = matrix_to_edges(coef, regulators_on_columns=False)
    genie3_imp = pd.read_csv(f"{DN}/6667/rules/genie3_importance_matrix.csv", index_col=0)
    genie3_edges = matrix_to_edges(genie3_imp, regulators_on_columns=False)
    methods = {"boba-T": bobat_edges, "CellOracle": celloracle_edges, "GENIE3": genie3_edges}

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
