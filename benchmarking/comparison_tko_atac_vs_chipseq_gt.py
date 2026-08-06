"""Score the TKO (real-ATAC-informed) CellOracle and SCENIC networks against the same
three independent SCLC ChIP-seq ground truths used in comparison1_structure_6667_vs_chipseq_gt.py.
TKO_final_arc is a mouse RPR2 (Trp53;Rb1;Rbl2 triple-knockout) model, so
borromeo2016_ascl1_mouse_chip.csv (the actual RPR2-mouse ChIP-seq ground truth) is the most
directly relevant of the three, but all three are scored, unpooled, for the same reasons
recorded in the README.

Also includes boba-T's EXISTING fitted `6667` network (load_bobat, signed_strengths.csv) in
the same comparison -- NOT refit/rerun here, but this IS a same-dataset comparison: `6667`'s
underlying scRNA-seq is the same TKO_final_arc cells used for the CellOracle/SCENIC fits
above (DIRECT-NET's own ATAC-based candidate-network construction for `6667` was run
directly on this data outside this project's own scripts). The real difference between the
three methods here is candidate-network SCOPE, not dataset: boba-T's DIRECT-NET+LASSO
candidate network is a curated 53-gene panel, while CellOracle/SCENIC ran genome-scale on
the same cells' ~3,000-HVG set with an independently-built (Signac ClosestFeature +
motif-scan) ATAC-informed candidate structure -- not DIRECT-NET's.

Inputs:
    data/tko_full/celloracle_tko_atac_coef_matrix.csv        (comparison_tko_fit_celloracle_atac.py)
    data/tko_atac/scenic_tko_regulon_edges.csv                (comparison_tko_fit_scenic.py; proper
                                                                mouse-case gene symbols -- upper-cased
                                                                here to match the ground truths' and
                                                                CellOracle's upper-cased convention)
    network-inference-DIRECT-NET/6667/rules/signed_strengths.csv (boba-T's existing 6667 fit)

Runs in either conda env (pandas/networkx/sklearn only).
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
    "borromeo2016_ascl1_mouse_RPR2": f"{GT_DIR}/borromeo2016_ascl1_mouse_chip.csv",
    "pozo2021_ascl1_direct": f"{GT_DIR}/pozo2021_ascl1_direct.csv",
}


def load_scenic_tko_edges(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)  # source,target,weight in proper mouse case (e.g. Ascl1)
    df["source"] = df["source"].str.upper()
    df["target"] = df["target"].str.upper()
    df["sign"] = 0  # regulon weights are unsigned correlation-derived importances
    return _finalize_edges(df[["source", "target", "weight", "sign"]])


def main():
    coef = pd.read_csv(f"data/tko_full/celloracle_tko_atac_coef_matrix.csv", index_col=0)
    celloracle_edges = matrix_to_edges(coef, regulators_on_columns=False)  # rows=regulators, cols=targets

    scenic_edges = load_scenic_tko_edges(f"{DN}/data/tko_atac/scenic_tko_regulon_edges.csv")

    bobat_edges = load_bobat(run="6667")  # existing fit, NOT rerun/refit on TKO -- see docstring

    methods = {
        "CellOracle (real ATAC, TKO)": celloracle_edges,
        "SCENIC (from scratch, TKO)": scenic_edges,
        "boba-T (existing 6667 fit, NOT TKO)": bobat_edges,
    }
    for name, e in methods.items():
        print(f"[load] {name}: {len(e)} edges, {e.source.nunique()} sources, {e.target.nunique()} targets")

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, path in GROUND_TRUTHS.items():
        gt = load_sclc_chipseq_gt(path)
        result = run_structure_comparison(methods, gt)
        out_path = f"{OUT_DIR}/comparison_tko_atac_vs_{name}.csv"
        result.to_csv(out_path)
        print(f"\n== {name} ({len(gt)} ASCL1-> edges) ==")
        print(result)
        print(f"-> {out_path}")


if __name__ == "__main__":
    main()
