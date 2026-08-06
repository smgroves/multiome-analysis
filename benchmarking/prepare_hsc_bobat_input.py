"""Convert BoolODE's HSC simulation output into boba-T's expected input format: CellID
rows x gene columns, gene symbols upper-cased to match hsc_ground_truth.py's candidate
network, values min-max normalized to [0,1] (BoolODE's Hill-equation concentrations are on
an arbitrary 0-20ish scale, not boba-T's usual [0,1] convention).

Run after data/BoolODE/output-HSC/HSC/ExpressionData.csv exists (see
config-files/hsc-config.yaml -- do_parallel: False; True hangs on macOS with this
2019-era codebase, see benchmarking/README.md).
"""

import pandas as pd

BOOLODE_OUT = "data/BoolODE/output-HSC/HSC"
OUT_DIR = "data/hsc_ground_truth"


def main():
    expr = pd.read_csv(f"{BOOLODE_OUT}/ExpressionData.csv", index_col=0)  # genes x cells
    expr = expr.T  # -> cells x genes
    expr.columns = [g.upper() for g in expr.columns]
    expr.index.name = "CellID"

    # BoolODE's Hill-equation concentrations are on an arbitrary ~0-20 scale, not the
    # [0,1] convention boba-T's own node_normalization expects -- min-max per gene.
    normalized = (expr - expr.min()) / (expr.max() - expr.min())

    normalized.to_csv(f"{OUT_DIR}/expr_bobat.csv")
    print(f"{normalized.shape[0]} cells x {normalized.shape[1]} genes -> {OUT_DIR}/expr_bobat.csv")

    # No real phenotype clusters for this model (single steady-state-approaching system,
    # nClusters=1 in the BoolODE config) -- a single constant "class" satisfies boba-T's
    # cellID_table/cluster_header_list requirement without implying clusters that aren't there.
    clusters = pd.DataFrame({"class": ["all"] * len(normalized)}, index=normalized.index)
    clusters.index.name = "CellID"
    clusters.to_csv(f"{OUT_DIR}/clusters_bobat.csv")


if __name__ == "__main__":
    main()
