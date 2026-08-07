"""organoid_shGFP is a pool of 5 distinct organoid lines (data/organoid/organoid_clusters.csv
`sample` column, restricted to `condition == "shGFP"`: D_Sample1/4/6/9/12), not one
homogeneous sample. This re-scores R^2/F1/AUC separately per line to check whether the
pooled organoid_shGFP score (the lowest of the 3 organoid samples in the main category
plot) is driven by one bad line, or is a property shared across all 5.

Reuses the EXACT per-cell predictions already on disk from the original organoid_shGFP
validation run (6667/validation/external_validation/organoid_shGFP/accuracy_plots/
{gene}_validation.csv -- one row per cell, columns CellID/predicted/actual) rather than
re-running fit_validation: those files retain each cell's own ID, so they can be split by
line and re-scored with the identical metric definitions bobaT's own
get_sklearn_metrics() uses (binary threshold 0.5 for classification stats; r2/AUC use the
continuous `predicted` against the binarized `actual`) -- no re-normalization or re-fit
needed, and this is exactly equivalent to what a per-line fit_validation() run would have
produced, since parent_heatmap's prediction for one cell never depends on any other cell.

SELF-LOOPS ARE INCLUDED, NOT DROPPED: all 53 network genes are scored per line, including
the 11 that are their own regulator (TFDP1, NFYC, CREB1, TCF4, ZEB1, ESR1, STAT1, RBPJ,
JUND, NR6A1, SOX9), matching the original organoid_shGFP validation run and the production
pipeline (remove_selfloops=False everywhere). Self-loop genes trivially score high even
under a fully scrambled null (see run_scrambled_null_validation.py) since a gene's own
value partially predicts itself through the self-referential regulator link -- worth
keeping in mind since it affects every line here equally, not just one.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/03_domain_shift_diagnostics/plot_organoid_shgfp_by_line.py
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score, roc_auc_score

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
ACC_DIR = f"{DIR_PREFIX}/6667/validation/external_validation/organoid_shGFP/accuracy_plots"
CLUSTERS_CSV = f"{DIR_PREFIX}/data/organoid/organoid_clusters.csv"
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic_and_organoid_walks"

LINE_COLORS = {
    "D_Sample1": "tab:blue", "D_Sample4": "tab:orange", "D_Sample6": "tab:green",
    "D_Sample9": "tab:red", "D_Sample12": "tab:purple",
}


def per_gene_metrics(val_df):
    actual_binary = (val_df["actual"] > 0.5).astype(int)
    predicted_binary = (val_df["predicted"] > 0.5).astype(int)
    r2 = r2_score(val_df["actual"], val_df["predicted"])
    f1 = f1_score(actual_binary, predicted_binary)
    try:
        auc = roc_auc_score(actual_binary, val_df["predicted"])
    except ValueError:
        auc = np.nan
    return r2, f1, auc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    clusters = pd.read_csv(CLUSTERS_CSV, index_col=0)
    shgfp = clusters[clusters["condition"] == "shGFP"]
    line_of_cell = shgfp["sample"]
    print(f"organoid_shGFP lines: {line_of_cell.value_counts().to_dict()}")

    gene_files = sorted(glob.glob(f"{ACC_DIR}/*_validation.csv"))
    print(f"Found {len(gene_files)} per-gene validation files")

    rows = []
    for f in gene_files:
        gene = os.path.basename(f).removesuffix("_validation.csv")
        val_df = pd.read_csv(f, index_col=0)
        val_df["line"] = line_of_cell.reindex(val_df.index)
        for line, sub in val_df.groupby("line"):
            r2, f1, auc = per_gene_metrics(sub)
            rows.append({"line": line, "gene": gene, "r2": r2, "f1": f1, "roc_auc_score": auc, "n_cells": len(sub)})
        # pooled (== the original organoid_shGFP row) as a reference check
        r2, f1, auc = per_gene_metrics(val_df)
        rows.append({"line": "organoid_shGFP (pooled, all lines)", "gene": gene, "r2": r2, "f1": f1, "roc_auc_score": auc, "n_cells": len(val_df)})

    per_gene = pd.DataFrame(rows)
    summary = per_gene.groupby("line").agg(
        mean_r2=("r2", "mean"), mean_f1=("f1", "mean"), mean_auc=("roc_auc_score", "mean"),
        n_cells=("n_cells", "first"),
    ).reset_index()
    summary.to_csv(f"{OUT_DIR}/organoid_shgfp_by_line_metrics.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {OUT_DIR}/organoid_shgfp_by_line_metrics.csv")

    lines = list(LINE_COLORS.keys())
    pooled = summary[summary["line"] == "organoid_shGFP (pooled, all lines)"].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, label in zip(
        axes, ["mean_r2", "mean_f1", "mean_auc"],
        ["Mean R²", "Mean F1", "Mean ROC AUC"],
    ):
        vals = [summary.loc[summary["line"] == ln, col].iloc[0] for ln in lines]
        colors = [LINE_COLORS[ln] for ln in lines]
        ns = [int(summary.loc[summary["line"] == ln, "n_cells"].iloc[0]) for ln in lines]
        bars = ax.bar(range(len(lines)), vals, color=colors, edgecolor="black", linewidth=0.7, zorder=3)
        ax.axhline(pooled[col], color="0.3", linewidth=1.5, linestyle="--", zorder=2)
        ax.text(len(lines) - 0.5, pooled[col], f"pooled={pooled[col]:.2f}", color="0.3",
                fontsize=8, ha="right", va="bottom" if pooled[col] >= 0 else "top", fontweight="bold")
        for i, (v, n) in enumerate(zip(vals, ns)):
            va = "bottom" if v >= 0 else "top"
            offset = 0.02 * (max(vals) - min(min(vals), 0)) * (1 if v >= 0 else -1)
            ax.text(i, v + offset, f"{v:.2f}\n(n={n})", ha="center", va=va, fontsize=8)
        ax.set_xticks(range(len(lines)))
        ax.set_xticklabels(lines, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "organoid_shGFP split by line -- is one line pulling down the pooled score?\n"
        "Same fitted rules (run 6667), each line's own cells re-scored separately",
        fontsize=12,
    )
    note = (
        "Each line's own predicted-vs-actual pairs (from the original organoid_shGFP validation run) re-scored independently, across\n"
        "all 53 network genes including the 11 self-loop genes (equally present in every line here). Dashed line = the original\n"
        "pooled organoid_shGFP score across all 5 lines together."
    )
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))

    for ext in ["png", "pdf"]:
        fig.savefig(f"{OUT_DIR}/organoid_shgfp_by_line.{ext}", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/organoid_shgfp_by_line.{{png,pdf}}")


if __name__ == "__main__":
    main()
