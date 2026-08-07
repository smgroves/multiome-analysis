"""Domain-shift diagnostic: is organoid/mets_compiled's external-validation shortfall (vs.
the in-vivo TKO GEMM allografts 6667 was originally validated on) explained by cell-culture
condition/stress rather than by BoBa-T's rules failing to generalize? Per gene, join each
accuracy_plots/<gene>_validation.csv's |predicted - actual| residual (already written by
fit_validation, keyed by CellID) to that dataset's own precomputed per-cell state/archetype
scores (stress_score1/predicted.id for organoid, Stress_score1/seurat_clusters for
mets_compiled), and check whether residual concentrates in high-stress cells/clusters
rather than being spread uniformly across the population -- the signature you'd expect if
culture-stress or a specific archetype, not the fitted rule itself, is driving the error.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/03_domain_shift_diagnostics/diagnose_domain_shift_6667.py
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
BRCD = "6667"
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic_low_R2_samples"

DATASETS = {
    "organoid_shGFP": {
        "cluster_csv": "data/organoid/organoid_clusters.csv",
        "stress_col": "stress_score1",
        "group_col": "predicted.id",
    },
    "organoid_shRORB1": {
        "cluster_csv": "data/organoid/organoid_clusters.csv",
        "stress_col": "stress_score1",
        "group_col": "predicted.id",
    },
    "organoid_shRORB2": {
        "cluster_csv": "data/organoid/organoid_clusters.csv",
        "stress_col": "stress_score1",
        "group_col": "predicted.id",
    },
    "mets_compiled": {
        "cluster_csv": "data/mets_compiled/mets_compiled_clusters.csv",
        "stress_col": "Stress_score1",
        "group_col": "seurat_clusters",
    },
}

# Known stress-response / cell-adhesion biology for the worst-validating genes found in each
# dataset's summary_stats.csv -- concrete, defensible domain-shift candidates on biological
# grounds alone (culture/attachment context differs sharply between a 3D organoid or a
# compiled metastasis atlas and an in-vivo GEMM tumor), independent of any correlation found
# below.
GENE_BIOLOGY_NOTES = {
    "FOXO3": "Canonical oxidative/cellular-stress-response TF (FOXO pathway) -- directly stress-sensitive.",
    "NCAM1": "Cell-adhesion molecule (NCAM); adhesion/ECM context differs sharply organoid/compiled-atlas vs. in-vivo GEMM.",
    "ICAM1": "Cell-adhesion molecule (ICAM); same culture/attachment-context sensitivity as NCAM1.",
    "GRHL2": "Epithelial-identity TF, sensitive to ECM/substrate and epithelial-mesenchymal balance shifts in culture.",
    "ETS1": "Stress/signaling-responsive TF, downstream of MAPK signaling that is culture-condition sensitive.",
    "TBX15": "Developmental/mesenchymal TF; identity may be less stable outside the native tumor microenvironment.",
    "KMT2A": "Chromatin regulator; broadly expressed, less an obvious stress-biology story than the others here.",
    "EHF": "Epithelial ETS-family TF, ECM/substrate-context sensitive like GRHL2.",
    "THRB": "Thyroid hormone receptor; hormone/media-composition sensitive, a plausible culture-condition confound.",
    "LMX1B": "Developmental TF; identity-defining, may not be under the same control outside the native tumor.",
    "TCF7L1": "Wnt-pathway TF; Wnt signaling is well known to be culture-substrate/media sensitive.",
    "HES1": "Notch-pathway TF; Notch signaling is contact/density dependent, differs in organoid vs. in-vivo geometry.",
    "NFIB": "Metastasis-associated TF; regulation may differ outside the in-vivo metastatic niche.",
    "EPCAM": "Epithelial-adhesion marker, same culture/attachment-context sensitivity as NCAM1/ICAM1.",
    "ASCL1": "Core NE-lineage TF; not itself an obvious stress/adhesion story, included as it validates poorly anyway.",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gene_summary_rows = []
    percell_rows = []

    for dataset, cfg in DATASETS.items():
        val_dir = f"{DIR_PREFIX}/{BRCD}/validation/external_validation/{dataset}"
        summary = pd.read_csv(f"{val_dir}/summary_stats.csv", index_col=0)
        clusters = pd.read_csv(f"{DIR_PREFIX}/{cfg['cluster_csv']}", index_col=0)

        worst_genes = summary.sort_values("r2").head(10)["gene"].tolist()
        print(f"\n=== {dataset}: worst 10 genes by r2 ===")
        print(summary.set_index("gene").loc[worst_genes, "r2"])

        for gene in worst_genes:
            acc_path = f"{val_dir}/accuracy_plots/{gene}_validation.csv"
            if not os.path.exists(acc_path):
                continue
            val_df = pd.read_csv(acc_path, index_col=0)
            val_df["residual"] = (val_df["predicted"] - val_df["actual"]).abs()
            joined = val_df.join(clusters[[cfg["stress_col"], cfg["group_col"]]], how="inner")
            joined = joined.dropna(subset=[cfg["stress_col"], "residual"])

            if len(joined) > 5 and joined[cfg["stress_col"]].std() > 0:
                r, p = pearsonr(joined[cfg["stress_col"]], joined["residual"])
            else:
                r, p = np.nan, np.nan

            by_group = joined.groupby(cfg["group_col"])["residual"].mean().sort_values(ascending=False)

            gene_summary_rows.append({
                "dataset": dataset, "gene": gene, "r2": summary.set_index("gene").loc[gene, "r2"],
                "n_cells": len(joined), "residual_vs_stress_pearson_r": r, "residual_vs_stress_p": p,
                "highest_residual_group": by_group.index[0] if len(by_group) else None,
                "highest_residual_group_mean": by_group.iloc[0] if len(by_group) else None,
                "lowest_residual_group": by_group.index[-1] if len(by_group) else None,
                "lowest_residual_group_mean": by_group.iloc[-1] if len(by_group) else None,
                "biology_note": GENE_BIOLOGY_NOTES.get(gene, ""),
            })

            for _, row in joined.iterrows():
                percell_rows.append({
                    "dataset": dataset, "gene": gene, "residual": row["residual"],
                    "stress_score": row[cfg["stress_col"]], "group": row[cfg["group_col"]],
                })

    gene_summary = pd.DataFrame(gene_summary_rows)
    gene_summary.to_csv(f"{OUT_DIR}/worst_genes_stress_correlation.csv", index=False)
    pd.DataFrame(percell_rows).to_csv(f"{OUT_DIR}/percell_residual_vs_stress.csv", index=False)

    print("\n=== Summary: residual-vs-stress correlation across worst genes ===")
    print(gene_summary[["dataset", "gene", "r2", "residual_vs_stress_pearson_r", "residual_vs_stress_p"]]
          .to_string(index=False))

    n_significant = (gene_summary["residual_vs_stress_p"] < 0.05).sum()
    n_positive_significant = ((gene_summary["residual_vs_stress_p"] < 0.05) &
                               (gene_summary["residual_vs_stress_pearson_r"] > 0)).sum()
    print(f"\n{n_significant}/{len(gene_summary)} worst-gene x dataset pairs have a significant "
          f"(p<0.05) residual-vs-stress correlation; {n_positive_significant} of those are positive "
          f"(higher stress -> higher error, the domain-shift signature).")

    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
