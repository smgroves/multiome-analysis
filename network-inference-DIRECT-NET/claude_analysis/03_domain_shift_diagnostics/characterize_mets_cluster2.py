"""Characterize mets_compiled's seurat_cluster 2 -- the subpopulation found in
diagnose_domain_shift_6667.py to carry the highest validation residual for 7/10 of
mets_compiled's worst-validating genes. Two questions:

1. What distinguishes cluster 2 biologically/technically from the rest of mets_compiled?
   (differential expression across the full preprocessed gene panel + the dataset's own
   precomputed archetype scores)
2. Is cluster 2 an "unseen" cell state relative to the 8 GEMM attractors 6667 already
   characterizes (Generalist_NE/nonNE, Arc_1-6, in 6667/attractors/average_states.txt) --
   relevant to BoBa-T's claim that Boolean structure + attractor basins let it predict
   dynamics in unseen clusters without a full dataset to fit on -- or is the deviation
   better explained by a technical confound the model was never expected to predict?

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/03_domain_shift_diagnostics/characterize_mets_cluster2.py
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic"
CLUSTER_OF_INTEREST = 2

WORST_GENES = ["FOXO3", "RORA_RORB", "GRHL2", "ETS1", "NCAM1", "TBX15", "KMT2A", "ICAM1", "NFKB1", "SMAD3"]
IEG_COLS = ["EGR1", "FOS", "JUN", "FOSB", "JUND", "JUNB"]  # canonical dissociation/acute-stress immediate-early genes
SCORE_COLS = ["Int_score1", "Stress_score1", "NE1_score1", "NE2_score1", "nonNE1_score1", "nonNE2_score1"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = pd.read_csv(f"{DIR_PREFIX}/data/mets_compiled/adata_mets_compiled_v3_RORA_RORB_ave.csv", index_col=0)
    clusters = pd.read_csv(f"{DIR_PREFIX}/data/mets_compiled/mets_compiled_clusters.csv", index_col=0)
    joined = data.join(clusters[["seurat_clusters"] + SCORE_COLS], how="inner")

    ieg_cols = [c for c in IEG_COLS if c in data.columns]
    joined["ieg_score"] = joined[ieg_cols].mean(axis=1)

    c2 = joined[joined["seurat_clusters"] == CLUSTER_OF_INTEREST]
    rest = joined[joined["seurat_clusters"] != CLUSTER_OF_INTEREST]
    print(f"cluster {CLUSTER_OF_INTEREST}: {len(c2)} cells; rest: {len(rest)} cells")

    print("\n=== Archetype score means: cluster2 vs rest ===")
    for col in SCORE_COLS + ["ieg_score"]:
        print(f"{col}: cluster2={c2[col].mean():.3f}  rest={rest[col].mean():.3f}  diff={c2[col].mean() - rest[col].mean():+.3f}")

    # Full differential-expression pass across every gene in the panel (not just network genes)
    diffs = []
    for g in data.columns:
        m2, mrest = c2[g].mean(), rest[g].mean()
        stat, p = mannwhitneyu(c2[g], rest[g])
        diffs.append((g, m2, mrest, m2 - mrest, p))
    diffs_df = pd.DataFrame(diffs, columns=["gene", "mean_cluster2", "mean_rest", "diff", "pval"]).sort_values("diff")
    diffs_df.to_csv(f"{OUT_DIR}/mets_cluster2_differential_expression.csv", index=False)

    print("\n=== Top 10 genes LOWER in cluster2 (whole panel) ===")
    print(diffs_df.head(10).to_string(index=False))
    print("\n=== Top 10 genes HIGHER in cluster2 (whole panel) ===")
    print(diffs_df.tail(10).sort_values("diff", ascending=False).to_string(index=False))

    print(f"\n=== The 10 worst-validating network genes' differential expression in cluster2 ===")
    print(diffs_df[diffs_df["gene"].isin(WORST_GENES)].to_string(index=False))

    # Question 2: is cluster2 "unseen" relative to the known GEMM attractors, or just
    # displaced from its own true identity by the stress/IEG program on top of it?
    attractors = pd.read_csv(f"{DIR_PREFIX}/6667/attractors/average_states.txt", index_col=0)
    network_genes = [g for g in attractors.columns if g in data.columns]

    mean_c2 = c2[network_genes].mean()
    mean_rest = rest[network_genes].mean()
    print(f"\n=== Euclidean distance from mean continuous state (network genes only) to each known GEMM attractor ===")
    dist_rows = []
    for arc, row in attractors[network_genes].iterrows():
        d_c2 = np.sqrt(((mean_c2 - row) ** 2).sum())
        d_rest = np.sqrt(((mean_rest - row) ** 2).sum())
        dist_rows.append((arc, d_c2, d_rest))
    dist_df = pd.DataFrame(dist_rows, columns=["attractor", "dist_cluster2", "dist_rest"]).sort_values("dist_cluster2")
    print(dist_df.to_string(index=False))
    dist_df.to_csv(f"{OUT_DIR}/mets_cluster2_attractor_distances.csv", index=False)

    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
