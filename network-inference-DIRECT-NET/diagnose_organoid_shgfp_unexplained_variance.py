"""§7 found organoid_shGFP's variance sits mostly on neither GEMM's identity axes (0.21)
nor the IEG axis (0.005) -- ~78% is on some uncharacterized axis. This fits PCA on
organoid_shGFP's OWN raw data (its own dominant axes of variation, not GEMM's basis) and
correlates each top PC's per-cell scores against every available metadata column
(archetype/prediction scores, technical depth, batch/well identifiers) to identify what
that axis actually is.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python diagnose_organoid_shgfp_unexplained_variance.py
"""

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd
from sklearn.decomposition import PCA

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic"
N_PCS = 6

NUMERIC_META_COLS = [
    "nCount_RNA", "nFeature_RNA", "gene_count", "tscp_count", "mread_count", "percent.mt",
    "prediction.score.Generalist.NE", "prediction.score.Neuroendocrine2", "prediction.score.Stress",
    "prediction.score.Generalist.nonNE", "prediction.score.nonNE1", "prediction.score.Neuroendocrine1",
    "prediction.score.Intermediate", "prediction.score.nonNE2", "prediction.score.max",
    "Ascl1_targets_score1", "Rest_targets_score1", "HES1_high_score1", "Liu_sclcI_score1",
    "Intermediate_score1", "NE1_score1", "NE2_score1", "nonNE1_score1", "nonNE2_score1", "stress_score1",
]
CATEGORICAL_META_COLS = ["predicted.id", "seurat_clusters", "tree.ident", "rcpa_clusters", "orig.ident"]


def main():
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

    raw = bb.load.load_data(
        f"{DIR_PREFIX}/data/organoid/adata_organoid_shGFP_v3_RORA_RORB_ave.csv", nodes,
        norm=None, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    clusters = pd.read_csv(f"{DIR_PREFIX}/data/organoid/organoid_clusters.csv", index_col=0)
    clusters = clusters.reindex(raw.index)

    pca = PCA(n_components=N_PCS, random_state=1234)
    scores = pca.fit_transform(raw.values)
    print(f"organoid_shGFP's own top {N_PCS} PCs explain {pca.explained_variance_ratio_.sum():.3f} of its own variance")
    print(f"Per-PC explained variance ratio: {np.round(pca.explained_variance_ratio_, 3)}")

    loadings = pd.DataFrame(pca.components_, columns=nodes, index=[f"PC{i+1}" for i in range(N_PCS)])
    print("\nTop +/- loading genes per PC:")
    for pc in loadings.index:
        top = loadings.loc[pc].sort_values()
        print(f"  {pc}: low=({top.index[0]}:{top.iloc[0]:.2f}, {top.index[1]}:{top.iloc[1]:.2f})  "
              f"high=({top.index[-1]}:{top.iloc[-1]:.2f}, {top.index[-2]}:{top.iloc[-2]:.2f})")

    scores_df = pd.DataFrame(scores, columns=loadings.index, index=raw.index)

    print("\n=== Correlation of each PC's per-cell score with numeric metadata ===")
    numeric_present = [c for c in NUMERIC_META_COLS if c in clusters.columns]
    corr_rows = []
    for pc in scores_df.columns:
        for col in numeric_present:
            valid = clusters[col].notna()
            if valid.sum() < 10:
                continue
            r = np.corrcoef(scores_df.loc[valid, pc], clusters.loc[valid, col])[0, 1]
            corr_rows.append({"PC": pc, "metadata_col": col, "corr": r})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(f"{OUT_DIR}/organoid_shgfp_own_pc_metadata_correlation.csv", index=False)

    for pc in scores_df.columns:
        sub = corr_df[corr_df["PC"] == pc].sort_values("corr", key=abs, ascending=False)
        print(f"\n{pc} (explains {pca.explained_variance_ratio_[int(pc[2:]) - 1]:.3f} of variance) -- top metadata correlations:")
        print(sub.head(5).to_string(index=False))

    print("\n=== Per-PC variance explained by categorical metadata (eta-squared: between-group / total variance) ===")
    cat_rows = []
    for pc in scores_df.columns:
        for col in CATEGORICAL_META_COLS:
            if col not in clusters.columns:
                continue
            valid = clusters[col].notna()
            if valid.sum() < 10:
                continue
            groups = clusters.loc[valid, col]
            vals = scores_df.loc[valid, pc]
            grand_mean = vals.mean()
            total_ss = ((vals - grand_mean) ** 2).sum()
            between_ss = sum(
                len(vals[groups == g]) * (vals[groups == g].mean() - grand_mean) ** 2
                for g in groups.unique() if (groups == g).sum() > 0
            )
            eta_sq = between_ss / total_ss if total_ss > 0 else np.nan
            cat_rows.append({"PC": pc, "metadata_col": col, "eta_squared": eta_sq, "n_groups": groups.nunique()})
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(f"{OUT_DIR}/organoid_shgfp_own_pc_categorical_variance.csv", index=False)
    pd.set_option("display.width", 160)
    print(cat_df.sort_values(["PC", "eta_squared"], ascending=[True, False]).to_string(index=False))

    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
