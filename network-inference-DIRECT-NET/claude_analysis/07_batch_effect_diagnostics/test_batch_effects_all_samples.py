"""Direct test: is poor external-validation R2 explained by technical BATCH effects (each
dataset's own sequencing platform/lab/processing pipeline forming its own separate cluster
in expression space, independent of real biology) rather than -- or in addition to -- the
genuine cross-context rewiring already established for organoid_shGFP
(domain_shift_diagnostic_and_organoid_walks/FINDINGS.md sec 9)?

Method (a kBET/LISI-style batch-mixing test, no extra package needed): pool GEMM training
cells with every external sample's cells (raw, comparable log1p+MAGIC-imputed units --
reuses umap_archetype_distance_analysis.py's exact pooling/subsampling logic and its 33
existing samples, extended with the 5 new Ireland et al. 2025 datasets), z-score scale,
PCA to top 20 PCs, build a shared kNN graph (k=30). For each cell, compute the fraction of
its k nearest neighbors that share its own dataset label ("observed same-dataset
fraction"). Compare against the "expected" same-dataset fraction under random mixing (that
dataset's share of the total pooled cell count, after the same per-dataset subsample cap
used everywhere else in this diagnostic series, so no single huge dataset dominates the
denominator). A per-dataset "batch separation score" = mean(observed) / expected: ~1 means
well-mixed with GEMM/other datasets (no batch effect), >>1 means cells preferentially
neighbor their own dataset regardless of biology (a real batch effect).

This is then correlated against each dataset's own mean R2 (already-scored samples only):
if batchier datasets validate worse, batch effects are a real contributor worth correcting
for; if not, it argues (consistent with the existing leaf-conditional-rewiring finding) that
technical batch separation isn't the bottleneck even where it's present.

Run in bobaT_env_py3.13 (needs `umap-learn`/sklearn, already present; UMAP itself isn't used
here -- PCA + kNN is enough for a mixing test and avoids UMAP's known distance distortion):
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/07_batch_effect_diagnostics/test_batch_effects_all_samples.py
"""

import glob
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/claude_analysis/07_batch_effect_diagnostics"
MAX_CELLS_PER_SAMPLE = 1500
RANDOM_STATE = 1234
KNN_K = 30
N_PCS = 20

IRELAND_2025_SAMPLES = [
    "cgrp_k5", "organoid_celltag", "tbo_allograft_5khvg", "rpr2_allograft", "celltag_fate_dpt",
]


def gather_sample_paths():
    paths = {}
    for f in glob.glob(f"{DIR_PREFIX}/data/allografts/adata_*_v3_RORA_RORB_ave.csv"):
        name = re.search(r"adata_(.+)_allografts_v3", f).group(1)
        paths[f"allograft_{name}"] = f
    for f in glob.glob(f"{DIR_PREFIX}/data/human_tumor_MSK/adata_*_v3_RORA_RORB_ave.csv"):
        name = re.search(r"adata_(.+)_v3", f).group(1)
        paths[f"human_{name}"] = f
    paths["organoid_shGFP"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_shGFP_v3_RORA_RORB_ave.csv"
    paths["organoid_shRORB1"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_shRORB1_v3_RORA_RORB_ave.csv"
    paths["organoid_shRORB2"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_shRORB2_v3_RORA_RORB_ave.csv"
    paths["organoid_combined"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_v3_RORA_RORB_ave.csv"
    paths["mets_compiled"] = f"{DIR_PREFIX}/data/mets_compiled/adata_mets_compiled_v3_RORA_RORB_ave.csv"
    for name in IRELAND_2025_SAMPLES:
        paths[name] = f"{DIR_PREFIX}/data/{name}/adata_{name}_v3_RORA_RORB_ave.csv"
    return paths


def _summary_stats_candidates(name):
    if name.startswith("allograft_"):
        return [f"{DIR_PREFIX}/6667/validation/allografts/{name[len('allograft_'):]}/summary_stats.csv"]
    if name.startswith("human_"):
        return [f"{DIR_PREFIX}/6667/validation/human_tumor_MSK/{name[len('human_'):]}/summary_stats.csv"]
    if name == "organoid_combined":
        return [f"{DIR_PREFIX}/6667/validation/external_validation/organoid/summary_stats.csv"]
    return [
        f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats.csv",
        # Ireland 2025 runs use get_sklearn_metrics_fixed (RORA_RORB name-truncation fix)
        # and were written with index=False -- no leading unnamed index column, unlike the
        # older summary_stats.csv files. Select columns by name only (never index_col=0)
        # so both formats read correctly here.
        f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats_fixed.csv",
    ]


def gather_metric(name, col):
    for c in _summary_stats_candidates(name):
        if os.path.exists(c):
            return pd.read_csv(c)[col].mean()
    return np.nan


def gather_r2(name):
    return gather_metric(name, "r2")


def gather_auc(name):
    return gather_metric(name, "roc_auc_score")


def subsample(df, n, rng):
    if len(df) <= n:
        return df
    return df.iloc[rng.choice(len(df), n, replace=False)]


def dataset_category(name):
    if name == "GEMM_train":
        return "GEMM_train"
    if name.startswith("allograft_"):
        return "allograft"
    if name.startswith("human_"):
        return "human_tumor"
    if name.startswith("organoid_"):
        return "organoid"
    if name == "mets_compiled":
        return "mets_compiled"
    if name in IRELAND_2025_SAMPLES:
        return "ireland_2025"
    return "other"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

    with open(f"{DIR_PREFIX}/6667/data_split/test_train_indicescombined.p", "rb") as f:
        split_indices = pickle.load(f)
    train_cellids = set(split_indices["train_cellID"])
    gemm_raw_full = bb.load.load_data(
        f"{DIR_PREFIX}/data/adata_imputed_combined_v3_RORA_RORB_ave.csv", nodes,
        norm=None, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    gemm_raw = gemm_raw_full.loc[gemm_raw_full.index.isin(train_cellids)]
    print(f"GEMM raw training data: {len(gemm_raw)}/{len(gemm_raw_full)} cells matched to the train split")

    sample_paths = gather_sample_paths()
    pooled_raw = [subsample(gemm_raw, MAX_CELLS_PER_SAMPLE, rng).assign(_dataset="GEMM_train")]

    for name, path in sample_paths.items():
        if not os.path.exists(path):
            print(f"MISSING: {name} -> {path}")
            continue
        d_raw = bb.load.load_data(path, nodes, norm=None, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)
        pooled_raw.append(subsample(d_raw, MAX_CELLS_PER_SAMPLE, rng).assign(_dataset=name))

    pooled = pd.concat(pooled_raw)
    # .to_numpy(dtype=object) instead of .values -- this pandas build backs plain string
    # columns with its "string" extension dtype, which doesn't support the 2D fancy
    # indexing (labels[knn_idx], labels[:, None]) used below the way a plain numpy array does.
    labels = pooled.pop("_dataset").to_numpy(dtype=object)
    print(f"Pooled {len(pooled)} cells across {len(set(labels))} datasets for PCA + kNN batch-mixing test.")

    scaler = StandardScaler()
    X = scaler.fit_transform(pooled.values)
    pca = PCA(n_components=N_PCS, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    print(f"PCA: top {N_PCS} PCs explain {pca.explained_variance_ratio_.sum():.3f} of total variance.")

    nn = NearestNeighbors(n_neighbors=KNN_K + 1).fit(X_pca)  # +1 to drop self-match
    _, knn_idx = nn.kneighbors(X_pca)
    knn_idx = knn_idx[:, 1:]  # drop self

    neighbor_labels = labels[knn_idx]  # (n_cells, KNN_K)
    same_dataset = (neighbor_labels == labels[:, None]).mean(axis=1)  # per-cell observed fraction

    dataset_counts = pd.Series(labels).value_counts()
    expected_fraction = dataset_counts / len(labels)  # per-dataset share of the pool

    per_cell = pd.DataFrame({"dataset": labels, "observed_same_dataset_frac": same_dataset})
    per_dataset = per_cell.groupby("dataset")["observed_same_dataset_frac"].mean().to_frame("mean_observed_same_dataset_frac")
    per_dataset["expected_same_dataset_frac"] = expected_fraction
    per_dataset["batch_separation_score"] = (
        per_dataset["mean_observed_same_dataset_frac"] / per_dataset["expected_same_dataset_frac"]
    )
    per_dataset["n_cells_used"] = dataset_counts
    per_dataset["mean_r2"] = [gather_r2(n) for n in per_dataset.index]
    per_dataset["mean_auc"] = [gather_auc(n) for n in per_dataset.index]
    per_dataset["category"] = [dataset_category(n) for n in per_dataset.index]
    # "percentage" framing: how much MORE (or less) often a cell neighbors its own dataset
    # than pure chance/random mixing would predict. 0% = perfectly mixed (no batch effect),
    # 100% = twice as batch-pure as chance, negative = more mixed than chance (essentially
    # never happens in practice, just a sanity floor).
    per_dataset["pct_excess_same_dataset_neighbors"] = (per_dataset["batch_separation_score"] - 1) * 100
    per_dataset = per_dataset.rename_axis("name").reset_index()
    per_dataset = per_dataset[per_dataset["name"] != "GEMM_train"].copy()  # GEMM itself isn't an "external" validation sample

    per_dataset.to_csv(f"{OUT_DIR}/batch_separation_scores.csv", index=False)

    scored = per_dataset.dropna(subset=["mean_r2"])
    corr_r2 = scored["pct_excess_same_dataset_neighbors"].corr(scored["mean_r2"])
    corr_auc = scored["pct_excess_same_dataset_neighbors"].corr(scored["mean_auc"])
    print(f"\n{len(scored)}/{len(per_dataset)} datasets have a scored mean_r2/mean_auc")
    print(f"corr(pct_excess_same_dataset_neighbors, mean_r2): {corr_r2:.3f}")
    print(f"corr(pct_excess_same_dataset_neighbors, mean_auc): {corr_auc:.3f}")
    print("(near 0 -> batch separation doesn't predict validation quality; strongly negative ->")
    print(" more batch-separated datasets validate worse, i.e. batch effects are a real driver)")

    pd.set_option("display.width", 160)
    print("\n=== Sorted by pct_excess_same_dataset_neighbors (highest = most batch-separated) ===")
    print(per_dataset.sort_values("pct_excess_same_dataset_neighbors", ascending=False).to_string(index=False))

    print("\n=== Ireland et al. 2025 datasets specifically ===")
    print(per_dataset[per_dataset["name"].isin(IRELAND_2025_SAMPLES)].to_string(index=False))

    # R2/AUC vs batch-effect-percentage plots, colored by category, requested directly.
    categories = sorted(scored["category"].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {c: cmap(i % 10) for i, c in enumerate(categories)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, ycol, ylabel, corr in [
        (axes[0], "mean_r2", "Mean R²", corr_r2),
        (axes[1], "mean_auc", "Mean AUC", corr_auc),
    ]:
        for cat in categories:
            sub = scored[scored["category"] == cat]
            ax.scatter(sub["pct_excess_same_dataset_neighbors"], sub[ycol], label=cat,
                       color=color_map[cat], s=60, edgecolor="k", linewidth=0.4, alpha=0.85)
        for _, row in scored.iterrows():
            ax.annotate(row["name"], (row["pct_excess_same_dataset_neighbors"], row[ycol]),
                        fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Batch separation: excess same-dataset kNN neighbors vs. chance (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. batch separation (r={corr:.2f})")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    axes[0].legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/r2_auc_vs_batch_separation.png", dpi=150)
    plt.savefig(f"{OUT_DIR}/r2_auc_vs_batch_separation.pdf")
    plt.close()

    print(f"\nWrote {OUT_DIR}/batch_separation_scores.csv, {OUT_DIR}/r2_auc_vs_batch_separation.{{png,pdf}}")


if __name__ == "__main__":
    main()
