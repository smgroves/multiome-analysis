"""The direct test: does the network-gene batch-separation signal found in
test_batch_effects_all_samples.py (r=+0.62 with mean R2) reflect a genuine TECHNICAL batch
effect, or real biology? If it's a pervasive technical confound, a panel of 12 housekeeping
genes (Actb, Gapdh, Tbp, B2m, Ppia, Rplp0, Ywhaz, Sdha, Hprt, Ubc, Polr2a, Tubb5 -- picked
for NOT being expected to vary with cell-state/lineage biology) extracted from the SAME raw
sources (extract_housekeeping_genes.py / extract_housekeeping_genes_allografts.py) should
show comparably strong dataset separation. If it's real biology, housekeeping genes should
be much better mixed across datasets than the network genes are.

Restricted to the 19 datasets where a housekeeping-gene extraction was actually built: GEMM
training data, organoid (combined), the 5 Ireland et al. 2025 datasets, and the 12
allografts (human tumors and mets_compiled excluded -- no raw full-transcriptome source
available for the former, R/Seurat re-access needed for the latter, out of scope for this
round per direct user decision). Both panels are scored with the IDENTICAL PCA+kNN
batch-mixing metric on the SAME pool of datasets, so the two numbers are directly comparable
for each dataset.

Run in bobaT_env_py3.13 (needs bobaT for network-gene loading + sklearn for PCA/kNN):
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/07_batch_effect_diagnostics/compare_network_vs_housekeeping_batch_separation.py
"""

import glob
import os
import pickle

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
HK_DIR = f"{DIR_PREFIX}/claude_analysis/07_batch_effect_diagnostics/housekeeping"
OUT_DIR = f"{DIR_PREFIX}/claude_analysis/07_batch_effect_diagnostics"
MAX_CELLS_PER_SAMPLE = 1500
RANDOM_STATE = 1234
KNN_K = 30
N_PCS = 10  # only 12 housekeeping genes available -- cap PCs below that

IRELAND_2025_SAMPLES = ["cgrp_k5", "organoid_celltag", "tbo_allograft_5khvg", "rpr2_allograft", "celltag_fate_dpt"]
ALLOGRAFTS = ["1L", "2L", "2LR", "3L", "5B", "TKO-luc", "mt2", "mt3", "mt4", "mt4Rf", "mt5", "mt6"]


def subsample(df, n, rng):
    if len(df) <= n:
        return df
    return df.iloc[rng.choice(len(df), n, replace=False)]


def gather_network_gene_paths():
    paths = {"GEMM_train": None}  # handled specially below
    for a in ALLOGRAFTS:
        paths[f"allograft_{a}"] = f"{DIR_PREFIX}/data/allografts/adata_{a}_allografts_v3_RORA_RORB_ave.csv"
    paths["organoid_combined"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_v3_RORA_RORB_ave.csv"
    for name in IRELAND_2025_SAMPLES:
        paths[name] = f"{DIR_PREFIX}/data/{name}/adata_{name}_v3_RORA_RORB_ave.csv"
    return paths


def gather_hk_paths():
    paths = {"GEMM_train": f"{HK_DIR}/hk_GEMM_train.csv"}
    for a in ALLOGRAFTS:
        paths[f"allograft_{a}"] = f"{HK_DIR}/hk_allograft_{a}.csv"
    paths["organoid_combined"] = f"{HK_DIR}/hk_organoid_combined.csv"
    for name in IRELAND_2025_SAMPLES:
        paths[name] = f"{HK_DIR}/hk_{name}.csv"
    return paths


def gather_r2(name):
    if name.startswith("allograft_"):
        c = f"{DIR_PREFIX}/6667/validation/allografts/{name[len('allograft_'):]}/summary_stats.csv"
    elif name == "organoid_combined":
        c = f"{DIR_PREFIX}/6667/validation/external_validation/organoid/summary_stats.csv"
    else:
        c = f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats_fixed.csv"
        if not os.path.exists(c):
            c = f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats.csv"
    return pd.read_csv(c)["r2"].mean() if os.path.exists(c) else np.nan


def batch_separation(pooled_by_dataset, rng, n_pcs):
    """pooled_by_dataset: {name: DataFrame (cells x genes)}. Returns per-dataset
    batch_separation_score (mean observed same-dataset kNN fraction / expected under
    random mixing)."""
    frames = []
    for name, df in pooled_by_dataset.items():
        frames.append(subsample(df, MAX_CELLS_PER_SAMPLE, rng).assign(_dataset=name))
    pooled = pd.concat(frames)
    labels = pooled.pop("_dataset").to_numpy(dtype=object)

    X = StandardScaler().fit_transform(pooled.values)
    n_pcs_use = min(n_pcs, X.shape[1])
    X_pca = PCA(n_components=n_pcs_use, random_state=RANDOM_STATE).fit_transform(X)

    nn = NearestNeighbors(n_neighbors=KNN_K + 1).fit(X_pca)
    _, knn_idx = nn.kneighbors(X_pca)
    knn_idx = knn_idx[:, 1:]

    neighbor_labels = labels[knn_idx]
    same_dataset = (neighbor_labels == labels[:, None]).mean(axis=1)

    dataset_counts = pd.Series(labels).value_counts()
    expected_fraction = dataset_counts / len(labels)

    per_cell = pd.DataFrame({"dataset": labels, "obs": same_dataset})
    per_dataset = per_cell.groupby("dataset")["obs"].mean().to_frame("mean_observed_same_dataset_frac")
    per_dataset["expected_same_dataset_frac"] = expected_fraction
    per_dataset["batch_separation_score"] = per_dataset["mean_observed_same_dataset_frac"] / per_dataset["expected_same_dataset_frac"]
    return per_dataset["batch_separation_score"]


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

    network_paths = gather_network_gene_paths()
    network_pooled = {"GEMM_train": gemm_raw}
    for name, path in network_paths.items():
        if name == "GEMM_train" or not os.path.exists(path):
            if name != "GEMM_train":
                print(f"MISSING network-gene file: {name} -> {path}")
            continue
        network_pooled[name] = bb.load.load_data(path, nodes, norm=None, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)

    hk_paths = gather_hk_paths()
    hk_pooled = {}
    for name, path in hk_paths.items():
        if not os.path.exists(path):
            print(f"MISSING housekeeping-gene file: {name} -> {path}")
            continue
        hk_pooled[name] = pd.read_csv(path, index_col=0)

    common_names = sorted(set(network_pooled) & set(hk_pooled))
    print(f"{len(common_names)} datasets with both network-gene and housekeeping-gene data: {common_names}")
    network_pooled = {k: v for k, v in network_pooled.items() if k in common_names}
    hk_pooled = {k: v for k, v in hk_pooled.items() if k in common_names}

    print("\nComputing network-gene batch separation...")
    network_scores = batch_separation(network_pooled, rng, N_PCS)
    print("Computing housekeeping-gene batch separation...")
    hk_scores = batch_separation(hk_pooled, rng, N_PCS)

    result = pd.DataFrame({
        "network_gene_batch_separation": network_scores,
        "housekeeping_gene_batch_separation": hk_scores,
    })
    result["mean_r2"] = [gather_r2(n) for n in result.index]
    result = result.rename_axis("name").reset_index()
    result = result[result["name"] != "GEMM_train"].copy()
    result.to_csv(f"{OUT_DIR}/network_vs_housekeeping_batch_separation.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n=== Network-gene vs. housekeeping-gene batch separation, per dataset ===")
    print(result.sort_values("network_gene_batch_separation", ascending=False).to_string(index=False))

    ratio = (result["housekeeping_gene_batch_separation"] / result["network_gene_batch_separation"])
    print(f"\nMean housekeeping/network batch-separation ratio: {ratio.mean():.3f}")
    print("(near 1.0 -> housekeeping genes separate by dataset just as much as network genes,")
    print(" i.e. a pervasive TECHNICAL batch effect; near 0 -> separation is specific to the")
    print(" biologically-relevant network genes, i.e. real biology, not a technical artifact)")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(result["network_gene_batch_separation"], result["housekeeping_gene_batch_separation"], s=60, edgecolor="k")
    for _, row in result.iterrows():
        ax.annotate(row["name"], (row["network_gene_batch_separation"], row["housekeeping_gene_batch_separation"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    lims = [0, max(result["network_gene_batch_separation"].max(), result["housekeeping_gene_batch_separation"].max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=0.8, label="y=x (equal separation)")
    ax.set_xlabel("Network-gene batch separation score")
    ax.set_ylabel("Housekeeping-gene batch separation score")
    ax.set_title("Is network-gene dataset separation just as strong in housekeeping genes?")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/network_vs_housekeeping_batch_separation.png", dpi=150)
    plt.savefig(f"{OUT_DIR}/network_vs_housekeeping_batch_separation.pdf")
    plt.close()

    print(f"\nWrote {OUT_DIR}/network_vs_housekeeping_batch_separation.csv, .png, .pdf")


if __name__ == "__main__":
    main()
