"""Does distance from the GEMM training distribution -- in a shared UMAP embedding, or
from the 8 known GEMM attractors -- predict how well a given external validation sample
scores, without assuming any category (in-vitro, species, tissue type) in advance? This
is the direct, assumption-free version of the domain-shift question raised in
diagnose_organoid_shgfp.py / compare_correlation_shift_across_samples.py, which found the
network's strong-edge sign-flip rate predicts mean R2 well (r=-0.72) but is itself an
indirect proxy. This script checks two more direct distance metrics against the same
population of 33 scored samples (12 allografts, 20 human tumors, 4 organoid variants,
mets_compiled):

1. UMAP-space distance: pool GEMM training cells with every external sample's cells (raw,
   node_normalization=None, comparable log1p+MAGIC-imputed expression units), fit one
   shared UMAP embedding, and for each sample compute (a) its centroid's Euclidean distance
   from the GEMM training centroid in UMAP space, and (b) the mean distance from each of its
   cells to its k=15 nearest GEMM training cells in the *original* 53-gene expression space
   (a less distortion-prone complement to the UMAP-space metric, since UMAP does not
   preserve global distances reliably).
2. Archetype distance: each sample's mean continuous state (node_normalization=0.3, the
   same representation used for the mets_compiled cluster2 attractor check) vs. the nearest
   of 6667's 8 characterized GEMM attractors (6667/attractors/average_states.txt).

Run in bobaT_env_py3.13 (needs `pip install umap-learn`, already present):
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/03_domain_shift_diagnostics/umap_archetype_distance_analysis.py
"""

import glob
import os
import pickle
import re

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd
import umap
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic"
MAX_CELLS_PER_SAMPLE = 1500
RANDOM_STATE = 1234
KNN_K = 15


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
    return paths


def gather_r2(name):
    candidates = []
    if name.startswith("allograft_"):
        candidates.append(f"{DIR_PREFIX}/6667/validation/allografts/{name[len('allograft_'):]}/summary_stats.csv")
    elif name.startswith("human_"):
        candidates.append(f"{DIR_PREFIX}/6667/validation/human_tumor_MSK/{name[len('human_'):]}/summary_stats.csv")
    elif name == "organoid_combined":
        candidates.append(f"{DIR_PREFIX}/6667/validation/external_validation/organoid/summary_stats.csv")
    else:
        candidates.append(f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats.csv")
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c, index_col=0)["r2"].mean()
    return np.nan


def subsample(df, n, rng):
    if len(df) <= n:
        return df
    return df.iloc[rng.choice(len(df), n, replace=False)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

    # NOTE: 6667/data_split/train_t0combined.csv is already node_normalization=0.3-clipped
    # (confirmed: exactly 60% of any gene's values are exactly 0 or 1), not raw, despite
    # loading with norm=None -- that mismatched GEMM's "raw" representation against every
    # external sample's genuinely-raw units in the pooled UMAP/kNN distance metrics below.
    # `gemm_normed` (norm=0.3 applied AGAIN on top) is not similarly affected: an
    # already-clipped variable's own 30th/70th percentiles are ~0/~1, so re-applying the
    # same transform is close to a no-op -- kept as-is for the attractor-distance metric.
    with open(f"{DIR_PREFIX}/6667/data_split/test_train_indicescombined.p", "rb") as f:
        split_indices = pickle.load(f)
    train_cellids = set(split_indices["train_cellID"])
    gemm_raw_full = bb.load.load_data(
        f"{DIR_PREFIX}/data/adata_imputed_combined_v3_RORA_RORB_ave.csv", nodes,
        norm=None, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    gemm_raw = gemm_raw_full.loc[gemm_raw_full.index.isin(train_cellids)]
    print(f"GEMM raw training data: {len(gemm_raw)}/{len(gemm_raw_full)} cells matched to the train split")
    gemm_normed = bb.load.load_data(
        f"{DIR_PREFIX}/6667/data_split/train_t0combined.csv", nodes,
        norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    attractors = pd.read_csv(f"{DIR_PREFIX}/6667/attractors/average_states.txt", index_col=0)[nodes]

    sample_paths = gather_sample_paths()
    pooled_raw = [subsample(gemm_raw, MAX_CELLS_PER_SAMPLE, rng).assign(_dataset="GEMM_train")]
    normed_means = {"GEMM_train": gemm_normed.mean()}

    for name, path in sample_paths.items():
        if not os.path.exists(path):
            print(f"MISSING: {name} -> {path}")
            continue
        d_raw = bb.load.load_data(path, nodes, norm=None, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)
        d_normed = bb.load.load_data(path, nodes, norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)
        pooled_raw.append(subsample(d_raw, MAX_CELLS_PER_SAMPLE, rng).assign(_dataset=name))
        normed_means[name] = d_normed.mean()

    pooled = pd.concat(pooled_raw)
    labels = pooled.pop("_dataset")
    print(f"Pooled {len(pooled)} cells across {labels.nunique()} datasets for UMAP + kNN distance.")

    scaler = StandardScaler()
    X = scaler.fit_transform(pooled.values)

    print("Fitting UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=RANDOM_STATE)
    embedding = reducer.fit_transform(X)
    emb_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"], index=pooled.index)
    emb_df["dataset"] = labels.values
    emb_df.to_csv(f"{OUT_DIR}/umap_embedding_all_samples.csv")

    gemm_mask = labels.values == "GEMM_train"
    gemm_centroid_umap = embedding[gemm_mask].mean(axis=0)
    gemm_X = X[gemm_mask]

    rows = []
    for name in list(sample_paths.keys()) :
        if name not in normed_means:
            continue
        mask = labels.values == name
        if mask.sum() == 0:
            continue
        sample_embedding = embedding[mask]
        sample_X = X[mask]

        umap_centroid_dist = np.linalg.norm(sample_embedding.mean(axis=0) - gemm_centroid_umap)

        dists_to_gemm = cdist(sample_X, gemm_X)
        knn_dist = np.sort(dists_to_gemm, axis=1)[:, :KNN_K].mean()

        normed_mean = normed_means[name]
        arc_dists = np.sqrt(((attractors - normed_mean) ** 2).sum(axis=1))
        nearest_arc = arc_dists.idxmin()
        nearest_arc_dist = arc_dists.min()

        rows.append({
            "name": name, "n_cells_used": mask.sum(),
            "umap_centroid_dist_from_gemm": umap_centroid_dist,
            "mean_knn_dist_from_gemm_raw_space": knn_dist,
            "nearest_attractor": nearest_arc,
            "nearest_attractor_dist": nearest_arc_dist,
            "mean_r2": gather_r2(name),
        })

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT_DIR}/umap_archetype_distance_vs_r2.csv", index=False)

    scored = res.dropna(subset=["mean_r2"])
    print(f"\n{len(scored)}/{len(res)} samples have a scored mean_r2")
    print(f"corr(umap_centroid_dist_from_gemm, mean_r2): {scored['umap_centroid_dist_from_gemm'].corr(scored['mean_r2']):.3f}")
    print(f"corr(mean_knn_dist_from_gemm_raw_space, mean_r2): {scored['mean_knn_dist_from_gemm_raw_space'].corr(scored['mean_r2']):.3f}")
    print(f"corr(nearest_attractor_dist, mean_r2): {scored['nearest_attractor_dist'].corr(scored['mean_r2']):.3f}")

    pd.set_option("display.width", 160)
    print("\n=== Bottom 8 by mean_r2 ===")
    print(scored.sort_values("mean_r2").head(8).to_string(index=False))
    print("\n=== Top 8 by mean_r2 ===")
    print(scored.sort_values("mean_r2", ascending=False).head(8).to_string(index=False))

    print(f"\nWrote {OUT_DIR}/umap_embedding_all_samples.csv, {OUT_DIR}/umap_archetype_distance_vs_r2.csv")


if __name__ == "__main__":
    main()
