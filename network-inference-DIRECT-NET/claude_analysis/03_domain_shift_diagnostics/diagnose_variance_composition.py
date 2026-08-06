"""§6 found relative diversity (mean per-gene raw std vs GEMM's) predicts R2 at r=0.58, but
doesn't fully explain organoid_shGFP (middling diversity, still near-bottom R2). Candidate
refinement: it's not how MUCH variance a sample has, but what AXIS that variance lies
along. A sample could have plenty of raw spread that is mostly stress/technical variance
(inflating the §6 number) rather than variance along the identity axes BoBa-T's rules
actually depend on (per §3's mets_compiled cluster2 finding: dissociation-stress IEG
signature, not a real archetype).

Fits PCA on GEMM training data (its own dominant axes of biological variation), projects
every external sample onto that FIXED basis (GEMM's own mean + loadings, not a
per-sample refit), and for each sample computes:
  - fraction of that sample's own total variance captured by GEMM's top-k identity PCs
  - fraction of that sample's own total variance along the specific IEG-gene axis
    (JUN/JUND/JUNB/FOS/FOSB/EGR1 subspace)
then correlates both against mean R2 (and against §6's raw diversity number) to see if
"the right kind of variance" is a better predictor than "how much variance."

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/03_domain_shift_diagnostics/diagnose_variance_composition.py
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
from sklearn.decomposition import PCA

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic"
N_IDENTITY_PCS = 5
IEG_GENES = ["JUN", "JUND", "JUNB", "FOS", "FOSB", "EGR1"]


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    # NOTE: 6667/data_split/train_t0combined.csv is already node_normalization=0.3-clipped
    # (confirmed: exactly 60% of any gene's values are exactly 0 or 1), not raw, despite
    # loading with norm=None. Use the true raw source file, restricted to the train split.
    with open(f"{DIR_PREFIX}/6667/data_split/test_train_indicescombined.p", "rb") as f:
        split_indices = pickle.load(f)
    train_cellids = set(split_indices["train_cellID"])
    gemm_raw_full = bb.load.load_data(
        f"{DIR_PREFIX}/data/adata_imputed_combined_v3_RORA_RORB_ave.csv", nodes,
        norm=None, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    gemm_raw = gemm_raw_full.loc[gemm_raw_full.index.isin(train_cellids)]
    print(f"GEMM raw training data: {len(gemm_raw)}/{len(gemm_raw_full)} cells matched to the train split")

    pca = PCA(n_components=N_IDENTITY_PCS, random_state=1234)
    pca.fit(gemm_raw.values)
    print(f"GEMM's top {N_IDENTITY_PCS} PCs explain {pca.explained_variance_ratio_.sum():.3f} of GEMM's own variance")
    print(f"Per-PC explained variance ratio (GEMM): {np.round(pca.explained_variance_ratio_, 3)}")

    loadings = pd.DataFrame(pca.components_, columns=nodes, index=[f"PC{i+1}" for i in range(N_IDENTITY_PCS)])
    print("\nTop +/- loading genes per PC (what each identity axis represents):")
    for pc in loadings.index:
        top = loadings.loc[pc].sort_values()
        print(f"  {pc}: low=({top.index[0]}:{top.iloc[0]:.2f}, {top.index[1]}:{top.iloc[1]:.2f})  "
              f"high=({top.index[-1]}:{top.iloc[-1]:.2f}, {top.index[-2]}:{top.iloc[-2]:.2f})")

    ieg_present = [g for g in IEG_GENES if g in nodes]
    ieg_idx = [nodes.index(g) for g in ieg_present]

    gemm_mean = gemm_raw.mean().values

    rows = []
    for name, path in gather_sample_paths().items():
        if not os.path.exists(path):
            continue
        d = bb.load.load_data(path, nodes, norm=None, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)
        X = d.values - gemm_mean  # center on GEMM's mean, not the sample's own (fixed reference)

        total_var = np.var(d.values, axis=0).sum()
        pc_scores = X @ pca.components_.T  # project onto GEMM's fixed identity axes
        identity_var = np.var(pc_scores, axis=0).sum()
        identity_frac = identity_var / total_var if total_var > 0 else np.nan

        ieg_var = np.var(d.values[:, ieg_idx], axis=0).sum()
        ieg_frac = ieg_var / total_var if total_var > 0 else np.nan

        rows.append({
            "name": name, "total_var": total_var,
            "identity_pc_var_frac": identity_frac, "ieg_var_frac": ieg_frac,
            "mean_r2": gather_r2(name),
        })

    res = pd.DataFrame(rows)
    diversity_df = pd.read_csv(f"{OUT_DIR}/sample_diversity_vs_signflip_r2.csv")
    merged = res.merge(diversity_df[["name", "relative_diversity_vs_gemm", "strong_sign_flip_rate"]], on="name")
    merged.to_csv(f"{OUT_DIR}/variance_composition_vs_r2.csv", index=False)

    scored = merged.dropna(subset=["mean_r2"])
    print(f"\ncorr(identity_pc_var_frac, mean_r2): {scored['identity_pc_var_frac'].corr(scored['mean_r2']):.3f}")
    print(f"corr(ieg_var_frac, mean_r2): {scored['ieg_var_frac'].corr(scored['mean_r2']):.3f}")
    print(f"corr(relative_diversity_vs_gemm, mean_r2) [for reference, sec 6]: {scored['relative_diversity_vs_gemm'].corr(scored['mean_r2']):.3f}")
    print(f"corr(identity_pc_var_frac, relative_diversity_vs_gemm): {scored['identity_pc_var_frac'].corr(scored['relative_diversity_vs_gemm']):.3f}")

    pd.set_option("display.width", 160)
    print("\n=== Bottom 8 by mean_r2 ===")
    print(scored.sort_values("mean_r2").head(8)[["name", "relative_diversity_vs_gemm", "identity_pc_var_frac", "ieg_var_frac", "mean_r2"]].to_string(index=False))
    print("\n=== Top 8 by mean_r2 ===")
    print(scored.sort_values("mean_r2", ascending=False).head(8)[["name", "relative_diversity_vs_gemm", "identity_pc_var_frac", "ieg_var_frac", "mean_r2"]].to_string(index=False))
    print(f"\n=== organoid_shGFP specifically vs mets_compiled ===")
    print(scored[scored["name"].isin(["organoid_shGFP", "mets_compiled"])][["name", "relative_diversity_vs_gemm", "identity_pc_var_frac", "ieg_var_frac", "mean_r2"]].to_string(index=False))

    print(f"\nWrote {OUT_DIR}/variance_composition_vs_r2.csv")


if __name__ == "__main__":
    main()
