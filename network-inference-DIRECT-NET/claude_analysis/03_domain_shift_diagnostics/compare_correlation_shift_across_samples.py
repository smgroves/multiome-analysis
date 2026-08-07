"""Does the gene-regulator correlation-structure shift (vs. GEMM training data) found for
organoid_shGFP in diagnose_organoid_shgfp.py generalize as a predictor of external
validation R2 -- and is organoid a categorical outlier, or does it sit on the same
continuum as allografts and human tumors?

Initial framing (in-vitro organoid vs in-vivo allograft/human-tumor) turned out to be
wrong when checked against the full sample set: organoid_shGFP's edge-correlation
preservation is unremarkable among ALL external samples -- several allograft and human
tumor samples show similarly degraded (or worse) correlation-structure preservation. What
DOES hold up, checked across every allograft/human-tumor/organoid/mets_compiled sample
6667 has external validation for: the network's strong-edge sign-flip rate (fraction of
GEMM's |r|>0.5 edges whose correlation sign flips in that sample) is a strong, general,
continuous predictor of that sample's mean R2 (r=-0.72 across n=33 scored samples) --
organoid_shGFP and mets_compiled sit at the bad tail of this SAME population-wide
relationship, not in a separate in-vitro category. The worst human tumor sample (RU1215)
and worst allografts (mt2, mt3) show comparably poor correlation preservation.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/03_domain_shift_diagnostics/compare_correlation_shift_across_samples.py
"""

import glob
import os
import re

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic_low_R2_samples"


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
    """Look up 6667's own summary_stats.csv mean r2 for this sample name, across whichever
    validation subdirectory it lives in."""
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
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")
    gemm_train = bb.load.load_data(
        f"{DIR_PREFIX}/6667/data_split/train_t0combined.csv", nodes,
        norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )

    rows = []
    for name, path in gather_sample_paths().items():
        if not os.path.exists(path):
            print(f"MISSING: {name} -> {path}")
            continue
        d = bb.load.load_data(path, nodes, norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)
        edge_rows = []
        for g, regs in regulators_dict.items():
            for r in regs:
                if r not in gemm_train.columns or r not in d.columns:
                    continue
                edge_rows.append((gemm_train[g].corr(gemm_train[r]), d[g].corr(d[r])))
        edf = pd.DataFrame(edge_rows, columns=["corr_gemm", "corr_ext"]).dropna()
        edf["sign_flip"] = np.sign(edf["corr_gemm"]) != np.sign(edf["corr_ext"])
        strong = edf[edf["corr_gemm"].abs() > 0.5]

        category = (
            "organoid" if name.startswith("organoid") else
            "mets_compiled" if name == "mets_compiled" else
            # TKO-luc lives under the allografts/ folder but isn't actually an allograft --
            # keep it out of that category rather than mislabeling it.
            "TKO" if name == "allograft_TKO-luc" else
            "allograft" if name.startswith("allograft") else "human_tumor"
        )
        rows.append({
            "name": name, "category": category,
            "overall_corr": edf["corr_gemm"].corr(edf["corr_ext"]),
            "sign_flip_rate": edf["sign_flip"].mean(),
            "strong_sign_flip_rate": strong["sign_flip"].mean(),
            "n_cells": len(d), "mean_r2": gather_r2(name),
        })

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT_DIR}/all_samples_corr_vs_r2.csv", index=False)

    scored = res.dropna(subset=["mean_r2"])
    print(f"{len(scored)}/{len(res)} samples have a scored mean_r2 (some human tumor samples have no "
          f"summary_stats.csv on disk yet)")
    print(f"\ncorr(strong_sign_flip_rate, mean_r2) across all {len(scored)} samples: "
          f"{scored['strong_sign_flip_rate'].corr(scored['mean_r2']):.3f}")
    print(f"corr(sign_flip_rate, mean_r2): {scored['sign_flip_rate'].corr(scored['mean_r2']):.3f}")
    print(f"corr(overall_corr, mean_r2): {scored['overall_corr'].corr(scored['mean_r2']):.3f}  (weak/noisy -- not a useful single predictor)")
    print(f"corr(n_cells, mean_r2): {scored['n_cells'].corr(scored['mean_r2']):.3f}")
    print(f"corr(n_cells, strong_sign_flip_rate): {scored['n_cells'].corr(scored['strong_sign_flip_rate']):.3f}  (weak -- not primarily an N artifact)")

    print("\n=== Bottom 8 by mean_r2 (worst-validating samples, any category) ===")
    pd.set_option("display.width", 160)
    print(scored.sort_values("mean_r2").head(8).to_string(index=False))
    print(f"\nWrote {OUT_DIR}/all_samples_corr_vs_r2.csv")


if __name__ == "__main__":
    main()
