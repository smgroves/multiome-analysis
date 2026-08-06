"""Full-scale version of diagnose_leaf_conditional_agreement.py's disambiguating test
(§9 of FINDINGS.md): instead of 5 genes x 1 sample, run every one of 6667's 53 network
genes against every scored external sample (allografts, human tumors, organoid variants,
mets_compiled), to build the per-gene x per-sample transferability matrix described in
BoBa-T_hyperparameters.md sec 11.

For each (gene, sample) pair: hard-binarize (>0.5) that sample's cells into their exact
combinatorial regulator state (same leaf indexing get_rules/parent_heatmap use), restrict
to leaves with >= MIN_CELLS_PER_LEAF cells, and compare each leaf's mean actual value to
GEMM's fitted rule value for that exact leaf. A gene's transferability score for a sample
is the cell-count-weighted fraction of its well-populated leaves with |residual| < 0.15.

Vectorized (bit bit-position weights instead of per-cell string ops) since this now runs
53 genes x ~33 samples, some with tens of thousands of cells.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/03_domain_shift_diagnostics/diagnose_leaf_conditional_agreement_full.py
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

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic"
MIN_CELLS_PER_LEAF = 10
RESIDUAL_THRESHOLD = 0.15


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


def leaf_indices_vectorized(data, regulators):
    """Vectorized version of hard-binarizing (>0.5) + state2idx: regulators[0] is the MSB,
    matching idx2binary(leaf, n)'s left-to-right bit order used in parent_heatmap."""
    n = len(regulators)
    binary = (data[regulators].values > 0.5).astype(int)
    weights = 2 ** np.arange(n - 1, -1, -1)
    return binary @ weights


def leaf_conditional_score(data, gene, regulators, rule):
    leaves = leaf_indices_vectorized(data, regulators)
    df = pd.DataFrame({"leaf": leaves, "actual": data[gene].values})
    grouped = df.groupby("leaf")["actual"].agg(["mean", "count"])
    grouped = grouped[grouped["count"] >= MIN_CELLS_PER_LEAF]
    if len(grouped) == 0:
        return np.nan, np.nan, 0
    grouped["gemm_rule_value"] = [rule[leaf] for leaf in grouped.index]
    grouped["residual"] = grouped["mean"] - grouped["gemm_rule_value"]
    grouped["agrees"] = grouped["residual"].abs() < RESIDUAL_THRESHOLD

    total_cells = grouped["count"].sum()
    transferability = (grouped["agrees"] * grouped["count"]).sum() / total_cells
    mean_abs_residual = (grouped["residual"].abs() * grouped["count"]).sum() / total_cells
    return transferability, mean_abs_residual, len(grouped)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")

    with open(f"{DIR_PREFIX}/6667/data_split/test_train_indicescombined.p", "rb") as f:
        split_indices = pickle.load(f)
    test_cellids = set(split_indices["test_cellID"])

    gemm_test = bb.load.load_data(
        f"{DIR_PREFIX}/6667/data_split/test_t0combined.csv", nodes,
        norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )

    all_rows = []

    # GEMM's own held-out test as the positive-control "sample"
    print("=== GEMM_held_out_test (positive control) ===")
    for gene in nodes:
        regulators = regulators_dict[gene]
        if len(regulators) == 0:
            continue
        transferability, mean_abs_residual, n_leaves = leaf_conditional_score(gemm_test, gene, regulators, rules[gene])
        all_rows.append({
            "gene": gene, "sample": "GEMM_held_out_test", "n_regulators": len(regulators),
            "n_well_populated_leaves": n_leaves, "transferability": transferability,
            "mean_abs_residual": mean_abs_residual, "sample_mean_r2": np.nan,
        })

    sample_paths = gather_sample_paths()
    for name, path in sample_paths.items():
        if not os.path.exists(path):
            print(f"MISSING: {name} -> {path}")
            continue
        print(f"=== {name} ===")
        d = bb.load.load_data(path, nodes, norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)
        sample_r2 = gather_r2(name)
        for gene in nodes:
            regulators = regulators_dict[gene]
            if len(regulators) == 0:
                continue
            transferability, mean_abs_residual, n_leaves = leaf_conditional_score(d, gene, regulators, rules[gene])
            all_rows.append({
                "gene": gene, "sample": name, "n_regulators": len(regulators),
                "n_well_populated_leaves": n_leaves, "transferability": transferability,
                "mean_abs_residual": mean_abs_residual, "sample_mean_r2": sample_r2,
            })

    result = pd.DataFrame(all_rows)
    result.to_csv(f"{OUT_DIR}/leaf_conditional_transferability_matrix.csv", index=False)
    print(f"\nWrote {len(result)} (gene, sample) rows to {OUT_DIR}/leaf_conditional_transferability_matrix.csv")

    # Per-gene robust-core score: mean transferability across all EXTERNAL samples
    # (excluding the GEMM positive control), weighted equally per sample.
    ext = result[result["sample"] != "GEMM_held_out_test"]
    per_gene = ext.groupby("gene").agg(
        mean_transferability=("transferability", "mean"),
        n_samples_tested=("transferability", "count"),
        mean_n_regulators=("n_regulators", "mean"),
    ).sort_values("mean_transferability")
    per_gene.to_csv(f"{OUT_DIR}/leaf_conditional_robust_core_per_gene.csv")

    gemm_control = result[result["sample"] == "GEMM_held_out_test"].set_index("gene")["transferability"]
    print(f"\nGEMM held-out test (positive control) mean transferability across genes: {gemm_control.mean():.3f}")

    print("\n=== Robust-core ranking: genes MOST transferable across external samples (top 10) ===")
    pd.set_option("display.width", 160)
    print(per_gene.tail(10).to_string())
    print("\n=== Genes LEAST transferable across external samples (bottom 10, i.e. most context-specific) ===")
    print(per_gene.head(10).to_string())

    # Per-sample summary: does mean transferability across genes track mean_r2 (sanity check)?
    per_sample = ext.groupby("sample").agg(
        mean_transferability=("transferability", "mean"),
        mean_r2=("sample_mean_r2", "first"),
    ).dropna()
    print(f"\ncorr(per-sample mean transferability, sample mean_r2) across {len(per_sample)} samples: "
          f"{per_sample['mean_transferability'].corr(per_sample['mean_r2']):.3f}")
    per_sample.to_csv(f"{OUT_DIR}/leaf_conditional_per_sample_summary.csv")

    print(f"\nWrote {OUT_DIR}/leaf_conditional_robust_core_per_gene.csv, "
          f"{OUT_DIR}/leaf_conditional_per_sample_summary.csv")


if __name__ == "__main__":
    main()
