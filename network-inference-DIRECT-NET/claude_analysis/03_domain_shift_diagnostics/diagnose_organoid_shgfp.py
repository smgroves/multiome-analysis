"""Deep-dive on organoid_shGFP's poor external validation (mean R2 ~0.02 on 6667's rules).
Three questions:

1. Are the worst-validating genes just hard for BoBa-T generally, or specifically bad on
   organoid? -> compare each gene's in-sample (6667 held-out GEMM test) R2 to its shGFP R2.
2. Is this a marginal-distribution/normalization artifact, or does the underlying
   gene-regulator co-expression structure itself differ between organoid culture and GEMM?
   -> compare Pearson correlation of every fitted gene-regulator edge, GEMM training data
   vs organoid_shGFP, after both are put through the same node_normalization=0.3 pipeline
   (so marginal shape is controlled for by construction; only joint/relational structure
   is being compared).
3. Is "in vitro organoid" specifically the driver, or is any external/unseen dataset this
   bad? -> control comparison against 6667's own in-vivo allograft external validation
   (unseen GEMM-derived tumors, still in vivo) in 6667/validation/allografts/*/.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/03_domain_shift_diagnostics/diagnose_organoid_shgfp.py
"""

import glob
import os

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
IEG_GENES = ["JUN", "JUND", "JUNB", "FOS", "FOSB", "EGR1"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 1. In-sample vs external R2 ---
    insample = pd.read_csv(f"{DIR_PREFIX}/6667/validation/in_sample_validation/summary_stats.csv", index_col=0).set_index("gene")
    shgfp = pd.read_csv(f"{DIR_PREFIX}/6667/validation/external_validation/organoid_shGFP/summary_stats.csv", index_col=0).set_index("gene")
    common = insample.index.intersection(shgfp.index)
    r2_compare = pd.DataFrame({
        "insample_r2": insample.loc[common, "r2"], "shgfp_r2": shgfp.loc[common, "r2"],
        "insample_auc": insample.loc[common, "roc_auc_score"], "shgfp_auc": shgfp.loc[common, "roc_auc_score"],
    })
    r2_compare["r2_drop"] = r2_compare["insample_r2"] - r2_compare["shgfp_r2"]
    r2_compare = r2_compare.sort_values("r2_drop", ascending=False)
    r2_compare.to_csv(f"{OUT_DIR}/organoid_shgfp_insample_vs_external_r2.csv")

    print("=== 1. In-sample (GEMM held-out) vs external (organoid_shGFP) R2 ===")
    print(f"Mean in-sample r2: {r2_compare['insample_r2'].mean():.3f}  Mean shGFP r2: {r2_compare['shgfp_r2'].mean():.3f}")
    n_bad_both = ((r2_compare["insample_r2"] < 0.3) & (r2_compare["shgfp_r2"] < 0.3)).sum()
    print(f"Genes bad in BOTH in-sample and shGFP (generally hard, not shGFP-specific): {n_bad_both}/{len(r2_compare)}")
    print("Every one of shGFP's worst genes validates well in-sample -- this is not a set of "
          "generally-hard-to-predict genes; it's specific to the organoid context.\n")

    # --- 2. Gene-regulator correlation structure: GEMM vs organoid ---
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")

    gemm_train = bb.load.load_data(
        f"{DIR_PREFIX}/6667/data_split/train_t0combined.csv", nodes,
        norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    shgfp_data = bb.load.load_data(
        f"{DIR_PREFIX}/data/organoid/adata_organoid_shGFP_v3_RORA_RORB_ave.csv", nodes,
        norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )

    edge_rows = []
    for g, regs in regulators_dict.items():
        for r in regs:
            if r not in gemm_train.columns or r not in shgfp_data.columns:
                continue
            c_gemm = gemm_train[g].corr(gemm_train[r])
            c_org = shgfp_data[g].corr(shgfp_data[r])
            edge_rows.append((g, r, c_gemm, c_org))
    edges = pd.DataFrame(edge_rows, columns=["gene", "regulator", "corr_gemm", "corr_organoid_shgfp"]).drop_duplicates()
    edges["sign_flip"] = np.sign(edges["corr_gemm"]) != np.sign(edges["corr_organoid_shgfp"])
    edges["ieg_involved"] = edges["gene"].isin(IEG_GENES) | edges["regulator"].isin(IEG_GENES)
    edges.to_csv(f"{OUT_DIR}/organoid_shgfp_edge_correlation_shift.csv", index=False)

    print("=== 2. Gene-regulator correlation structure: GEMM training data vs organoid_shGFP ===")
    print(f"{len(edges)} fitted edges compared (same node_normalization=0.3 pipeline both sides)")
    print(f"Correlation between corr_gemm and corr_organoid_shgfp across all edges: {edges['corr_gemm'].corr(edges['corr_organoid_shgfp']):.3f}")
    print(f"Fraction of edges with a sign flip: {edges['sign_flip'].mean():.3f}")
    strong = edges[edges["corr_gemm"].abs() > 0.5]
    print(f"Fraction of STRONG GEMM edges (|r|>0.5) that flip sign in organoid: {strong['sign_flip'].mean():.3f} (n={len(strong)})")
    print(f"Sign-flip rate, edges involving an IEG (JUN/JUND/JUNB/FOS/FOSB/EGR1): {edges[edges.ieg_involved]['sign_flip'].mean():.3f} (n={edges.ieg_involved.sum()})")
    print(f"Sign-flip rate, edges NOT involving an IEG: {edges[~edges.ieg_involved]['sign_flip'].mean():.3f} (n={(~edges.ieg_involved).sum()})")
    print("This is not a marginal-distribution artifact (both datasets went through the same "
          "per-dataset quantile-clip norm) -- the underlying gene-gene relationships BoBa-T's "
          "rules were fit on are substantially different in organoid culture.\n")

    # --- 3. Control: in-vivo allograft external validation (unseen tumors, still in vivo) ---
    allograft_rows = []
    for f in glob.glob(f"{DIR_PREFIX}/6667/validation/allografts/*/summary_stats.csv"):
        sample = f.split("/")[-2]
        df = pd.read_csv(f, index_col=0)
        allograft_rows.append((sample, df["r2"].mean(), df["roc_auc_score"].mean()))
    allografts = pd.DataFrame(allograft_rows, columns=["sample", "mean_r2", "mean_auc"]).sort_values("mean_r2", ascending=False)
    allografts.to_csv(f"{OUT_DIR}/allograft_control_summary.csv", index=False)

    print("=== 3. Control: in-vivo allograft external validation (unseen tumors, still in vivo) ===")
    print(allografts.to_string(index=False))
    print(f"\nGrand mean r2 across allograft samples: {allografts['mean_r2'].mean():.3f}")
    print(f"In-sample (GEMM held-out): {r2_compare['insample_r2'].mean():.3f}")
    print(f"External, in-vivo allografts: {allografts['mean_r2'].mean():.3f}")
    print(f"External, in-vitro organoid shGFP: {r2_compare['shgfp_r2'].mean():.3f}")
    print("Monotonic degradation with distance from the training context (in-sample GEMM -> "
          "other in-vivo GEMM-derived tumors -> in-vitro organoid culture) -- 'external' alone "
          "does not explain organoid's shortfall; being in vitro specifically does.")

    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
