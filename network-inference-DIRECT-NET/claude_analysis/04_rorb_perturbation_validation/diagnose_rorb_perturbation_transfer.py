"""Does GEMM's fitted network correctly predict the DIRECTION of RORB knockdown's effect
in organoid, even though §9/§10 showed absolute leaf-conditional predictions don't
transfer well to organoid? These are different claims: R2/leaf-conditional agreement
tests ABSOLUTE prediction accuracy (does the rule's exact fitted value match organoid's
actual value for a given regulator combination). A perturbation comparison (shGFP vs
shRORB) is a WITHIN-organoid DIFFERENTIAL -- comparing two conditions that share the same
systematic organoid-context shift. If that shift applies similarly to both conditions
(same culture system, same technical pipeline, differing only in the RORB shRNA), it can
cancel out of the difference even though it corrupts the absolute prediction.

Directly simulates GEMM's rule-predicted RORB-knockdown effect using organoid_shGFP's own
cells as the baseline background context (their own actual values for every OTHER
regulator), rather than a coarse net-signed-strength proxy: for every gene with RORA_RORB
as a fitted regulator, compute the rule's predicted value using organoid_shGFP's real
regulator profile (predicted_baseline), then again with RORA_RORB artificially set to 0
(predicted_kd, simulating full knockdown), holding every other regulator at its
organoid_shGFP value. predicted_shift = mean(predicted_kd) - mean(predicted_baseline) is
GEMM's rule's answer to "if you knock down RORA_RORB starting from organoid's own
baseline state, which direction does this gene move" -- compared against the REAL
observed shift (organoid_shRORB1/2's actual mean minus organoid_shGFP's actual mean).

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/diagnose_rorb_perturbation_transfer.py
"""

import os

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd
from scipy.stats import pearsonr

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic_and_organoid_walks"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")

    targets = [g for g, regs in regulators_dict.items() if "RORA_RORB" in regs]
    print(f"{len(targets)} genes have RORA_RORB as a fitted regulator: {targets}")

    shgfp = bb.load.load_data(
        f"{DIR_PREFIX}/data/organoid/adata_organoid_shGFP_v3_RORA_RORB_ave.csv", nodes,
        norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    shrorb1 = bb.load.load_data(
        f"{DIR_PREFIX}/data/organoid/adata_organoid_shRORB1_v3_RORA_RORB_ave.csv", nodes,
        norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    shrorb2 = bb.load.load_data(
        f"{DIR_PREFIX}/data/organoid/adata_organoid_shRORB2_v3_RORA_RORB_ave.csv", nodes,
        norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )

    shgfp_kd = shgfp.copy()
    shgfp_kd["RORA_RORB"] = 0.0

    rows = []
    for gene in targets:
        heat_baseline, _ = bb.tl.parent_heatmap(shgfp, regulators_dict, gene)
        predicted_baseline = (heat_baseline @ rules[gene]).mean()

        heat_kd, _ = bb.tl.parent_heatmap(shgfp_kd, regulators_dict, gene)
        predicted_kd = (heat_kd @ rules[gene]).mean()

        predicted_shift = predicted_kd - predicted_baseline

        real_shift_1 = shrorb1[gene].mean() - shgfp[gene].mean()
        real_shift_2 = shrorb2[gene].mean() - shgfp[gene].mean()

        rows.append({
            "gene": gene, "predicted_shift_gemm_rule": predicted_shift,
            "real_shift_shRORB1_minus_shGFP": real_shift_1,
            "real_shift_shRORB2_minus_shGFP": real_shift_2,
            "sign_agree_shRORB1": np.sign(predicted_shift) == np.sign(real_shift_1),
            "sign_agree_shRORB2": np.sign(predicted_shift) == np.sign(real_shift_2),
        })

    df = pd.DataFrame(rows).sort_values("predicted_shift_gemm_rule")
    df.to_csv(f"{OUT_DIR}/rorb_kd_perturbation_transfer.csv", index=False)

    pd.set_option("display.width", 160)
    print(df.to_string(index=False))

    print(f"\nSign agreement with shRORB1: {df['sign_agree_shRORB1'].sum()}/{len(df)}")
    print(f"Sign agreement with shRORB2: {df['sign_agree_shRORB2'].sum()}/{len(df)}")

    r1, p1 = pearsonr(df["predicted_shift_gemm_rule"], df["real_shift_shRORB1_minus_shGFP"])
    r2, p2 = pearsonr(df["predicted_shift_gemm_rule"], df["real_shift_shRORB2_minus_shGFP"])
    print(f"\nCorrelation between predicted shift and real shRORB1 shift: r={r1:.3f} (p={p1:.4f})")
    print(f"Correlation between predicted shift and real shRORB2 shift: r={r2:.3f} (p={p2:.4f})")

    print(f"\nWrote {OUT_DIR}/rorb_kd_perturbation_transfer.csv")


if __name__ == "__main__":
    main()
