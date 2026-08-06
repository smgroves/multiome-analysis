"""(C) of the RORB-kd validation plan (/Users/xpz5km/.claude/plans/it-is-looking-like-elegant-patterson.md):
independent, non-simulation check of whether organoid's real experimental RORB knockdown
(condition: shGFP vs shRORB1/shRORB2) shows a statistically real shift from NE-ness scores
toward Intermediate-ness scores, in organoid's own Seurat-based metadata
(data/organoid/organoid_clusters.csv). Fully decoupled from BoBa-T's rules -- a sanity
check of the premise itself, separate from whether GEMM's simulated dynamics reproduce it
(see run_organoid_perturbation_walks.py for that side).

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/validate_organoid_rorb_archetype_shift.py
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic_and_organoid_walks"

NE_SCORE_COLS = ["prediction.score.Generalist.NE", "NE1_score1", "NE2_score1"]
INTERMEDIATE_SCORE_COLS = ["prediction.score.Intermediate", "Intermediate_score1"]
CONDITIONS_TO_COMPARE = [("shGFP", "shRORB1"), ("shGFP", "shRORB2")]
NEGATIVE_CONTROL_SEED = 1234


def run_comparisons(clusters, group_col, group_a, group_b, score_cols, label):
    rows = []
    a = clusters.loc[clusters[group_col] == group_a]
    b = clusters.loc[clusters[group_col] == group_b]
    for col in score_cols:
        stat, p = mannwhitneyu(a[col].dropna(), b[col].dropna(), alternative="two-sided")
        rows.append({
            "comparison": label, "score": col,
            f"mean_{group_a}": a[col].mean(), f"mean_{group_b}": b[col].mean(),
            "diff": b[col].mean() - a[col].mean(), "mannwhitney_p": p,
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    clusters = pd.read_csv(f"{DIR_PREFIX}/data/organoid/organoid_clusters.csv", index_col=0)

    all_score_cols = NE_SCORE_COLS + INTERMEDIATE_SCORE_COLS
    n_tests_per_comparison = len(all_score_cols)
    bonferroni_alpha = 0.05 / n_tests_per_comparison
    print(f"Bonferroni-corrected alpha per comparison: {bonferroni_alpha:.5f} ({n_tests_per_comparison} score columns tested)")

    all_results = []
    for group_a, group_b in CONDITIONS_TO_COMPARE:
        res = run_comparisons(clusters, "condition", group_a, group_b, all_score_cols, f"{group_a}_vs_{group_b}")
        res["significant_bonferroni"] = res["mannwhitney_p"] < bonferroni_alpha
        res["ne_vs_intermediate"] = res["score"].apply(lambda c: "NE" if c in NE_SCORE_COLS else "Intermediate")
        all_results.append(res)

    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv(f"{OUT_DIR}/organoid_rorb_archetype_shift.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n=== NE-ness and Intermediate-ness scores: shGFP vs shRORB1/shRORB2 ===")
    print(result_df.to_string(index=False))

    print("\n=== Interpretation ===")
    ne_rows = result_df[result_df["ne_vs_intermediate"] == "NE"]
    int_rows = result_df[result_df["ne_vs_intermediate"] == "Intermediate"]
    print(f"NE-ness scores: {(ne_rows['diff'] < 0).sum()}/{len(ne_rows)} comparisons show a DECREASE "
          f"in shRORB vs shGFP (expected direction if RORB kd moves cells away from NE)")
    print(f"Intermediate-ness scores: {(int_rows['diff'] > 0).sum()}/{len(int_rows)} comparisons show an INCREASE "
          f"in shRORB vs shGFP (expected direction if RORB kd moves cells toward Intermediate)")
    print(f"Significant (Bonferroni-corrected) results: {result_df['significant_bonferroni'].sum()}/{len(result_df)}")

    # predicted.id categorical proportions per condition
    print("\n=== predicted.id proportions per condition ===")
    proportions = pd.crosstab(clusters["predicted.id"], clusters["condition"], normalize="columns")
    print(proportions.round(4).to_string())
    proportions.to_csv(f"{OUT_DIR}/organoid_predicted_id_proportions_by_condition.csv")

    # Negative control: split shGFP randomly into two halves, run the same tests -- should
    # NOT show significant differences if the test/pipeline is behaving correctly.
    print("\n=== Negative control: shGFP split randomly into two halves ===")
    rng = np.random.default_rng(NEGATIVE_CONTROL_SEED)
    shgfp = clusters[clusters["condition"] == "shGFP"].copy()
    half_mask = rng.random(len(shgfp)) < 0.5
    shgfp["condition"] = np.where(half_mask, "shGFP_half1", "shGFP_half2")
    neg_control = run_comparisons(shgfp, "condition", "shGFP_half1", "shGFP_half2", all_score_cols, "shGFP_half1_vs_half2")
    neg_control["significant_bonferroni"] = neg_control["mannwhitney_p"] < bonferroni_alpha
    print(neg_control.to_string(index=False))
    print(f"Negative control significant results (should be ~0): {neg_control['significant_bonferroni'].sum()}/{len(neg_control)}")

    print(f"\nWrote {OUT_DIR}/organoid_rorb_archetype_shift.csv, "
          f"{OUT_DIR}/organoid_predicted_id_proportions_by_condition.csv")


if __name__ == "__main__":
    main()
