"""Same raw %-Hamming-agreement axis view as diagnose_walk_axis_position.py, but poled on
NE1(Arc_5)/nonNE1(Arc_4) -- the real pseudotime extremes, per pseudotime_axis_utils.py's
face-validity check -- instead of the Generalists, and with NO palantir calibration
applied (unlike diagnose_walk_pseudotime_axis.py). Purpose: isolate whether the "smushing"
of several archetypes onto nearly the same value in the calibrated plot is an artifact of
the calibration curve (binned isotonic regression) or is already present in the raw
41-gene Hamming metric before any calibration.

Answer, checked directly on the 8 archetypes' own average states (see
pseudotime_axis_utils.py's own face-validity table): `Generalist_nonNE`/`Arc_4`/`Arc_2`
are already collapsed to 0-2.4% (nearly identical) in the RAW metric, before any
calibration -- these 3 archetypes' own average states simply don't differ from the Arc_4
pole on these particular 41 genes, even though they differ from each other/Arc_4 on other
genes outside this axis. So removing calibration does NOT un-smush that group. The
NE-side archetypes (`Arc_5`/`Generalist_NE`/`Arc_6` at 100/97.6/90.2, `Arc_1`/`Arc_3` at
68.3/56.1) were already fairly separated in the raw metric too -- whether the calibration
step compressed them further is exactly what this plot's reference lines show directly,
compared against diagnose_walk_pseudotime_axis.py's calibrated version.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/diagnose_walk_axis_position_ne1nonne1_raw.py [ascl1]
"""

import ast
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
from scipy.stats import mannwhitneyu

import pseudotime_axis_utils as pt
from diagnose_walk_pseudotime_axis import group_and_space_labels, ARCHETYPE_LABELS

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/organoid_walks/axis_position_plots/ne1_nonne1_raw_axis"
WALK_PATH = f"{DIR_PREFIX}/6667/organoid_seeded/walks/long_walks/4000_step_walks"
ORGANOID_STARTS = {
    "Neuroendocrine1": 5500048749758430,
    "Generalist NE": 7861313821741716,
    "Neuroendocrine2": 17609338899731,
}
STEP_STRIDE = 10
WITH_ASCL1 = "ascl1" in [a.lower() for a in sys.argv[1:]]
OUT_SUFFIX = "_with_ASCL1" if WITH_ASCL1 else ""


def parse_walk_file(path):
    with open(path) as f:
        return [ast.literal_eval(line.strip()) for line in f if line.strip()]


def position_series(walk, ne1_idx, mask, n_axis_genes):
    steps = walk[::STEP_STRIDE]
    return np.array([pt.axis_position(s, ne1_idx, mask, n_axis_genes) for s in steps])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    ne1_idx, mask, n_axis_genes = pt.load_axis(nodes)
    print(f"NE1(Arc_5)<->nonNE1(Arc_4) axis: {n_axis_genes} genes, NO palantir calibration (raw %-agreement)")

    archetype_positions = pt.archetype_axis_positions(nodes, ne1_idx, mask, n_axis_genes)
    print("Archetype raw axis positions (0=nonNE1-like, 100=NE1-like):")
    for a, p in sorted(archetype_positions.items(), key=lambda kv: -kv[1]):
        print(f"  {a}: {p:.1f}")

    for label, start_idx in ORGANOID_STARTS.items():
        walks_kd = parse_walk_file(f"{WALK_PATH}/{start_idx}/results_RORA_RORB_kd.csv")
        walks_unpert = parse_walk_file(f"{WALK_PATH}/{start_idx}/results.csv")

        kd_pos = [position_series(w, ne1_idx, mask, n_axis_genes) for w in walks_kd]
        unpert_pos = [position_series(w, ne1_idx, mask, n_axis_genes) for w in walks_unpert]
        step_axis = np.arange(len(kd_pos[0])) * STEP_STRIDE

        kd_mean = np.mean(kd_pos, axis=0)
        unpert_mean = np.mean(unpert_pos, axis=0)
        kd_final = np.array([c[-1] for c in kd_pos])
        unpert_final = np.array([c[-1] for c in unpert_pos])
        kd_walkmean = np.array([c.mean() for c in kd_pos])
        unpert_walkmean = np.array([c.mean() for c in unpert_pos])
        _, p_final = mannwhitneyu(kd_final, unpert_final, alternative="two-sided")
        _, p_walkmean = mannwhitneyu(kd_walkmean, unpert_walkmean, alternative="two-sided")

        if WITH_ASCL1:
            walks_ascl1 = parse_walk_file(f"{WALK_PATH}/{start_idx}/results_ASCL1_kd.csv")
            ascl1_pos = [position_series(w, ne1_idx, mask, n_axis_genes) for w in walks_ascl1]
            ascl1_mean = np.mean(ascl1_pos, axis=0)
            ascl1_final = np.array([c[-1] for c in ascl1_pos])
            ascl1_walkmean = np.array([c.mean() for c in ascl1_pos])
            _, p_final_ascl1 = mannwhitneyu(ascl1_final, unpert_final, alternative="two-sided")
            _, p_walkmean_ascl1 = mannwhitneyu(ascl1_walkmean, unpert_walkmean, alternative="two-sided")

        plt.figure(figsize=(9.5, 5.5))
        if WITH_ASCL1:
            for curve in ascl1_pos:
                plt.plot(step_axis, curve, color="tab:red", alpha=0.08, linewidth=0.8, zorder=1)
            plt.plot(step_axis, ascl1_mean, color="tab:red", linewidth=2.5,
                      label=f"organoid '{label}' start, ASCL1 knockout (positive control, mean)", zorder=3)
        for curve in kd_pos:
            plt.plot(step_axis, curve, color="tab:purple", alpha=0.08, linewidth=0.8, zorder=1)
        plt.plot(step_axis, kd_mean, color="tab:purple", linewidth=2.5,
                  label=f"organoid '{label}' start, RORA_RORB knockdown (mean)", zorder=3)
        for curve in unpert_pos:
            plt.plot(step_axis, curve, color="0.6", alpha=0.06, linewidth=0.8, zorder=1)
        plt.plot(step_axis, unpert_mean, color="0.5", linewidth=2.5, linestyle="--",
                  label=f"organoid '{label}' start, unperturbed (negative control, mean)", zorder=2)

        plt.text(step_axis[-1] * 1.03, 108, "GEMM archetypes\n(raw, no calibration):", fontsize=8, va="center", ha="left", fontweight="bold")
        for g in group_and_space_labels(archetype_positions, min_gap=4.5):
            plt.axhline(g["y"], color="black", linewidth=0.6, linestyle=":", alpha=0.5, zorder=0)
            names = "/".join(ARCHETYPE_LABELS[n] for n in g["names"])
            plt.text(step_axis[-1] * 1.03, g["text_y"], f"GEMM {names}", fontsize=7.5, va="center", ha="left")

        plt.xlabel("Walk step")
        plt.ylabel("Position on NE1 <-> nonNE1 axis\n(%agreement with NE1/Arc_5 on the 41 differentiating genes; NO pseudotime calibration)")
        plt.title(f"organoid '{label}' start: position along NE1<->nonNE1 axis (raw %, uncalibrated)")
        plt.xlim(0, step_axis[-1] * 1.42)
        plt.ylim(-5, 112)
        plt.legend(loc="lower left", fontsize=8.5)

        if WITH_ASCL1:
            stats_text = (
                f"Mann-Whitney U vs. unperturbed (n=100 walks each) -- RORA_RORB kd: final-step "
                f"{kd_final.mean():.1f} vs. {unpert_final.mean():.1f}, p={p_final:.1e}; whole-walk mean "
                f"{kd_walkmean.mean():.1f} vs. {unpert_walkmean.mean():.1f}, p={p_walkmean:.1e}\n"
                f"ASCL1 ko (positive control): final-step {ascl1_final.mean():.1f} vs. {unpert_final.mean():.1f}, "
                f"p={p_final_ascl1:.1e}; whole-walk mean {ascl1_walkmean.mean():.1f} vs. {unpert_walkmean.mean():.1f}, "
                f"p={p_walkmean_ascl1:.1e}"
            )
            rect, text_y = [0, 0.10, 1, 1], -0.04
        else:
            stats_text = (
                f"Mann-Whitney U, knockdown vs. unperturbed (n=100 walks each): "
                f"final-step position {kd_final.mean():.1f} vs. {unpert_final.mean():.1f}, p={p_final:.1e}; "
                f"whole-walk mean position {kd_walkmean.mean():.1f} vs. {unpert_walkmean.mean():.1f}, p={p_walkmean:.1e}"
            )
            rect, text_y = [0, 0.06, 1, 1], -0.02
        plt.gcf().text(0.5, text_y, stats_text, ha="center", va="top", fontsize=7.5,
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))
        plt.tight_layout(rect=rect)
        for ext in ["png", "pdf"]:
            plt.savefig(f"{OUT_DIR}/walk_axis_position_ne1nonne1_raw_{label.replace(' ', '_')}{OUT_SUFFIX}.{ext}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote walk_axis_position_ne1nonne1_raw_{label.replace(' ', '_')}{OUT_SUFFIX}.{{png,pdf}} "
              f"(kd mean start={kd_mean[0]:.1f}, end={kd_mean[-1]:.1f}; "
              f"unpert mean start={unpert_mean[0]:.1f}, end={unpert_mean[-1]:.1f}; "
              f"p_final={p_final:.2e}, p_walkmean={p_walkmean:.2e})")


if __name__ == "__main__":
    main()
