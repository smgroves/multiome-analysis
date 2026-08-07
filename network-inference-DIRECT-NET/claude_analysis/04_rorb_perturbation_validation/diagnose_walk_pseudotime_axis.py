"""Same walk-trajectory-over-time view as diagnose_walk_axis_position.py, but the y-axis
is real GEMM cells' own `palantir_pseudotime` (calibrated from real data via
pseudotime_axis_utils.py) instead of a raw %-Hamming-agreement number. Two changes from
that script, both empirically driven, not assumed:

1. Axis poles are NE1(Arc_5)/nonNE1(Arc_4), not Generalist_NE/Generalist_nonNE -- checked
   directly against real GEMM cells' mean pseudotime per archetype and confirmed the
   Generalists are NOT the pseudotime extremes (see pseudotime_axis_utils.py docstring for
   the full table). Re-deriving which 41 (of 53) genes differ between the TRUE extremes.
2. The raw 0-100 %-agreement-with-NE1 position for every walk step is passed through a
   monotonic (isotonic, decreasing) calibration curve fit on every real GEMM cell's own
   (axis position, palantir_pseudotime) pair, so a walk's plotted position is in genuine
   pseudotime units, ordered the same way real cells are, not just a bit-count fraction.

Default (no CLI args) does knockdown-vs-unperturbed only; pass "ascl1" to additionally
overlay the ASCL1-knockout positive control, written to a separate, differently-named
output.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/diagnose_walk_pseudotime_axis.py [ascl1]
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

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/organoid_walks/axis_position_plots/pseudotime_calibrated_axis"
WALK_PATH = f"{DIR_PREFIX}/6667/organoid_seeded/walks/long_walks/4000_step_walks"
ORGANOID_STARTS = {
    "Neuroendocrine1": 5500048749758430,
    "Generalist NE": 7861313821741716,
    "Neuroendocrine2": 17609338899731,
}
STEP_STRIDE = 10
WITH_ASCL1 = "ascl1" in [a.lower() for a in sys.argv[1:]]
OUT_SUFFIX = "_with_ASCL1" if WITH_ASCL1 else ""
# Biological names (Arc_1=Intermediate, Arc_2=nonNE2, Arc_3=Secretory, Arc_4=nonNE1,
# Arc_5=NE1, Arc_6=NE2) for the reference-line labels.
ARCHETYPE_LABELS = {
    "Generalist_NE": "Generalist_NE", "Arc_6": "NE2 (Arc_6)", "Arc_5": "NE1 (Arc_5)",
    "Arc_1": "Intermediate (Arc_1)", "Arc_3": "Secretory (Arc_3)", "Arc_4": "nonNE1 (Arc_4)",
    "Arc_2": "nonNE2 (Arc_2)", "Generalist_nonNE": "Generalist_nonNE",
}


def group_and_space_labels(archetype_calibrated, min_gap):
    """Merges archetypes whose calibrated position is indistinguishable (within min_gap/4
    -- e.g. Generalist_nonNE/Arc_4/Arc_2 all project to ~0% NE1-agreement on this 41-gene
    axis, a genuine property of the discrete metric, not a plotting bug) into one combined
    label at their shared line position. Remaining distinct groups get their TEXT y-nudged
    with a greedy minimum-gap pass so labels don't overlap; the dotted reference line stays
    at each group's true calibrated position regardless of any text nudge."""
    items = sorted(archetype_calibrated.items(), key=lambda kv: kv[1])
    groups = []
    for arc_name, y in items:
        if groups and (y - groups[-1]["y"]) < min_gap / 4:
            groups[-1]["names"].append(arc_name)
            groups[-1]["y"] = np.mean([archetype_calibrated[n] for n in groups[-1]["names"]])
        else:
            groups.append({"names": [arc_name], "y": y})

    text_y_prev = None
    for g in groups:
        g["text_y"] = g["y"] if text_y_prev is None else max(g["y"], text_y_prev + min_gap)
        text_y_prev = g["text_y"]
    return groups


def parse_walk_file(path):
    with open(path) as f:
        return [ast.literal_eval(line.strip()) for line in f if line.strip()]


def position_series(walk, ne1_idx, mask, n_axis_genes, calibrate):
    steps = walk[::STEP_STRIDE]
    raw = np.array([pt.axis_position(s, ne1_idx, mask, n_axis_genes) for s in steps])
    return calibrate(raw)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    ne1_idx, mask, n_axis_genes = pt.load_axis(nodes)
    print(f"NE1<->nonNE1 axis: {n_axis_genes} genes")
    calibrate = pt.fit_calibration(nodes, ne1_idx, mask, n_axis_genes)
    archetype_raw = pt.archetype_axis_positions(nodes, ne1_idx, mask, n_axis_genes)
    archetype_calibrated = {a: calibrate(np.array([p]))[0] for a, p in archetype_raw.items()}
    y_lo, y_hi = min(archetype_calibrated.values()), max(archetype_calibrated.values())
    y_pad = 0.06 * (y_hi - y_lo)

    for label, start_idx in ORGANOID_STARTS.items():
        walks_kd = parse_walk_file(f"{WALK_PATH}/{start_idx}/results_RORA_RORB_kd.csv")
        walks_unpert = parse_walk_file(f"{WALK_PATH}/{start_idx}/results.csv")

        kd_pos = [position_series(w, ne1_idx, mask, n_axis_genes, calibrate) for w in walks_kd]
        unpert_pos = [position_series(w, ne1_idx, mask, n_axis_genes, calibrate) for w in walks_unpert]
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
            ascl1_pos = [position_series(w, ne1_idx, mask, n_axis_genes, calibrate) for w in walks_ascl1]
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

        plt.text(step_axis[-1] * 1.03, y_hi + y_pad, "GEMM archetypes\n(calibrated):", fontsize=8, va="center", ha="left", fontweight="bold")
        label_groups = group_and_space_labels(archetype_calibrated, min_gap=0.045 * (y_hi - y_lo))
        for g in label_groups:
            plt.axhline(g["y"], color="black", linewidth=0.6, linestyle=":", alpha=0.5, zorder=0)
            names = "/".join(ARCHETYPE_LABELS[n] for n in g["names"])
            plt.text(step_axis[-1] * 1.03, g["text_y"], f"GEMM {names}", fontsize=7.5, va="center", ha="left")

        plt.xlabel("Walk step")
        plt.ylabel("Real GEMM cells' palantir pseudotime\n(calibrated from axis position; low=NE1-like, high=nonNE1-like)")
        plt.title(f"organoid '{label}' start: position along NE1<->nonNE1 axis, calibrated to real pseudotime")
        plt.xlim(0, step_axis[-1] * 1.42)
        plt.ylim(y_lo - y_pad, y_hi + 2.2 * y_pad)
        plt.legend(loc="upper left" if kd_mean[0] < (y_lo + y_hi) / 2 else "lower left", fontsize=8.5)

        if WITH_ASCL1:
            stats_text = (
                f"Mann-Whitney U vs. unperturbed (n=100 walks each) -- RORA_RORB kd: final-step "
                f"{kd_final.mean():.3f} vs. {unpert_final.mean():.3f}, p={p_final:.1e}; whole-walk mean "
                f"{kd_walkmean.mean():.3f} vs. {unpert_walkmean.mean():.3f}, p={p_walkmean:.1e}\n"
                f"ASCL1 ko (positive control): final-step {ascl1_final.mean():.3f} vs. {unpert_final.mean():.3f}, "
                f"p={p_final_ascl1:.1e}; whole-walk mean {ascl1_walkmean.mean():.3f} vs. {unpert_walkmean.mean():.3f}, "
                f"p={p_walkmean_ascl1:.1e}"
            )
            rect = [0, 0.10, 1, 1]
            text_y = -0.04
        else:
            stats_text = (
                f"Mann-Whitney U, knockdown vs. unperturbed (n=100 walks each): "
                f"final-step position {kd_final.mean():.3f} vs. {unpert_final.mean():.3f}, p={p_final:.1e}; "
                f"whole-walk mean position {kd_walkmean.mean():.3f} vs. {unpert_walkmean.mean():.3f}, p={p_walkmean:.1e}"
            )
            rect = [0, 0.06, 1, 1]
            text_y = -0.02
        plt.gcf().text(0.5, text_y, stats_text, ha="center", va="top", fontsize=7.5,
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))
        plt.tight_layout(rect=rect)
        for ext in ["png", "pdf"]:
            plt.savefig(f"{OUT_DIR}/walk_pseudotime_axis_{label.replace(' ', '_')}{OUT_SUFFIX}.{ext}", dpi=150, bbox_inches="tight")
        plt.close()
        extra = f", ascl1 mean start={ascl1_mean[0]:.3f}, end={ascl1_mean[-1]:.3f}" if WITH_ASCL1 else ""
        print(f"Wrote walk_pseudotime_axis_{label.replace(' ', '_')}{OUT_SUFFIX}.{{png,pdf}} "
              f"(kd mean start={kd_mean[0]:.3f}, end={kd_mean[-1]:.3f}; "
              f"unpert mean start={unpert_mean[0]:.3f}, end={unpert_mean[-1]:.3f}; "
              f"p_final={p_final:.2e}, p_walkmean={p_walkmean:.2e}{extra})")


if __name__ == "__main__":
    main()
