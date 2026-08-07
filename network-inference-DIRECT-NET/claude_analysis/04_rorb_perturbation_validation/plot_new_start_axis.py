"""Pseudotime-calibrated NE1<->nonNE1 axis plot (see pseudotime_axis_utils.py) for each of
the 5 new real-cell-seeded starting points -- see plot_new_start_hexagon.py's docstring for
what each one is. Mirrors diagnose_walk_pseudotime_axis.py's method exactly, just for these
new single-real-cell starts instead of the organoid-population starts.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/plot_new_start_axis.py
"""

import ast
import os

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
NEW_WALK_DIR = f"{DIR_PREFIX}/6667/new_starts_seeded/walks/long_walks/4000_step_walks"
ORGANOID_WALK_DIR = f"{DIR_PREFIX}/6667/organoid_seeded/walks/long_walks/4000_step_walks"
STEP_STRIDE = 10

SPECS = [
    ("gemm_ne1_start", "axis_GEMM_NE1", "GEMM NE1 real cell", NEW_WALK_DIR, 8914049663948766,
     "results_RORA_RORB_kd.csv", "RORA_RORB knockdown", "tab:purple"),
    ("gemm_nonne_start_oe", "axis_GEMM_Generalist_nonNE", "GEMM Generalist_nonNE real cell", NEW_WALK_DIR, 1222330852650596,
     "results_RORA_RORB_act.csv", "RORA_RORB overexpression", "tab:green"),
    ("gemm_nonne_start_oe", "axis_GEMM_Arc4_nonNE1", "GEMM nonNE1 (Arc_4) real cell", NEW_WALK_DIR, 3474165076405860,
     "results_RORA_RORB_act.csv", "RORA_RORB overexpression", "tab:green"),
    ("gemm_nonne_start_oe", "axis_GEMM_Arc2_nonNE2", "GEMM nonNE2 (Arc_2) real cell", NEW_WALK_DIR, 3474165059628640,
     "results_RORA_RORB_act.csv", "RORA_RORB overexpression", "tab:green"),
    ("organoid_shrorb_nonne_oe", "axis_organoid_shRORB_Generalist_nonNE", "organoid shRORB Generalist_nonNE", ORGANOID_WALK_DIR, 1371881672649956,
     "results_RORA_RORB_act.csv", "RORA_RORB overexpression (rescue?)", "tab:green"),
]


def parse_walk_file(path):
    with open(path) as f:
        return [ast.literal_eval(line.strip()) for line in f if line.strip()]


def position_series(walk, ne1_idx, mask, n_axis_genes, calibrate):
    steps = walk[::STEP_STRIDE]
    raw = np.array([pt.axis_position(s, ne1_idx, mask, n_axis_genes) for s in steps])
    return calibrate(raw)


def main():
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    ne1_idx, mask, n_axis_genes = pt.load_axis(nodes)
    calibrate = pt.fit_calibration(nodes, ne1_idx, mask, n_axis_genes, verbose=False)
    archetype_raw = pt.archetype_axis_positions(nodes, ne1_idx, mask, n_axis_genes)
    archetype_calibrated = {a: calibrate(np.array([p]))[0] for a, p in archetype_raw.items()}
    y_lo, y_hi = min(archetype_calibrated.values()), max(archetype_calibrated.values())
    y_pad = 0.06 * (y_hi - y_lo)

    for out_subdir, out_name, label, walk_dir, start_idx, pert_file, pert_display, pert_color in SPECS:
        out_dir = f"{DIR_PREFIX}/comparisons/organoid_walks/{out_subdir}"
        os.makedirs(out_dir, exist_ok=True)

        walks_pert = parse_walk_file(f"{walk_dir}/{start_idx}/{pert_file}")
        walks_unpert = parse_walk_file(f"{walk_dir}/{start_idx}/results.csv")

        pert_pos = [position_series(w, ne1_idx, mask, n_axis_genes, calibrate) for w in walks_pert]
        unpert_pos = [position_series(w, ne1_idx, mask, n_axis_genes, calibrate) for w in walks_unpert]
        step_axis = np.arange(len(pert_pos[0])) * STEP_STRIDE

        pert_mean = np.mean(pert_pos, axis=0)
        unpert_mean = np.mean(unpert_pos, axis=0)
        pert_final = np.array([c[-1] for c in pert_pos])
        unpert_final = np.array([c[-1] for c in unpert_pos])
        pert_walkmean = np.array([c.mean() for c in pert_pos])
        unpert_walkmean = np.array([c.mean() for c in unpert_pos])
        _, p_final = mannwhitneyu(pert_final, unpert_final, alternative="two-sided")
        _, p_walkmean = mannwhitneyu(pert_walkmean, unpert_walkmean, alternative="two-sided")

        plt.figure(figsize=(9.5, 5.5))
        for curve in pert_pos:
            plt.plot(step_axis, curve, color=pert_color, alpha=0.08, linewidth=0.8, zorder=1)
        plt.plot(step_axis, pert_mean, color=pert_color, linewidth=2.5,
                  label=f"{label}, {pert_display} (mean)", zorder=3)
        for curve in unpert_pos:
            plt.plot(step_axis, curve, color="0.6", alpha=0.06, linewidth=0.8, zorder=1)
        plt.plot(step_axis, unpert_mean, color="0.5", linewidth=2.5, linestyle="--",
                  label=f"{label}, unperturbed (mean)", zorder=2)

        plt.text(step_axis[-1] * 1.03, y_hi + y_pad, "GEMM archetypes\n(calibrated):", fontsize=8, va="center", ha="left", fontweight="bold")
        for g in group_and_space_labels(archetype_calibrated, min_gap=0.045 * (y_hi - y_lo)):
            plt.axhline(g["y"], color="black", linewidth=0.6, linestyle=":", alpha=0.5, zorder=0)
            names = "/".join(ARCHETYPE_LABELS[n] for n in g["names"])
            plt.text(step_axis[-1] * 1.03, g["text_y"], f"GEMM {names}", fontsize=7.5, va="center", ha="left")

        plt.xlabel("Walk step")
        plt.ylabel("Real GEMM cells' palantir pseudotime\n(calibrated from axis position; low=NE1-like, high=nonNE1-like)")
        plt.title(f"{label}: position along NE1<->nonNE1 axis, calibrated to real pseudotime")
        plt.xlim(0, step_axis[-1] * 1.42)
        plt.ylim(y_lo - y_pad, y_hi + 2.2 * y_pad)
        plt.legend(loc="upper left" if pert_mean[0] < (y_lo + y_hi) / 2 else "lower left", fontsize=8.5)

        stats_text = (
            f"Mann-Whitney U, {pert_display} vs. unperturbed (n=100 walks each): "
            f"final-step {pert_final.mean():.3f} vs. {unpert_final.mean():.3f}, p={p_final:.1e}; "
            f"whole-walk mean {pert_walkmean.mean():.3f} vs. {unpert_walkmean.mean():.3f}, p={p_walkmean:.1e}"
        )
        plt.gcf().text(0.5, -0.02, stats_text, ha="center", va="top", fontsize=7.5,
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        for ext in ["png", "pdf"]:
            plt.savefig(f"{out_dir}/{out_name}.{ext}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out_dir}/{out_name}.{{png,pdf}} (pert mean start={pert_mean[0]:.3f}, end={pert_mean[-1]:.3f}; "
              f"unpert mean start={unpert_mean[0]:.3f}, end={unpert_mean[-1]:.3f}; p_final={p_final:.2e}, p_walkmean={p_walkmean:.2e})")


if __name__ == "__main__":
    main()
