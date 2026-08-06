"""Follow-up to run_organoid_perturbation_walks.py: the basin-occupancy summary showed
organoid-seeded walks mostly move into "None" (not within radius=4 of any of the 5
FILTERED attractor basins) under RORA_RORB knockdown, rather than landing cleanly on a
named alternative. This asks a more sensitive question: even without reaching a basin,
did the walks move SIGNIFICANTLY CLOSER to any of the network's 8 archetype AVERAGE
STATES (6667/attractors/average_states.txt -- includes Arc_1/Arc_2/Arc_4, which were
excluded from the filtered attractor set used for basin-occupancy measurement, but still
have a well-defined theoretical average state to measure distance to)? In particular:
did knockdown move organoid's real starting states significantly closer to Arc_1
(informally labeled "Intermediate" in one unrelated script), even if they never get
within the radius that would count as "reaching" it?

For each organoid starting label x condition (unperturbed / RORA_RORB_kd), reads the
already-computed raw walk trajectories (100 walks x ~4000 steps each) and computes, per
walk, the mean Hamming distance to each of the 8 archetypes across the whole trajectory,
and the minimum (closest approach) distance reached. Compares the 100-walk distributions
between unperturbed and knockdown via Mann-Whitney U per archetype, Bonferroni-corrected
across the 8 archetypes, separately for the mean-distance and min-distance metrics.

Hamming distance is computed via XOR + popcount (`(x ^ y).bit_count()`), mathematically
identical to bb.utils.hamming_idx(x, y, n) for x, y < 2**n, but far faster in pure Python
-- lets this run on the full walk trajectories without subsampling.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/diagnose_walk_archetype_distance.py
"""

import ast
import os

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd
from scipy.stats import mannwhitneyu

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic"
WALK_PATH = f"{DIR_PREFIX}/6667/organoid_seeded/walks/long_walks/4000_step_walks"
ORGANOID_STARTS = {
    "Neuroendocrine1": 5500048749758430,
    "Generalist NE": 7861313821741716,
    "Intermediate": 2308064086923296,
    "Neuroendocrine2": 17609338899731,
}
ALL_ARCHETYPES = ["Generalist_NE", "Generalist_nonNE", "Arc_1", "Arc_2", "Arc_3", "Arc_4", "Arc_5", "Arc_6"]


def load_archetype_indices(nodes):
    avg_states = pd.read_csv(f"{DIR_PREFIX}/6667/attractors/average_states.txt", index_col=0)
    avg_states = avg_states[nodes]  # reorder columns to match nodes' canonical bit order
    indices = {}
    for archetype, row in avg_states.iterrows():
        bits = "".join(str(int(v)) for v in row)
        indices[archetype] = int(bits, 2)
    return indices


def parse_walk_file(path):
    walks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            walks.append(ast.literal_eval(line))
    return walks


def per_walk_distance_stats(walks, archetype_idx):
    means, mins = [], []
    for walk in walks:
        dists = [(state ^ archetype_idx).bit_count() for state in walk]
        means.append(np.mean(dists))
        mins.append(np.min(dists))
    return np.array(means), np.array(mins)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

    archetype_idx = load_archetype_indices(nodes)
    print(f"Archetype average-state indices: {archetype_idx}")

    n_tests = len(ALL_ARCHETYPES)
    bonferroni_alpha = 0.05 / n_tests
    print(f"Bonferroni-corrected alpha per metric: {bonferroni_alpha:.5f} ({n_tests} archetypes tested)")

    all_rows = []
    for label, start_idx in ORGANOID_STARTS.items():
        walks_unpert = parse_walk_file(f"{WALK_PATH}/{start_idx}/results.csv")
        walks_kd = parse_walk_file(f"{WALK_PATH}/{start_idx}/results_RORA_RORB_kd.csv")
        print(f"\n=== {label} (start_idx={start_idx}): {len(walks_unpert)} unperturbed walks, {len(walks_kd)} kd walks ===")

        for archetype, arch_idx in archetype_idx.items():
            mean_unpert, min_unpert = per_walk_distance_stats(walks_unpert, arch_idx)
            mean_kd, min_kd = per_walk_distance_stats(walks_kd, arch_idx)

            _, p_mean = mannwhitneyu(mean_unpert, mean_kd, alternative="two-sided")
            _, p_min = mannwhitneyu(min_unpert, min_kd, alternative="two-sided")

            all_rows.append({
                "organoid_start": label, "archetype": archetype,
                "mean_dist_unperturbed": mean_unpert.mean(), "mean_dist_kd": mean_kd.mean(),
                "mean_dist_shift": mean_kd.mean() - mean_unpert.mean(), "mean_dist_p": p_mean,
                "mean_dist_significant": p_mean < bonferroni_alpha,
                "min_dist_unperturbed": min_unpert.mean(), "min_dist_kd": min_kd.mean(),
                "min_dist_shift": min_kd.mean() - min_unpert.mean(), "min_dist_p": p_min,
                "min_dist_significant": p_min < bonferroni_alpha,
            })

    result = pd.DataFrame(all_rows)
    result.to_csv(f"{OUT_DIR}/walk_archetype_distance_shift.csv", index=False)

    pd.set_option("display.width", 200)
    for label in ORGANOID_STARTS:
        sub = result[result["organoid_start"] == label].sort_values("mean_dist_shift")
        print(f"\n=== {label}: mean/min distance to each archetype, unperturbed vs RORA_RORB_kd ===")
        print(sub[["archetype", "mean_dist_unperturbed", "mean_dist_kd", "mean_dist_shift", "mean_dist_p", "mean_dist_significant",
                    "min_dist_unperturbed", "min_dist_kd", "min_dist_shift", "min_dist_p", "min_dist_significant"]].to_string(index=False))

    print("\n=== Summary: significant movement (Bonferroni-corrected) toward any archetype (negative shift = closer under kd) ===")
    sig_mean = result[result["mean_dist_significant"] & (result["mean_dist_shift"] < 0)]
    sig_min = result[result["min_dist_significant"] & (result["min_dist_shift"] < 0)]
    print("By mean distance across the walk:")
    print(sig_mean[["organoid_start", "archetype", "mean_dist_shift", "mean_dist_p"]].to_string(index=False))
    print("\nBy minimum (closest approach) distance:")
    print(sig_min[["organoid_start", "archetype", "min_dist_shift", "min_dist_p"]].to_string(index=False))

    print(f"\nWrote {OUT_DIR}/walk_archetype_distance_shift.csv")


if __name__ == "__main__":
    main()
