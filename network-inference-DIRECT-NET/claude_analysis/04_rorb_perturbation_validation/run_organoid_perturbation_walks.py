"""(B) of the RORB-kd validation plan (/Users/xpz5km/.claude/plans/it-is-looking-like-elegant-patterson.md):
run BoBa-T's actual perturbation/random-walk machinery (bb.rw.long_random_walks -- the
discrete async Gillespie-style walk, NOT the continuous parent_heatmap validation
mechanism) seeded from organoid_shGFP's OWN real cells, using GEMM's fitted rule set
(rules_6667.txt), with and without the RORA_RORB knockdown clamp. Measures resulting
basin occupancy against GEMM's fixed 5-basin filtered attractor set
(Generalist_NE, Generalist_nonNE, Arc_3, Arc_5, Arc_6) -- the same basins used for every
other perturbation/walk analysis on this network -- to see whether organoid's real
states, walked forward under GEMM's dynamics, reproduce the same basin-occupancy shift
GEMM's own Arc_5/Arc_3/Arc_6-seeded walks already show (6667/walks/long_walks/4000_step_walks/).

Starting states: organoid_shGFP cells grouped by organoid's OWN predicted.id label
(Seurat-based archetype calls), binarized (bb.proc.binarize_data) and collapsed to one
representative discrete state per label (bb.tl.find_avg_states) -- lets us directly ask
"starting from cells organoid itself calls NE-like, does GEMM's RORB-kd dynamics move
them toward organoid's own Intermediate-like region."

Caveat (documented, not worked around): make_percentage_popd_df's attractor_dict serves
two roles at once (finding the starting index AND defining the measurement grid), so the
organoid-derived starting points are necessarily also present as extra columns in the
output -- these represent "still near its own starting point," not a GEMM attractor, and
should not be over-interpreted as attractors reached. Only the 5 genuine GEMM basin
columns (+ "None") are meaningful for the comparison this script is designed to make.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/run_organoid_perturbation_walks.py
"""

import os
import sys

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd

sys.path.insert(0, "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET")
import bb_utils

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/organoid_walks"
ORGANOID_WALK_DIR = f"{DIR_PREFIX}/6667/organoid_seeded"
GEMM_WALK_DIR = f"{DIR_PREFIX}/6667"
MIN_CELLS_PER_GROUP = 20
MAX_STEPS = 4000
ITERS = 100
RADIUS = 4
NUM_WALKS = 100
GEMM_BASINS = ["Generalist_NE", "Generalist_nonNE", "Arc_3", "Arc_5", "Arc_6"]


def main():
    os.makedirs(ORGANOID_WALK_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")

    # --- 1. Load organoid_shGFP and binarize per predicted.id group ---
    data = bb.load.load_data(
        f"{DIR_PREFIX}/data/organoid/adata_organoid_shGFP_v3_RORA_RORB_ave.csv", nodes,
        norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    clusters = pd.read_csv(f"{DIR_PREFIX}/data/organoid/organoid_clusters.csv", index_col=0)
    clusters = clusters.reindex(data.index)

    group_sizes = clusters["predicted.id"].value_counts()
    usable_groups = group_sizes[group_sizes >= MIN_CELLS_PER_GROUP].index.tolist()
    print(f"predicted.id group sizes (organoid_shGFP):\n{group_sizes}")
    print(f"\nUsing groups with >= {MIN_CELLS_PER_GROUP} cells: {usable_groups}")

    phenotype_labels = pd.DataFrame({"class": clusters["predicted.id"]}, index=data.index)
    phenotype_labels = phenotype_labels[phenotype_labels["class"].isin(usable_groups)]
    data_for_binarize = data.loc[phenotype_labels.index]

    binarized = bb.proc.binarize_data(data_for_binarize, phenotype_labels=phenotype_labels, threshold=0.5)
    for k, v in binarized.items():
        print(f"  {k}: {len(v)} unique binarized states")

    # --- 2. Collapse each group to one representative discrete starting state ---
    organoid_avg_states = bb.tl.find_avg_states(binarized, nodes, save_dir=ORGANOID_WALK_DIR)
    print(f"\nOrganoid-derived starting states (idx): {organoid_avg_states}")

    # --- 3. Combine with GEMM's fixed 5-basin attractor_dict (measurement grid + start lookup) ---
    gemm_attractor_dict = bb.utils.get_attractor_dict(f"{GEMM_WALK_DIR}/attractors/attractors_threshold_0.5", filtered=True)
    print(f"\nGEMM basins: {list(gemm_attractor_dict.keys())}")

    organoid_start_dict = {k: [v] for k, v in organoid_avg_states.items()}
    combined_dict = {**gemm_attractor_dict, **organoid_start_dict}

    # Face-validity check: which GEMM basin is each organoid starting state nearest to at t=0?
    print("\n=== Face-validity: nearest GEMM basin to each organoid starting state (Hamming distance) ===")
    for label, idx in organoid_avg_states.items():
        dists = {basin: min(bb.utils.hamming_idx(idx, b_idx, len(nodes)) for b_idx in gemm_attractor_dict[basin])
                 for basin in GEMM_BASINS}
        nearest = min(dists, key=dists.get)
        print(f"  organoid '{label}' (idx={idx}) -> nearest GEMM basin: {nearest} (distance={dists[nearest]}); all distances: {dists}")

    # --- 4. Run long_random_walks: unperturbed + RORA_RORB knockdown, same params as GEMM's own runs ---
    # long_random_walks uses plain os.mkdir (not makedirs) for "walks/long_walks/" -- the
    # immediate parent must already exist.
    os.makedirs(f"{ORGANOID_WALK_DIR}/walks", exist_ok=True)
    print(f"\nRunning long_random_walks for {list(organoid_start_dict.keys())}...")
    bb.rw.long_random_walks(
        list(organoid_start_dict.keys()), combined_dict, rules, regulators_dict, nodes,
        save_dir=ORGANOID_WALK_DIR, on_nodes=[], off_nodes=["RORA_RORB"],
        max_steps=MAX_STEPS, iters=ITERS, overwrite_walks=False,
    )

    # --- 5. Aggregate basin occupancy ---
    walk_path = f"{ORGANOID_WALK_DIR}/walks/long_walks/{MAX_STEPS}_step_walks"
    results = {}
    for label in organoid_start_dict.keys():
        for perturbation in [None, "RORA_RORB_kd"]:
            df = bb_utils.make_percentage_popd_df(
                walk_path, label, NUM_WALKS, RADIUS, combined_dict, len(nodes), perturbation=perturbation,
            )
            key = f"{label}__{perturbation or 'unperturbed'}"
            results[key] = df

    # --- 6. Report: mean % of walk steps spent in each GEMM basin, unperturbed vs knockdown ---
    print(f"\n=== Mean %% of {MAX_STEPS}-step walk spent in each GEMM basin (organoid-seeded) ===")
    summary_rows = []
    for label in organoid_start_dict.keys():
        unpert = results[f"{label}__unperturbed"]
        kd = results[f"{label}__RORA_RORB_kd"]
        for basin in GEMM_BASINS + ["None"]:
            pct_unpert = 100 * unpert[basin].mean() / MAX_STEPS
            pct_kd = 100 * kd[basin].mean() / MAX_STEPS
            summary_rows.append({
                "organoid_start": label, "basin": basin,
                "pct_time_unperturbed": pct_unpert, "pct_time_RORA_RORB_kd": pct_kd,
                "shift": pct_kd - pct_unpert,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{OUT_DIR}/organoid_seeded_walk_basin_occupancy.csv", index=False)
    pd.set_option("display.width", 160)
    print(summary_df.to_string(index=False))

    print(f"\nWrote outputs to {ORGANOID_WALK_DIR}/walks/long_walks/{MAX_STEPS}_step_walks/ "
          f"and {OUT_DIR}/organoid_seeded_walk_basin_occupancy.csv")


if __name__ == "__main__":
    main()
