"""Generates ASCL1-knockout long walks as a positive control alongside the RORA_RORB
knockdown walks already used throughout the RadViz/axis-position/temporal-distance plots.
ASCL1 is a canonical NE master regulator -- knocking it out should drive a strong,
unambiguous NE -> nonNE shift, giving a clear "does the walk machinery even detect an
obvious perturbation" baseline to compare RORB's more subtle, hypothesis-specific effect
against.

Runs bb.rw.long_random_walks with off_nodes=["ASCL1"] (writes results_ASCL1_kd.csv
alongside the existing results.csv/results_RORA_RORB_kd.csv in each start directory --
does not touch either existing file) for:
  (A) the same 3 organoid-seeded NE starting states already used in
      diagnose_walk_axis_position.py / diagnose_walk_temporal_ordering.py
      (Neuroendocrine1, Generalist NE, Neuroendocrine2).
  (B) GEMM's own Arc_5 (NE1) basin -- all 11 member states, matching
      plot_radviz_archetype_projection_gemm_native.py's pooling.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/run_ascl1_positive_control_walks.py
"""

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
ORGANOID_WALK_DIR = f"{DIR_PREFIX}/6667/organoid_seeded"
GEMM_WALK_DIR = f"{DIR_PREFIX}/6667"
MAX_STEPS = 4000
ITERS = 100

ORGANOID_NE_STARTS = {
    "Neuroendocrine1": 5500048749758430,
    "Generalist NE": 7861313821741716,
    "Neuroendocrine2": 17609338899731,
}


def main():
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")

    # --- (A) Organoid-seeded ASCL1 knockout walks ---
    organoid_attractor_dict = {label: [idx] for label, idx in ORGANOID_NE_STARTS.items()}
    print(f"=== (A) Organoid-seeded ASCL1 knockout walks: {list(organoid_attractor_dict.keys())} ===")
    bb.rw.long_random_walks(
        list(organoid_attractor_dict.keys()), organoid_attractor_dict, rules, regulators_dict, nodes,
        save_dir=ORGANOID_WALK_DIR, on_nodes=[], off_nodes=["ASCL1"],
        max_steps=MAX_STEPS, iters=ITERS, overwrite_walks=False,
    )

    # --- (B) GEMM-native Arc_5 basin ASCL1 knockout walks (all 11 member states) ---
    gemm_attractor_dict = bb.utils.get_attractor_dict(f"{GEMM_WALK_DIR}/attractors/attractors_threshold_0.5", filtered=True)
    print(f"\n=== (B) GEMM-native ASCL1 knockout walks: Arc_5 ({len(gemm_attractor_dict['Arc_5'])} member states) ===")
    bb.rw.long_random_walks(
        ["Arc_5"], gemm_attractor_dict, rules, regulators_dict, nodes,
        save_dir=GEMM_WALK_DIR, on_nodes=[], off_nodes=["ASCL1"],
        max_steps=MAX_STEPS, iters=ITERS, overwrite_walks=False,
    )

    print("\nDone. New files: results_ASCL1_kd.csv alongside existing results.csv/results_RORA_RORB_kd.csv in:")
    for label, idx in organoid_attractor_dict.items():
        print(f"  {ORGANOID_WALK_DIR}/walks/long_walks/{MAX_STEPS}_step_walks/{idx[0]}/  ({label})")
    for idx in gemm_attractor_dict["Arc_5"]:
        print(f"  {GEMM_WALK_DIR}/walks/long_walks/{MAX_STEPS}_step_walks/{idx}/  (Arc_5 member)")


if __name__ == "__main__":
    main()
