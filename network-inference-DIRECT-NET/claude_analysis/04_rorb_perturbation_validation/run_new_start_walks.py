"""Runs long_random_walks from specific new starting states, each the exact discrete state
of a real GEMM cell at Hamming distance 0 from its archetype's average state (confirmed
directly: at least one real, phenotype-labeled cell achieves each of these average states
exactly -- see session notes). Two independent uses:

(A) GEMM NE1 (Arc_5) start -- unperturbed vs. RORA_RORB knockdown (off_nodes).
(B) GEMM nonNE starts, one per nonNE archetype (Generalist_nonNE, Arc_4/nonNE1,
    Arc_2/nonNE2) -- unperturbed vs. RORA_RORB overexpression (on_nodes), asking whether
    forcing RORB back on from an already-nonNE state pushes back toward NE ("rescue").

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/run_new_start_walks.py
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
SAVE_DIR = f"{DIR_PREFIX}/6667/new_starts_seeded"
MAX_STEPS = 4000
ITERS = 100

# (A) NE1 start: unperturbed + RORA_RORB knockdown
NE1_STARTS = {"GEMM_NE1_real_cell": 8914049663948766}

# (B) nonNE starts (one per archetype): unperturbed + RORA_RORB overexpression
NONNE_STARTS = {
    "GEMM_Generalist_nonNE_real_cell": 1222330852650596,
    "GEMM_Arc_4_nonNE1_real_cell": 3474165076405860,
    "GEMM_Arc_2_nonNE2_real_cell": 3474165059628640,
}


def main():
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")

    print("=== (A) GEMM NE1 real-cell start: unperturbed + RORA_RORB knockdown ===")
    ne1_dict = {label: [idx] for label, idx in NE1_STARTS.items()}
    bb.rw.long_random_walks(
        list(ne1_dict.keys()), ne1_dict, rules, regulators_dict, nodes,
        save_dir=SAVE_DIR, on_nodes=[], off_nodes=["RORA_RORB"],
        max_steps=MAX_STEPS, iters=ITERS, overwrite_walks=False,
    )

    print("\n=== (B) GEMM nonNE real-cell starts: unperturbed + RORA_RORB overexpression ===")
    nonne_dict = {label: [idx] for label, idx in NONNE_STARTS.items()}
    bb.rw.long_random_walks(
        list(nonne_dict.keys()), nonne_dict, rules, regulators_dict, nodes,
        save_dir=SAVE_DIR, on_nodes=["RORA_RORB"], off_nodes=[],
        max_steps=MAX_STEPS, iters=ITERS, overwrite_walks=False,
    )

    print("\nDone. New files under 6667/new_starts_seeded/walks/long_walks/4000_step_walks/<idx>/:")
    for label, idx in {**ne1_dict, **nonne_dict}.items():
        print(f"  {idx[0]}  ({label})")


if __name__ == "__main__":
    main()
