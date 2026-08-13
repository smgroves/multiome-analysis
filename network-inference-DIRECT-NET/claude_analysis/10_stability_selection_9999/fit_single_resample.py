"""One worker of the stability-selection prototype (barcode 9999). Subsamples GEMM's
training cells WITHOUT replacement at half size (the classic stability-selection /
"subagging" design -- Meinshausen & Buhlmann 2010 -- preferred over a full with-replacement
bootstrap here both statistically, since it avoids duplicate-cell ties in the heat
computation, and computationally, since it's cheaper), refits with the REAL, unmodified
bb.tl.get_rules (default pseudocount_mode="max_heat", pseudocount_target="uniform" -- same
as 6667 -- so this experiment isolates only the pruning-robustness question, not any prior
change), and records which regulators survived pruning for each gene.

Uses the real package function directly, not a standalone reimplementation, specifically to
avoid the class of mistake that confounded barcode 7777 (see 7777/README.md).

Usage: called many times in parallel with different --seed values by
run_stability_selection.sh.
"""
import argparse
import os
import pickle

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
TRAIN_PATH = f"{DIR_PREFIX}/6667/data_split/train_t0combined.csv"
OUT_DIR = f"{DIR_PREFIX}/claude_analysis/10_stability_selection_9999/bootstrap_runs"
SUBSAMPLE_FRAC = 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    out_path = f"{OUT_DIR}/resample_seed{args.seed}.pkl"
    if os.path.exists(out_path):
        print(f"seed {args.seed} already done, skipping")
        return

    graph, vertex_dict = bb.load.load_network(
        f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False
    )
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

    train_data = bb.load.load_data(
        TRAIN_PATH, nodes, norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    rng = np.random.default_rng(args.seed)
    n_sub = int(len(train_data) * SUBSAMPLE_FRAC)
    sub_idx = rng.choice(train_data.index, size=n_sub, replace=False)
    sub_data = train_data.loc[sub_idx]
    print(f"[seed {args.seed}] fitting on {len(sub_data)}/{len(train_data)} cells")

    rules, regulators_dict, strengths, signed_strengths = bb.tl.get_rules(
        data=sub_data, vertex_dict=vertex_dict, plot=False, threshold=0,
    )

    with open(out_path, "wb") as f:
        pickle.dump({"seed": args.seed, "regulators_dict": regulators_dict}, f)
    print(f"[seed {args.seed}] done -> {out_path}")


if __name__ == "__main__":
    main()
