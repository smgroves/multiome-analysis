"""Aggregate the stability-selection bootstrap runs (fit_single_resample.py x N) into a
final regulator set per gene, then refit final rule VALUES on the FULL (non-subsampled)
training data restricted to that stable set.

Stability selection (Meinshausen & Buhlmann 2010): a candidate regulator is kept for a gene
only if it survived bb.tl.get_rules's own pruning (detect_irrelevant_regulator) in at least
STABILITY_THRESHOLD fraction of the subsample fits. This targets exactly the failure mode
found in the 7777 postmortem: the discrete keep/prune decision is a one-shot call on a
single dataset, so it's sensitive to whatever noise happens to be in that one fit (in that
case, a network-loading flag; here, ordinary single-cell sampling noise). Averaging the
decision over many resamples should make it robust to noise unless a regulator's relevance
is genuinely and consistently near the pruning threshold.

The final refit is done via a graph-surgery trick to reuse the REAL, unmodified
bb.tl.get_rules rather than reimplementing its math: load the network normally
(remove_selfloops=True, matching 6667), then remove the graph edges for any
(candidate_regulator -> gene) pair that didn't survive stability selection, and call
get_rules once on the pruned graph with the FULL training data. Any gene whose stable set
is empty falls back to get_rules's own existing self-fallback (regulators=[gene]) -- the
same mechanism 6667 already relies on for genuinely under-regulated genes, not new behavior.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/10_stability_selection_9999/aggregate_and_refit_9999.py
"""
import glob
import pickle

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
TRAIN_PATH = f"{DIR_PREFIX}/6667/data_split/train_t0combined.csv"
OUT_RULES_PATH = f"{DIR_PREFIX}/9999/rules/rules_9999.txt"
BOOTSTRAP_DIR = f"{DIR_PREFIX}/claude_analysis/10_stability_selection_9999/bootstrap_runs"
STABILITY_THRESHOLD = 0.5


def main():
    graph, vertex_dict = bb.load.load_network(
        f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False
    )
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    v_names_map = {v: k for k, v in vertex_dict.items()}

    bootstrap_files = sorted(glob.glob(f"{BOOTSTRAP_DIR}/resample_seed*.pkl"))
    print(f"Found {len(bootstrap_files)} completed bootstrap resamples")
    bootstrap_regulators = []
    for f in bootstrap_files:
        with open(f, "rb") as fh:
            bootstrap_regulators.append(pickle.load(fh)["regulators_dict"])
    n_boot = len(bootstrap_regulators)

    selection_freq_rows = []
    stable_sets = {}
    for gene in nodes:
        candidates = [v_names_map[v] for v in vertex_dict[gene].in_neighbors()]
        stable = set()
        for cand in candidates:
            freq = sum(1 for rd in bootstrap_regulators if cand in rd.get(gene, [])) / n_boot if n_boot else 0.0
            selection_freq_rows.append({"gene": gene, "candidate_regulator": cand, "selection_freq": freq})
            if freq >= STABILITY_THRESHOLD:
                stable.add(cand)
        stable_sets[gene] = stable

    freq_df = pd.DataFrame(selection_freq_rows)
    freq_df.to_csv(f"{DIR_PREFIX}/claude_analysis/10_stability_selection_9999/selection_frequencies.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n=== Genes where stability selection changed the candidate set vs. taking all network edges ===")
    for gene in nodes:
        candidates = set(v_names_map[v] for v in vertex_dict[gene].in_neighbors())
        if stable_sets[gene] != candidates:
            dropped = candidates - stable_sets[gene]
            print(f"{gene}: dropped {sorted(dropped)} (kept {sorted(stable_sets[gene])})")

    # Graph surgery: remove edges for candidates that didn't survive stability selection.
    for gene in nodes:
        v_gene = vertex_dict[gene]
        candidates = [v_names_map[v] for v in v_gene.in_neighbors()]
        for cand in candidates:
            if cand not in stable_sets[gene]:
                v_cand = vertex_dict[cand]
                e = graph.edge(v_cand, v_gene)
                if e is not None:
                    graph.remove_edge(e)

    print("\nLoading full GEMM training data for final refit...")
    train_data = bb.load.load_data(
        TRAIN_PATH, nodes, norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    rules, regulators_dict, strengths, signed_strengths = bb.tl.get_rules(
        data=train_data, vertex_dict=vertex_dict, plot=False, threshold=0,
    )
    bb.tl.save_rules(rules, regulators_dict, fname=OUT_RULES_PATH)
    print(f"\nWrote {OUT_RULES_PATH}")

    # Compare final regulator sets vs 6667's.
    rules_6667, regulators_dict_6667 = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")
    n_same_set = sum(1 for g in nodes if set(regulators_dict[g]) == set(regulators_dict_6667.get(g, [])))
    print(f"\nFinal (9999) regulator sets identical to 6667: {n_same_set}/{len(nodes)}")


if __name__ == "__main__":
    main()
