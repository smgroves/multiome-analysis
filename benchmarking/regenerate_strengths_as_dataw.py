"""Switch boba-T's default edge-weight output from sum-based to data-weighted (dataw) --
see benchmarking/README.md ("Alternative edge-weight summaries") for why: dataw_signed was
the one alternative that was consistently competitive (never worst) across the real
networks tested, and the user has decided to make it the default going forward.

Overwrites, for every existing boba-T run below, IN PLACE:
    rules/strengths.csv         (was sum_abs)        -> now dataw_abs
    rules/signed_strengths.csv  (was sum_signed)      -> now dataw_signed

Does NOT touch rules_<run>.txt (the actual fitted truth tables -- the source of truth) or
edge_weights.csv (draw_grn's separate plotting-oriented output) -- both are untouched, so
this is fully reversible by rerunning with sum_abs/sum_signed if ever needed. Reuses
comparison_edge_weight_summaries_6667.py's edge_summaries_for_gene/leaf_weights_from_data
directly (same computation already used for the README's dataw_* results) -- no boba-T
refit, just a different collapse of the already-fitted rule.

One correctness note: comparison_edge_weight_summaries_6667.py's build_all_matrices()
skips genes whose only "regulator" is themselves (boba-T's fallback when no real regulator
survives -- regulators_dict[gene] == [gene]), since that's not a real fitted edge worth
comparing across methods. But those rows DO exist in the original strengths.csv/
signed_strengths.csv (a self-loop weight), so this script computes them too --
mathematically dataw collapses to the same value as sum/mean/max for a single-regulator
(n=1) rule (there's exactly one context, so summing/averaging/maxing/data-weighting one
number all return that number), which this script's --verify step below confirms directly
against the existing files rather than assuming it.

Run in any env with pandas/numpy (same requirement as comparison_edge_weight_summaries_6667.py).
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from comparison_edge_weight_summaries_6667 import edge_summaries_for_gene, load_rules

REPO = "/Users/xpz5km/Documents/GitHub/multiome-analysis"

# run label -> (rules_path, data_path, strengths_dir)
RUNS = {
    "6667": (
        f"{REPO}/network-inference-DIRECT-NET/6667/rules/rules_6667.txt",
        f"{REPO}/network-inference-DIRECT-NET/6667/data_split/train_t0combined.csv",
        f"{REPO}/network-inference-DIRECT-NET/6667/rules",
    ),
    "hsc (Track 1, synthetic)": (
        f"{REPO}/benchmarking/hsc/rules/rules_hsc.txt",
        f"{REPO}/benchmarking/hsc/data_split/train_t0hsc.csv",
        f"{REPO}/benchmarking/hsc/rules",
    ),
    "hsc_chea (Track 3, synthetic+ChEA)": (
        f"{REPO}/benchmarking/hsc_chea/rules/rules_hsc_chea.txt",
        f"{REPO}/benchmarking/hsc_chea/data_split/train_t0hsc_chea.csv",
        f"{REPO}/benchmarking/hsc_chea/rules",
    ),
    "hsc_multiome (Track 2, real ChEA)": (
        f"{REPO}/benchmarking/hsc_multiome/rules/rules_hsc_multiome.txt",
        f"{REPO}/benchmarking/hsc_multiome/data_split/train_t0hsc_multiome.csv",
        f"{REPO}/benchmarking/hsc_multiome/rules",
    ),
    "hsc_multiome_atac (Track 2, real ATAC)": (
        f"{REPO}/benchmarking/hsc_multiome_atac/rules/rules_hsc_multiome_atac.txt",
        f"{REPO}/benchmarking/hsc_multiome_atac/data_split/train_t0hsc_multiome_atac.csv",
        f"{REPO}/benchmarking/hsc_multiome_atac/rules",
    ),
    "hsc_multiome_directnet (Track 2, real DIRECT-NET)": (
        f"{REPO}/benchmarking/hsc_multiome_directnet/rules/rules_hsc_multiome_directnet.txt",
        f"{REPO}/benchmarking/hsc_multiome_directnet/data_split/train_t0hsc_multiome_directnet.csv",
        f"{REPO}/benchmarking/hsc_multiome_directnet/rules",
    ),
    "hsc_multiome_cicero (Track 2, real Cicero)": (
        f"{REPO}/benchmarking/hsc_multiome_cicero/rules/rules_hsc_multiome_cicero.txt",
        f"{REPO}/benchmarking/hsc_multiome_cicero/data_split/train_t0hsc_multiome_cicero.csv",
        f"{REPO}/benchmarking/hsc_multiome_cicero/rules",
    ),
}


def build_dataw_matrices(rules_path: str, data_path: str, nodes: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-shape (nodes x nodes) dataw_abs / dataw_signed matrices, matching strengths.csv's
    own index/columns/NaN convention exactly -- including self-loop-only genes this time.
    """
    rules, regulators_dict = load_rules(rules_path)
    data = pd.read_csv(data_path, index_col=0)

    dataw_abs = pd.DataFrame(np.nan, index=nodes, columns=nodes)
    dataw_signed = pd.DataFrame(np.nan, index=nodes, columns=nodes)
    for gene, regs in regulators_dict.items():
        per_reg = edge_summaries_for_gene(gene, regs, rules[gene], data)
        for reg, summaries in per_reg.items():
            dataw_abs.loc[gene, reg] = summaries["dataw_abs"]
            dataw_signed.loc[gene, reg] = summaries["dataw_signed"]
    return dataw_abs, dataw_signed


def verify_self_loops_unchanged(strengths_dir: str, dataw_abs: pd.DataFrame) -> None:
    """Sanity check (not an assumption): for genes whose only regulator is themselves,
    dataw_abs must equal the existing strengths.csv value exactly (n=1 rule -> every
    summary collapses to the same single number). Raises if that's ever not the case,
    rather than silently overwriting with something inconsistent.
    """
    old = pd.read_csv(f"{strengths_dir}/strengths.csv", index_col=0)
    self_loop_genes = [g for g in old.index if old.loc[g].dropna().index.tolist() == [g]]
    for g in self_loop_genes:
        old_val, new_val = old.loc[g, g], dataw_abs.loc[g, g]
        if not np.isclose(old_val, new_val, atol=1e-8):
            raise ValueError(
                f"{strengths_dir}: self-loop gene {g} changed under dataw "
                f"({old_val} -> {new_val}) -- expected identical for a 1-regulator rule."
            )
    print(f"  verified {len(self_loop_genes)} self-loop-only gene(s) unchanged under dataw")


def main():
    for label, (rules_path, data_path, strengths_dir) in RUNS.items():
        print(f"\n[{label}]")
        try:
            rules, regulators_dict = load_rules(rules_path)
        except FileNotFoundError:
            print(f"  skip -- {rules_path} not found")
            continue
        nodes = list(regulators_dict.keys())

        dataw_abs, dataw_signed = build_dataw_matrices(rules_path, data_path, nodes)
        verify_self_loops_unchanged(strengths_dir, dataw_abs)

        # Back up the sum-based originals once, alongside the new files, rather than just
        # overwriting blind -- rules_<run>.txt (the real source of truth) is untouched
        # either way, but this makes the sum-based values recoverable without a recompute.
        shutil.copy(f"{strengths_dir}/strengths.csv", f"{strengths_dir}/strengths_sum_abs.csv.bak")
        shutil.copy(f"{strengths_dir}/signed_strengths.csv", f"{strengths_dir}/signed_strengths_sum_signed.csv.bak")

        dataw_abs.to_csv(f"{strengths_dir}/strengths.csv")
        dataw_signed.to_csv(f"{strengths_dir}/signed_strengths.csv")
        n_edges = int((dataw_signed.fillna(0) != 0).sum().sum())
        print(f"  wrote {strengths_dir}/{{strengths,signed_strengths}}.csv "
              f"(dataw_abs/dataw_signed, {n_edges} edges); "
              f"sum-based originals backed up as *_sum_*.csv.bak")


if __name__ == "__main__":
    main()
