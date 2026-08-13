"""Barcode 7778: same split-by-failure-type design as 7779, but the variance_collapse
branch MARGINALIZES the gene out (as a regulator, and as a scoring target -- same
"treat as missing" convention as claude_analysis/06_marginal_rule_validation/
marginalize_rules.py) instead of GMM-pole-assigning it. Branch 1 (mean_shift) is IDENTICAL
to 7779's (rescale_mean_shift_gene in adaptive_rescale_v2.py, upgraded to location+scale) --
7778 and 7779 differ in exactly one thing: how a collapsed gene is handled.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_rescale_v2 import classify_genes, rescale_mean_shift_gene

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "06_marginal_rule_validation"))
from marginalize_rules import build_marginal_rules


def build_hybrid_data_and_rules(raw_sample_df, normed_sample_df, nodes, rules, regulators_dict, classification=None):
    """Returns (adaptive_data, rules_out, regulators_dict_out, nodes_out).
    - mean_shift genes: data values corrected in place (rescale_mean_shift_gene), rules
      untouched, gene stays a regulator AND a scoring target.
    - variance_collapse genes: marginalized out of every rule that uses them as a
      regulator, and dropped as a scoring target entirely (no reliable ground truth) --
      data values for these genes are irrelevant after this and left untouched.
    - unflagged genes: untouched in every respect.
    """
    if classification is None:
        classification = classify_genes(raw_sample_df, nodes)

    adaptive = normed_sample_df.copy()
    collapse_genes = set()
    for gene, kind in classification.items():
        if kind == "mean_shift" and gene in raw_sample_df.columns:
            raw_vals = raw_sample_df.loc[normed_sample_df.index, gene]
            adaptive[gene] = rescale_mean_shift_gene(raw_vals, gene).values
        elif kind == "variance_collapse":
            collapse_genes.add(gene)

    rules_out, regulators_dict_out, nodes_out = build_marginal_rules(rules, regulators_dict, nodes, collapse_genes)
    return adaptive, rules_out, regulators_dict_out, nodes_out
