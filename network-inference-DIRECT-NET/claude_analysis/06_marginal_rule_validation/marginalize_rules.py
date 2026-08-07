"""Marginalize BoBa-T rule tables over regulators missing from an external validation
dataset, as an alternative to forcing a missing regulator's value to 0.

Background (confirmed by reading bobaT/tl.py's parent_heatmap and bobaT/load.py's
load_data directly, not assumed): bobaT.load.load_data raises a hard error if a network
node is completely absent from an input CSV -- there is no existing "silently treat a
missing gene as 0" behavior in the live pipeline. If one were built (e.g. by zero-filling
a missing regulator's column before calling load_data), parent_heatmap's leaf-weighting
(`heat[i, leaf] *= float(df[regulator])` for ON-leaves, `1 - float(df[regulator])` for
OFF-leaves) means a forced value of exactly 0 makes every ON-leaf get weight 0 and every
OFF-leaf keep full weight -- i.e. it deterministically asserts "this regulator is OFF",
not "unknown". This module instead collapses a missing regulator out of the rule table
entirely (average its ON/OFF entries, a uniform 0.5/0.5 prior), producing a smaller rule
over only the regulators that are actually present in the external dataset -- no
zero-filled placeholder column needed for that node at all.

Rule-table bit convention (confirmed against bobaT.utils.idx2binary and
bobaT.tl.parent_heatmap): a rule for a node with regulators [r0, r1, ..., r{n-1}] (this
exact order, from the rule file / regulators_dict) is a length-2**n array indexed by
idx2binary(leaf, n) -- a binary string, MSB-first, zero-padded to length n -- where bit 0
(the string's leftmost/most-significant character) is regulator r0, bit 1 is r1, etc.
`rule.reshape([2]*n)` in numpy's default C order reproduces exactly this: axis 0 is the
slowest-varying (most-significant) axis, matching r0. Averaging out (`.mean(axis=...)`)
the axes for missing regulators and flattening back (`.reshape(-1)`) therefore yields a
new rule table in the same convention, just over the remaining (present) regulators, in
their original relative order -- directly usable by bobaT's own unmodified
parent_heatmap/fit_validation/plot_accuracy (which only look up `rule = rules[node]` and
`regulators_dict[node]`, with no other coupling), no changes to the BoBa-T package itself.
"""

import numpy as np


def marginalize_rule(rule, regulators, missing_regulators):
    """Marginalize a single node's rule table over its missing regulators.

    :param rule: 1D array-like of length 2**n (n = len(regulators)).
    :param regulators: list of n regulator names, same order as the rule file.
    :param missing_regulators: iterable of regulator names (subset of `regulators`) to
        marginalize out.
    :return: (marginal_rule, marginal_regulators) -- marginal_rule has length
        2**(n - m), marginal_regulators is `regulators` with the missing ones removed,
        same relative order. If no regulators are missing, returns the inputs unchanged
        (as an ndarray / list).
    """
    n = len(regulators)
    missing_set = set(missing_regulators)
    missing_axes = tuple(i for i, r in enumerate(regulators) if r in missing_set)
    present_regulators = [r for r in regulators if r not in missing_set]

    rule = np.asarray(rule, dtype=float)
    if not missing_axes:
        return rule, list(regulators)

    assert rule.shape == (2**n,), f"expected rule of length 2**{n}={2**n}, got {rule.shape}"
    tensor = rule.reshape([2] * n)
    marginal_tensor = tensor.mean(axis=missing_axes)
    marginal_rule = marginal_tensor.reshape(-1)
    return marginal_rule, present_regulators


def build_marginal_rules(rules, regulators_dict, nodes, missing_genes):
    """Build a reduced (rules, regulators_dict, nodes) triple for validating against a
    dataset that's missing some of the network's genes.

    :param rules: {node: np.array} as returned by bobaT.load.load_rules.
    :param regulators_dict: {node: [reg1, ...]} as returned by bobaT.load.load_rules.
    :param nodes: full list of node names (e.g. bobaT.utils.get_nodes's second return
        value), same case convention as `rules`/`regulators_dict` keys.
    :param missing_genes: iterable of gene names absent from the external dataset (same
        case convention as `nodes`).
    :return: (rules_out, regulators_dict_out, nodes_out).
        nodes_out = nodes with missing_genes removed (a gene with no ground truth in the
        external data can't be scored regardless of how its regulators are handled).
        For every remaining node, if any of its regulators are in missing_genes, its rule
        and regulator list are marginalized via marginalize_rule; otherwise passed through
        unchanged. rules_out/regulators_dict_out cover exactly nodes_out.
    """
    missing_set = set(missing_genes)
    nodes_out = [n for n in nodes if n not in missing_set]

    rules_out = {}
    regulators_dict_out = {}
    for node in nodes_out:
        regs = list(regulators_dict[node])
        rule = rules[node]
        node_missing = [r for r in regs if r in missing_set]
        if node_missing:
            marginal_rule, marginal_regs = marginalize_rule(rule, regs, node_missing)
            rules_out[node] = marginal_rule
            regulators_dict_out[node] = marginal_regs
        else:
            rules_out[node] = np.asarray(rule, dtype=float)
            regulators_dict_out[node] = regs

    return rules_out, regulators_dict_out, nodes_out


if __name__ == "__main__":
    # Self-test: marginalizing over ALL of a node's regulators must reduce to the simple
    # mean of the rule table (sanity check on axis/reshape correctness), and marginalizing
    # over a proper subset must preserve the correct entries for a hand-checkable 2-regulator
    # example.
    rng = np.random.default_rng(0)

    rule4 = rng.random(16)  # 4 regulators -> 16-entry rule
    regs4 = ["A", "B", "C", "D"]
    marg_all, regs_all = marginalize_rule(rule4, regs4, regs4)
    assert regs_all == []
    assert np.allclose(marg_all, rule4.mean()), "marginalizing over all regulators must equal the overall mean"

    # 2-regulator hand-checkable case: rule indexed [A=0,B=0]=p00 [A=0,B=1]=p01 [A=1,B=0]=p10 [A=1,B=1]=p11
    p00, p01, p10, p11 = 0.1, 0.2, 0.7, 0.9
    rule2 = np.array([p00, p01, p10, p11])
    # marginalize out B: new rule over just A, index 0 = A off, index 1 = A on
    marg_B, regs_B = marginalize_rule(rule2, ["A", "B"], ["B"])
    assert regs_B == ["A"]
    assert np.allclose(marg_B, [(p00 + p01) / 2, (p10 + p11) / 2]), marg_B
    # marginalize out A: new rule over just B, index 0 = B off, index 1 = B on
    marg_A, regs_A = marginalize_rule(rule2, ["A", "B"], ["A"])
    assert regs_A == ["B"]
    assert np.allclose(marg_A, [(p00 + p10) / 2, (p01 + p11) / 2]), marg_A

    # no missing regulators -> unchanged
    unchanged_rule, unchanged_regs = marginalize_rule(rule2, ["A", "B"], [])
    assert unchanged_regs == ["A", "B"]
    assert np.allclose(unchanged_rule, rule2)

    print("marginalize_rule self-tests passed")

    # build_marginal_rules smoke test against the real 6667 rules, pretending a couple of
    # real regulator genes are missing.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # Minimal reimplementation of bobaT.load.load_rules's parsing (avoids requiring the
    # bobaT package / a specific conda env just to smoke-test this module).
    rules_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "6667", "rules", "rules_6667.txt",
    )
    rules_6667 = {}
    regulators_dict_6667 = {}
    with open(rules_path) as f:
        for line in f:
            parts = line.strip().split("|")
            regulators_dict_6667[parts[0]] = parts[1].split(",")
            rules_6667[parts[0]] = np.asarray([float(x) for x in parts[2].split(",")])
    nodes_6667 = sorted(rules_6667.keys())

    fake_missing = {"ASCL1", "REST"}  # pretend these two genes are absent from a dataset
    r_out, rd_out, n_out = build_marginal_rules(rules_6667, regulators_dict_6667, nodes_6667, fake_missing)
    assert set(n_out) == set(nodes_6667) - fake_missing
    affected = [n for n in n_out if any(g in fake_missing for g in regulators_dict_6667[n])]
    print(f"{len(affected)} of {len(n_out)} remaining nodes had a regulator in {fake_missing} and were marginalized")
    for n in affected[:5]:
        orig_n_regs = len(regulators_dict_6667[n])
        new_n_regs = len(rd_out[n])
        assert new_n_regs == orig_n_regs - len([g for g in regulators_dict_6667[n] if g in fake_missing])
        assert len(rules_6667[n]) == 2**orig_n_regs
        assert len(r_out[n]) == 2**new_n_regs
        print(f"  {n}: {orig_n_regs} regulators -> {new_n_regs}, rule length {2**orig_n_regs} -> {2**new_n_regs}")
    print("build_marginal_rules smoke test passed")
