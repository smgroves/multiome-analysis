"""Ground truth for the HSC combinatorial-logic comparison (see benchmarking/README.md).

Parses data/BoolODE/data/HSC.txt (Krumsiek et al. 2011's 11-gene hematopoietic
differentiation Boolean model -- the GATA1/PU.1 toggle switch) into two things:

1. A canonical candidate-network edge list (source, target), for boba-T/GENIE3/CellOracle
   to fit on -- same role as DIRECT-NET's candidate network for the SCLC comparisons.
2. The literal ground-truth TRUTH TABLE per gene: every combination of its regulators'
   Boolean states, and the rule's output for that combination. This is the actual object
   being compared in the new "combinatorial logic recovery" comparison -- not just whether
   an edge exists (comparison 1) but whether the *combinatorial structure* (AND/OR/NOT
   between co-regulators) is right. No other method in this benchmark outputs anything
   directly comparable to a truth table except boba-T (whose fitted rule IS one); GENIE3's
   and CellOracle's continuous outputs get an "implied truth table" reconstructed by
   thresholding their predicted expression at every regulator-state combination -- see
   comparison_hsc_truth_table_scoring.py.

Self-loops (e.g. `Cebpa and not(...)`, `Gata2 and not(...)`) are real in this model
(autoregulation keeping a gene ON once activated) and are kept, matching this benchmark's
existing convention (boba-T's own rules also keep self-loops; see edges.py).
"""

from __future__ import annotations

import itertools
import os
import re

import pandas as pd

RULE_FILE = "data/BoolODE/data/HSC.txt"
OUT_DIR = "data/hsc_ground_truth"

# BoolODE's HSC.txt uses these gene names; keep them as-is (they're already the "real"
# gene symbols this model is about -- Gata1, Pu1, etc. -- just not upper-cased the way
# this benchmark's other human-convention data is). Upper-case for consistency with every
# other network in this project.


def parse_rule(rule: str) -> tuple[str, list[str]]:
    """Return (python-evaluable expression, list of regulator gene names) for one rule."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rule)
    regulators = sorted({t for t in tokens if t not in ("and", "or", "not")})
    expr = rule
    for op_word, op_sym in [("and", " and "), ("or", " or "), ("not", " not ")]:
        expr = re.sub(rf"\b{op_word}\b", op_sym, expr)
    return expr, regulators


def load_rules(path: str = RULE_FILE) -> dict[str, tuple[str, list[str]]]:
    df = pd.read_csv(path, sep="\t")
    return {row["Gene"]: parse_rule(row["Rule"]) for _, row in df.iterrows()}


def truth_table(expr: str, regulators: list[str]) -> pd.DataFrame:
    """Every combination of `regulators`' Boolean states -> the rule's output."""
    rows = []
    for combo in itertools.product([False, True], repeat=len(regulators)):
        env = dict(zip(regulators, combo))
        rows.append({**{r: int(v) for r, v in env.items()}, "output": int(eval(expr, {}, env))})
    return pd.DataFrame(rows)


def build_candidate_network(rules: dict[str, tuple[str, list[str]]]) -> pd.DataFrame:
    edges = [
        {"source": reg.upper(), "target": target.upper()}
        for target, (_, regulators) in rules.items()
        for reg in regulators
    ]
    return pd.DataFrame(edges).drop_duplicates()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rules = load_rules()

    candidate_net = build_candidate_network(rules)
    candidate_net.to_csv(f"{OUT_DIR}/candidate_network.csv", header=False, index=False)
    print(f"{len(candidate_net)} candidate edges over "
          f"{len(set(candidate_net.source) | set(candidate_net.target))} genes -> "
          f"{OUT_DIR}/candidate_network.csv")

    for gene, (expr, regulators) in rules.items():
        tt = truth_table(expr, regulators)  # eval() needs the original-case names used in `expr`
        tt = tt.rename(columns={r: r.upper() for r in regulators})
        tt.to_csv(f"{OUT_DIR}/truth_table_{gene.upper()}.csv", index=False)
        n_on = tt["output"].sum()
        print(f"  {gene.upper()}: {len(regulators)} regulators ({','.join(r.upper() for r in regulators)}), "
              f"{n_on}/{len(tt)} combinations -> ON")


if __name__ == "__main__":
    main()
