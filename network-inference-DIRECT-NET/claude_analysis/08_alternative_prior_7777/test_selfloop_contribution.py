"""Test: of the 13 genes whose regulator SET changed vs 6667 (not just reordered), all 13
gained themselves as a new self-loop regulator under 7777. Self-loop regulators are already
known (see 6667's 11 pre-existing self-loop genes) to trivially inflate R2/AUC, since the
gene's own already-known state is used as one of its own predictors. This script checks how
much of 7777's in-sample R2 gain on these 13 genes survives if the self term is marginalized
back out (uniform 0.5/0.5), holding everything else about 7777's fit (the aggregate prior,
all other rules) fixed.
"""
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd
import sys, os
sys.path.insert(0, "claude_analysis/06_marginal_rule_validation")
sys.path.insert(0, "claude_analysis/misc")
from marginalize_rules import marginalize_rule
from fixed_get_sklearn_metrics import get_sklearn_metrics_fixed

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
SELF_LOOP_GAINED = ["BACH1","BACH2","EHF","FOXO3","GRHL2","HES1","NFIA","PBX1","REST","RUNX1","SIX1","TCF7L2","THRB"]

rules_7777, regs_7777 = bb.load.load_rules(fname=f"{DIR_PREFIX}/7777/rules/rules_7777.txt")
rules_6667, regs_6667 = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")

graph, vertex_dict = bb.load.load_network(
    f"{DIR_PREFIX}/networks/feature_selection/DIRECT-NET_network_2020db_0.1/combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv",
    remove_sinks=False, remove_selfloops=False, remove_sources=False)
v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

test_data = bb.load.load_data(f"{DIR_PREFIX}/6667/data_split/test_t0combined.csv", nodes, norm=0.3,
                               delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)

# Build a "7777 minus self-loop" rule set: for the 13 genes, marginalize out the self regulator.
rules_noself = dict(rules_7777)
regs_noself = dict(regs_7777)
for gene in SELF_LOOP_GAINED:
    regs = regs_7777[gene]
    new_rule, new_regs = marginalize_rule(rules_7777[gene], regs, [gene])
    rules_noself[gene] = new_rule
    regs_noself[gene] = new_regs

VAL_DIR = f"{DIR_PREFIX}/7777/validation/in_sample_test_check_noself"
os.makedirs(VAL_DIR, exist_ok=True)
_, tprs_all, fprs_all, area_all = bb.tl.fit_validation(
    test_data, data_test_t1=None, nodes=nodes, regulators_dict=regs_noself, rules=rules_noself,
    save=True, save_dir=VAL_DIR, plot=False, show_plots=False, save_df=True, fname="insample_7777_noself")
bb.tl.save_auc_by_gene(area_all, nodes, VAL_DIR)
stats_noself = get_sklearn_metrics_fixed(VAL_DIR, nodes)
stats_noself = stats_noself.set_index("gene")

print(f"{'gene':10s} {'r2_6667':>9s} {'r2_7777':>9s} {'r2_7777_noself':>15s} {'pct_gain_from_self':>20s}")

# Load the previously computed 6667/7777 in-sample stats for reference
prev = pd.read_csv(f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777/in_sample_6667_vs_7777.csv").set_index("gene")
rows = []
for gene in SELF_LOOP_GAINED:
    r6667 = prev.loc[gene, "r2_6667"]
    r7777 = prev.loc[gene, "r2_7777"]
    rnoself = stats_noself.loc[gene, "r2"] if gene in stats_noself.index else np.nan
    total_gain = r7777 - r6667
    gain_kept_without_self = rnoself - r6667
    pct_from_self = 100 * (1 - gain_kept_without_self / total_gain) if total_gain != 0 else np.nan
    rows.append({"gene": gene, "r2_6667": r6667, "r2_7777": r7777, "r2_7777_noself": rnoself,
                 "total_gain": total_gain, "gain_without_self": gain_kept_without_self, "pct_gain_from_self": pct_from_self})
    print(f"{gene:10s} {r6667:9.4f} {r7777:9.4f} {rnoself:15.4f} {pct_from_self:19.1f}%")

pd.DataFrame(rows).to_csv(f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777/selfloop_contribution_test.csv", index=False)
