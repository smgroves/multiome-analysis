"""Extend diagnose_leaf_conditional_agreement_full.py's per-gene x per-sample
transferability matrix (FINDINGS.md sec 10: which genes' rules hold up leaf-conditionally
across external samples, vs. which are context-specific) to include the 11 Ireland et al.
2025 condition-level samples. Question: do the SAME genes (TEAD1, PROX1, BACH1, HSF2...)
top the "robust core" ranking on this genuinely independent dataset, or was that ranking an
artifact of the original 33-sample population?

Imports the original script as a module and monkey-patches its `gather_sample_paths`/
`gather_r2` functions (looked up from the module's global namespace at call time, so this
works cleanly) before calling its unmodified `main()` -- guarantees identical
leaf-conditional-scoring logic, just a larger sample set and a separate output directory
(does not touch the original 33-sample-only output files).

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/07_batch_effect_diagnostics/leaf_conditional_transferability_with_ireland2025.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "03_domain_shift_diagnostics"))
import diagnose_leaf_conditional_agreement_full as base

DIR_PREFIX = base.DIR_PREFIX
OUT_DIR = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET/comparisons/ireland2025_external_validation"

IRELAND_PATHS = {
    "cgrp_k5_K5": f"{DIR_PREFIX}/data/cgrp_k5/adata_cgrp_k5_K5_v3_RORA_RORB_ave.csv",
    "cgrp_k5_CGRP": f"{DIR_PREFIX}/data/cgrp_k5/adata_cgrp_k5_CGRP_v3_RORA_RORB_ave.csv",
    "organoid_celltag_RPM": f"{DIR_PREFIX}/data/organoid_celltag/adata_organoid_celltag_RPM_v3_RORA_RORB_ave.csv",
    "organoid_celltag_RPMA": f"{DIR_PREFIX}/data/organoid_celltag/adata_organoid_celltag_RPMA_v3_RORA_RORB_ave.csv",
    "organoid_celltag_WT": f"{DIR_PREFIX}/data/organoid_celltag/adata_organoid_celltag_WT_v3_RORA_RORB_ave.csv",
    "tbo_allograft_5khvg_RPM_CTpostCre": f"{DIR_PREFIX}/data/tbo_allograft_5khvg/adata_tbo_allograft_5khvg_RPM_CTpostCre_v3_RORA_RORB_ave.csv",
    "tbo_allograft_5khvg_RPM_CTpreCre": f"{DIR_PREFIX}/data/tbo_allograft_5khvg/adata_tbo_allograft_5khvg_RPM_CTpreCre_v3_RORA_RORB_ave.csv",
    "rpr2_allograft_RPM": f"{DIR_PREFIX}/data/rpr2_allograft/adata_rpr2_allograft_RPM_v3_RORA_RORB_ave.csv",
    "rpr2_allograft_RPR2": f"{DIR_PREFIX}/data/rpr2_allograft/adata_rpr2_allograft_RPR2_v3_RORA_RORB_ave.csv",
    "celltag_fate_dpt_RPM": f"{DIR_PREFIX}/data/celltag_fate_dpt/adata_celltag_fate_dpt_RPM_v3_RORA_RORB_ave.csv",
    "celltag_fate_dpt_RPMA": f"{DIR_PREFIX}/data/celltag_fate_dpt/adata_celltag_fate_dpt_RPMA_v3_RORA_RORB_ave.csv",
}

_orig_gather_sample_paths = base.gather_sample_paths
_orig_gather_r2 = base.gather_r2


def gather_sample_paths():
    paths = _orig_gather_sample_paths()
    paths.update(IRELAND_PATHS)
    return paths


def gather_r2(name):
    if name in IRELAND_PATHS:
        c = f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats_fixed.csv"
        import pandas as pd
        return pd.read_csv(c)["r2"].mean() if os.path.exists(c) else float("nan")
    return _orig_gather_r2(name)


base.gather_sample_paths = gather_sample_paths
base.gather_r2 = gather_r2
base.OUT_DIR = OUT_DIR

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    base.main()
