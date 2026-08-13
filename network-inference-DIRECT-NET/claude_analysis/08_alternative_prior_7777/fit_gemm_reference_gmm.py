"""Fit a per-gene 2-component Gaussian Mixture Model on GEMM's own RAW (pre-quantile-clip,
log1p+MAGIC-imputed) training data -- the reference distribution barcode 7779 scores new
external samples against. Reuses bobaT's own `norm="gmm"` convention exactly (n_components=2,
"ON" cluster = whichever component has the higher mean) from bobaT/load.py:152-162, just fit
once on GEMM and reused across every external sample, instead of being refit fresh on each
sample the way bobaT's built-in norm="gmm" option does.

Why fit on GEMM and reuse, rather than fit fresh per sample (bobaT's existing behavior): a
per-sample GMM fit implicitly assumes that sample's own bimodal ON/OFF split is meaningful --
exactly the assumption that fails for a gene whose variance has collapsed in that sample (a
2-component GMM fit on a single collapsed cluster will just split it arbitrarily down the
middle, manufacturing a fake ON/OFF split out of noise). Using GEMM's reference components
and asking the new sample's raw values "which of these two established clusters are you
closer to" is well-defined even when the sample itself has no real internal bimodality --
it degrades gracefully to "you're all confidently on one side," not a manufactured split.

Run in bobaT_env_py3.13 (needs bobaT for node list + sklearn for GaussianMixture):
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/08_alternative_prior_7777/fit_gemm_reference_gmm.py
"""

import pickle

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd
from sklearn.mixture import GaussianMixture

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
RULES_PATH = f"{DIR_PREFIX}/6667/rules/rules_6667.txt"
OUT_PATH = f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777/gemm_reference_gmm.pkl"


def load_nodes():
    nodes = []
    with open(RULES_PATH) as f:
        for line in f:
            nodes.append(line.split("|")[0].strip().upper())
    return nodes


def main():
    nodes = load_nodes()

    gemm_full = pd.read_csv(f"{DIR_PREFIX}/data/adata_imputed_combined_v3_RORA_RORB_ave.csv", index_col=0)
    gemm_full.columns = [c.upper() for c in gemm_full.columns]
    with open(f"{DIR_PREFIX}/6667/data_split/test_train_indicescombined.p", "rb") as f:
        split_indices = pickle.load(f)
    train_cellids = set(split_indices["train_cellID"])
    gemm_train = gemm_full.loc[gemm_full.index.isin(train_cellids), nodes]
    print(f"GEMM training data for GMM reference fit: {gemm_train.shape}")

    reference = {}
    for gene in nodes:
        d = gemm_train[gene].values.reshape(-1, 1)
        gm = GaussianMixture(n_components=2, random_state=0)
        gm.fit(d)
        on_idx = 1 if gm.means_[0][0] < gm.means_[1][0] else 0
        reference[gene] = {
            "means": gm.means_.flatten(), "covariances": gm.covariances_.flatten(),
            "weights": gm.weights_, "on_idx": on_idx,
            "gmm": gm,  # keep the fitted object itself for direct predict_proba reuse
        }
        print(f"{gene}: ON cluster mean={gm.means_[on_idx][0]:.4f}, OFF cluster mean={gm.means_[1-on_idx][0]:.4f}, "
              f"ON weight={gm.weights_[on_idx]:.3f}")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(reference, f)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
