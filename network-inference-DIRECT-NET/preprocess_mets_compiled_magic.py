"""Finish preprocessing Mets_compiled for external validation: QC filter -> normalize by
total UMI count -> log1p -> MAGIC impute -> add RORA_RORB. Takes over from
preprocess_mets_compiled.R, which exports raw network-gene counts + QC metadata (R doesn't
have a convenient MAGIC implementation; this uses the same `magic` package + solver the
allografts' own preprocessing notebook used).

QC filter is a documented substitute, not a re-derivation of the original notebook's
private `mazebox.pp.dropkick_recipe` step: percent.mt < 20 and nFeature_RNA > 200, both
standard thresholds. Both this dataset's percent.mt (max 6.8%) and nFeature_RNA (min 441)
already look like they were QC'd before this object was saved, so this step is expected to
drop few or zero cells -- included anyway per explicit instruction to not skip it.

Run in bobaT_env (has `magic`):
    /opt/anaconda3/envs/bobaT_env/bin/python preprocess_mets_compiled_magic.py
"""

import magic
import numpy as np
import pandas as pd

OUT_DIR = "data/mets_compiled"


def main():
    raw = pd.read_csv(f"{OUT_DIR}/raw_counts_network_genes.csv", index_col=0)
    qc = pd.read_csv(f"{OUT_DIR}/qc_metadata.csv", index_col=0)
    qc = qc.reindex(raw.index)
    print(f"Loaded: {raw.shape}")

    keep = (qc["percent.mt"] < 20) & (qc["nFeature_RNA"] > 200)
    print(f"QC filter (percent.mt<20, nFeature_RNA>200): {keep.sum()}/{len(keep)} cells kept")
    raw, qc = raw[keep], qc[keep]

    # Normalize by total UMI count (genome-wide nCount_RNA, not re-derived from just these
    # ~92 genes) to a common target sum, then log1p -- same normalization family as the
    # allografts' log1p_norm step, computed properly this time (previous attempt mistakenly
    # used a "data" slot that turned out to be identical to raw counts).
    size_factor = qc["nCount_RNA"] / qc["nCount_RNA"].median()
    normalized = raw.div(size_factor, axis=0)
    log_normalized = np.log1p(normalized)
    print(f"Normalized + log1p. Value range sample (ASCL1): "
          f"{log_normalized['ASCL1'].min():.3f} to {log_normalized['ASCL1'].max():.3f}, "
          f"{log_normalized['ASCL1'].nunique()} distinct values")

    magic_operator = magic.MAGIC(solver="approximate")
    imputed = magic_operator.fit_transform(log_normalized)
    print(f"MAGIC done: {imputed.shape}")

    if "RORA" in imputed.columns and "RORB" in imputed.columns:
        imputed["RORA_RORB"] = imputed[["RORA", "RORB"]].mean(axis=1)

    imputed.index.name = "CellID"
    imputed.to_csv(f"{OUT_DIR}/adata_mets_compiled_v3_RORA_RORB_ave.csv")
    print(f"Wrote {OUT_DIR}/adata_mets_compiled_v3_RORA_RORB_ave.csv: {imputed.shape}")


if __name__ == "__main__":
    main()
