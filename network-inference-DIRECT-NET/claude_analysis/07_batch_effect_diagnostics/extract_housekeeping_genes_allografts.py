"""Extract the same 12-gene housekeeping panel from the 12 allografts' raw velocyto .loom
files (data/external_validation_looms/allografts/*.loom), for the batch-effect check in
this directory. Same normalization recipe as extract_housekeeping_genes.py (size-factor by
total UMI count, log1p, no MAGIC) but reading from loom instead of h5ad.

No loom-filename-to-allograft mapping exists (each loom's internal CellID prefix, e.g.
"1FL4V", bears no relationship to the allograft name, e.g. "1L") -- reconstructed here from
the *_clusters.csv files already in the same directory. Checked directly (not assumed):
each clusters.csv has two columns, `CellID` (format `<loom's own raw CellID>-YN<n>`, where
"YN<n>" is an internal sequencing-run code, e.g. "-YN12" -- NOT the allograft name) and
`allograft` (the actual label, e.g. "1L"). Confirmed by direct example: mt2_clusters.csv's
CellID `possorted_genome_bam_1FL4V:AACCATGGTGCCTGTGx-YN1` strips to exactly
`possorted_genome_bam_1FL4V:AACCATGGTGCCTGTGx`, which is a real cell in
possorted_genome_bam_YN1.loom -- so the mapping key is CellID with the trailing `-YN\d+`
regex-stripped, mapped to the separate `allograft` column value (an earlier version of this
script wrongly assumed the CellID's own suffix WAS the allograft name, which fails for
every file, most obviously TKO-luc: its CellID suffix is "-YN13", not "-TKO-luc").

Run in celloracle_env (has loompy; bobaT_env doesn't):
    /opt/anaconda3/envs/celloracle_env/bin/python claude_analysis/07_batch_effect_diagnostics/extract_housekeeping_genes_allografts.py
"""

import glob
import os
import re

import loompy
import numpy as np
import pandas as pd

LOOM_DIR = "/Users/xpz5km/Documents/GitHub/multiome-analysis/data/external_validation_looms/allografts"
OUT_DIR = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET/claude_analysis/07_batch_effect_diagnostics/housekeeping"
HK_GENES = ["Actb", "Gapdh", "Tbp", "B2m", "Ppia", "Rplp0", "Ywhaz", "Sdha", "Hprt", "Ubc", "Polr2a", "Tubb5"]


YN_SUFFIX_RE = re.compile(r"-YN\d+$")


def build_cellid_to_allograft():
    """{stripped_cellid (matches a loom's own raw CellID): allograft_name}, built from the
    `CellID`/`allograft` columns directly, not from the clusters.csv filename."""
    mapping = {}
    unmatched = 0
    for f in glob.glob(f"{LOOM_DIR}/*_clusters.csv"):
        df = pd.read_csv(f)
        for cid, allograft in zip(df["CellID"], df["allograft"]):
            if YN_SUFFIX_RE.search(cid):
                mapping[YN_SUFFIX_RE.sub("", cid)] = allograft
            else:
                unmatched += 1
    if unmatched:
        print(f"WARNING: {unmatched} CellIDs across all clusters.csv did not match the -YN\\d+ suffix pattern")
    return mapping


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cellid_to_allograft = build_cellid_to_allograft()
    print(f"Master mapping: {len(cellid_to_allograft)} cells across {len(set(cellid_to_allograft.values()))} allografts")

    all_rows = []
    all_labels = []
    all_index = []

    loom_files = sorted(glob.glob(f"{LOOM_DIR}/*.loom"))
    for lf in loom_files:
        with loompy.connect(lf) as ds:
            gene_names = ds.ra["Gene"]
            cell_ids = ds.ca["CellID"]
            hk_mask = np.isin(gene_names, HK_GENES)
            if hk_mask.sum() == 0:
                print(f"{os.path.basename(lf)}: no housekeeping genes found, skipping")
                continue
            hk_gene_names = gene_names[hk_mask]

            matched_mask = np.isin(cell_ids, list(cellid_to_allograft.keys()))
            n_matched = matched_mask.sum()
            if n_matched == 0:
                print(f"{os.path.basename(lf)}: 0/{len(cell_ids)} cells matched to a known allograft, skipping")
                continue

            main_layer = ds.layers[""]
            total_counts = main_layer[:, matched_mask].sum(axis=0)  # sum across ALL genes, matched cells only
            hk_counts = main_layer[hk_mask, :][:, matched_mask]  # (n_hk_genes, n_matched_cells)

            matched_cell_ids = cell_ids[matched_mask]
            allografts = [cellid_to_allograft[c] for c in matched_cell_ids]

            df = pd.DataFrame(hk_counts.T, columns=hk_gene_names, index=matched_cell_ids)
            # some housekeeping genes may be duplicated in the gene annotation; keep first occurrence
            df = df.loc[:, ~df.columns.duplicated()]
            size_factor = total_counts / np.median(total_counts)
            normalized = df.div(size_factor, axis=0)
            log_normalized = np.log1p(normalized)

            all_rows.append(log_normalized)
            all_labels.extend(allografts)
            all_index.extend(matched_cell_ids)
            print(f"{os.path.basename(lf)}: {n_matched}/{len(cell_ids)} cells matched -> allografts {sorted(set(allografts))}")

    combined = pd.concat(all_rows)
    combined["_allograft"] = all_labels
    combined.index.name = "CellID"

    for allograft, sub in combined.groupby("_allograft"):
        sub = sub.drop(columns="_allograft")
        out_path = f"{OUT_DIR}/hk_allograft_{allograft}.csv"
        sub.to_csv(out_path)
        print(f"Wrote {out_path}: {sub.shape}")


if __name__ == "__main__":
    main()
