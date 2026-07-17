# Benchmark data

All large files here are **gitignored** (see repo `.gitignore`); only this README and the
small BEELINE / ChIP-Atlas ground truth are tracked. Re-download the rest with the
commands below.

## Layout

```
data/
├── beeline/GSD/GroundTruthNetwork.csv        # BEELINE example ground truth (tracked)
├── celloracle/
│   ├── chip_atlas_gt/{Heart,Kidney,Liver,Lung,Spleen}/chip_GT_links.csv  # ground truth (tracked)
│   ├── chip_atlas_gt/TFs_in_gimmev5_mouse.npy
│   ├── human_promoter_base_GRN.parquet       # CellOracle base GRN (gitignored)
│   └── inference_results/<sample>/<method>/link.csv   # Fig-7 method outputs (gitignored, 294MB)
├── tabula_muris/                             # scRNA-seq input for boba-T (gitignored)
└── cusanovich_atac/                          # scATAC-seq input for boba-T (gitignored)
```

## scRNA-seq — Tabula Muris droplet (10x), GSE109774

Source: figshare "Single-cell RNA-seq data from microfluidic emulsion (v2)"
(article 5968960). The 13 channels matching the CellOracle Fig-7 benchmark samples are
under `tabula_muris/droplet/<Channel>/{barcodes.tsv,genes.tsv,matrix.mtx}` (standard 10x).

NOTE: the benchmark's "Heart-10X_P7_4" is Tabula Muris **`Heart_and_Aorta-10X_P7_4`**
(same P7_4 channel); it is stored here renamed to `Heart-10X_P7_4` to match the
ChIP-Atlas ground-truth tissue folders. The other 12 channels keep their original names.

Channels: Heart-10X_P7_4; Kidney-10X_P4_5/P4_6/P7_5; Liver-10X_P4_2/P7_0/P7_1;
Lung-10X_P7_8/P7_9/P8_12/P8_13; Spleen-10X_P4_7/P7_6.

Re-download:
```bash
curl -L https://ndownloader.figshare.com/files/10700167 -o droplet.zip   # 393 MB
# extract only the 13 channels above (Heart_and_Aorta -> rename to Heart)
curl -L https://ndownloader.figshare.com/files/10700161 -o tabula_muris/metadata_droplet.csv
curl -L https://ndownloader.figshare.com/files/13088039 -o tabula_muris/annotations_droplet.csv
```

## scATAC-seq — Cusanovich mouse sci-ATAC-seq atlas, GSE111586

Source: atlas.gs.washington.edu/mouse-atac/. Whole-atlas binarized peak×cell matrix
(81,173 cells × 436,206 peaks); subset by the `tissue` column of `cell_metadata.txt`
to Heart / Kidney / Liver / Lung / Spleen for the base GRN.

Files in `cusanovich_atac/`:
- `atac_matrix.binary.qc_filtered.mtx.gz` (1.1 GB)
- `cells.txt`, `peaks.txt`, `cell_metadata.txt` (tissue, replicate, cluster, cell_label)

Re-download:
```bash
B=http://krishna.gs.washington.edu/content/members/ajh24/mouse_atlas_data_release
curl -L $B/matrices/atac_matrix.binary.qc_filtered.mtx.gz   -o cusanovich_atac/atac_matrix.binary.qc_filtered.mtx.gz
curl -L $B/matrices/atac_matrix.binary.qc_filtered.cells.txt -o cusanovich_atac/cells.txt
curl -L $B/matrices/atac_matrix.binary.qc_filtered.peaks.txt -o cusanovich_atac/peaks.txt
curl -L $B/metadata/cell_metadata.txt                        -o cusanovich_atac/cell_metadata.txt
```

## Note on pairing

The scRNA (Tabula Muris) and scATAC (Cusanovich) are **unpaired** — matched by tissue,
not by cell. This mirrors exactly how CellOracle built its benchmark (scATAC → base GRN,
scRNA → GRN inference), so it is the right setup for a like-for-like boba-T comparison,
but differs from a true multiome experiment.
