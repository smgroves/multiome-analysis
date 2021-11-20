# CIBERSORT analysis 

1. Build a signature matrix from scRNA-seq data
    1.	Pre-label each cell with cell’s phenotype
    2.	Only include cells in clusters/closest to archetype with labels
    3.	No missing entries, tab-delimited
    4.	Will automatically normalize by total reads
    5. Signature matrix and mixture files must be in same normalization space
    7. Data should be in linear space (although CIBERSORTX should automatically fix this if max <50)
1.	Mixture file: already done, use human cell lines and human tumors
    1. In this folder: cell line data (linear space, RPKM data) and thomas tumor data (RPKM, linear space)
    2. Note: clean-read-counts.py shows how I generated the final datasets in RPKM format. Can provide raw data upon request.
    3. Note #2: Tumor dataset was read into excel at some point, so some gene names are dates and will now be excluded from analysis.
