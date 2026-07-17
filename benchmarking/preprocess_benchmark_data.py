"""Preprocess the Tabula Muris scRNA channels for the GRN benchmark.

Reproduces CellOracle's standard scRNA preprocessing (the shared input given to every
benchmark method) and writes log_data.mtx / all_genes.csv / var_genes.csv / meta_data.csv
per sample under benchmarking/data/preprocessed/<channel>/. boba-T consumes the same
log_data + var_genes, so the comparison stays like-for-like.

Run in the celloracle_env conda env:
    python preprocess_benchmark_data.py                 # all 13 channels, defaults
    python preprocess_benchmark_data.py Liver-10X_P7_0  # one channel

See grn_benchmark/preprocess.py for the recipe and the parameters left open by the paper.
"""

import sys

from grn_benchmark.preprocess import BENCHMARK_CHANNELS, preprocess_all_channels

if __name__ == "__main__":
    channels = sys.argv[1:] or BENCHMARK_CHANNELS
    preprocess_all_channels(channels=channels)
