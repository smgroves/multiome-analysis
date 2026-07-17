"""Paths and run configuration for the GRN benchmark."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# Repo root, derived from this file (.../benchmarking/grn_benchmark/config.py).
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIRECT_NET = os.path.join(REPO, "network-inference-DIRECT-NET")
BENCH_DATA = os.path.join(REPO, "benchmarking", "data")

# BEELINE example ground truth, downloaded from Murali-group/Beeline (GSD dataset).
BEELINE_GSD = os.path.join(BENCH_DATA, "beeline", "GSD")

# CellOracle Fig-7 benchmark data (ChIP-Atlas ground truth + released method outputs).
CO_CHIP_GT_DIR = os.path.join(BENCH_DATA, "celloracle", "chip_atlas_gt")
CO_INFERENCE_DIR = os.path.join(BENCH_DATA, "celloracle", "inference_results")


@dataclass
class BenchmarkConfig:
    # boba-T run to benchmark (folder under DIRECT_NET, e.g. "9999").
    bobat_run: str = "6667"
    # Restrict every comparison to this node set so metrics are not dominated by
    # genes a given method could never call. Default: intersection of all methods'
    # nodes, computed at runtime when None.
    node_universe: Optional[list[str]] = None
    # Directed comparison (respect source->target) vs. undirected (edge as a set).
    directed: bool = True
    # Where to drop benchmark tables/plots.
    out_dir: str = os.path.join(REPO, "benchmarking", "benchmarking_out")


CFG = BenchmarkConfig()
