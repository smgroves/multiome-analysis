"""Benchmark boba-T (BooleaBayes) GRNs against other network-inference methods.

Four comparison points (see benchmarking/README.md for the full write-up):

    1. NETWORK STRUCTURE   (metrics.structure_metrics) -- do the methods recover the
       same edges/topology as a reference network? Edge Jaccard / precision / recall /
       F1 / degree correlation / sign concordance.
    2. EDGE WEIGHTS (BEELINE) (metrics.beeline_metrics) -- rank a method's weighted
       edges as a classifier of true edges: AUROC / AUPRC / EPR (Pratapa et al. 2020).
       co_reproduction.py reproduces CellOracle's own Fig. S2a/b this way.
    3. PREDICTED VS. ACTUAL EXPRESSION -- NOT in this module; scored with boba-T's own
       get_sklearn_metrics on both methods' validation CSVs. See
       comparison3_fit_celloracle_6667.py and comparison3_score_celloracle_vs_bobat_6667.py.
    4. IN SILICO PERTURBATION (metrics.run_perturbation_comparison) -- stubbed.

Every method normalises to one canonical edge table (see `edges`), so comparisons 1 and
2 are method-agnostic. Add a method by writing a loader and registering it in
`loaders.METHOD_LOADERS`.

Module map:
    config          paths + BenchmarkConfig (CFG)
    edges           canonical edge schema + graph helpers
    loaders         method loaders + ground-truth loaders
    metrics         comparisons 1, 2, and 4
    co_reproduction reproduce the CellOracle Fig-S2 AUROC/EPR benchmark (comparison 2)
    runners         orchestration (run_benchmarks, self-test, main)

ENVIRONMENT: run in the `celloracle_env` conda env (Python 3.10) so CellOracle and the
benchmark share one interpreter (built by benchmarking/setup_celloracle_env.sh). The
core benchmark (pandas/numpy/networkx/sklearn) also runs in `bobaT_env`.
"""

from __future__ import annotations

from .config import CFG, BenchmarkConfig
from .runners import (
    main,
    run_benchmarks,
    run_beeline_dataset,
    run_beeline_selftest,
    run_methods_vs_reference,
)
from .co_reproduction import run_co_reproduction, run_co_reproduction_all

__all__ = [
    "CFG",
    "BenchmarkConfig",
    "main",
    "run_benchmarks",
    "run_beeline_dataset",
    "run_beeline_selftest",
    "run_methods_vs_reference",
    "run_co_reproduction",
    "run_co_reproduction_all",
]
