"""CLI entry point for the GRN benchmark.

The implementation now lives in the `grn_benchmark` package (config, edges, loaders,
metrics, co_reproduction, runners). This file just runs the default pipeline; import
the package directly for interactive use:

    from grn_benchmark import run_co_reproduction_all, run_beeline_selftest

Run in the `celloracle_env` conda env (see benchmarking/setup_celloracle_env.sh).
"""

from grn_benchmark.runners import main

if __name__ == "__main__":
    main()
