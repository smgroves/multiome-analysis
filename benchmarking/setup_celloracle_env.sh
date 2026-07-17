#!/usr/bin/env bash
# Reproducible setup for the `celloracle_env` conda environment (Apple Silicon / macOS).
#
# CellOracle 0.20.0 does not pip-install cleanly on an M-series Mac out of the box:
#   1. velocyto's C extension needs OpenMP; Apple's system clang rejects `-fopenmp`.
#   2. gimmemotifs is pinned to <=0.17.2, whose C sources use K&R implicit-int that
#      modern clang treats as a hard error.
#   3. gimmemotifs 0.17.2 needs pkg_resources (removed in setuptools >=81).
#   4. CellOracle imports IPython/ipywidgets at package load.
# This script works around all four. Re-run is idempotent-ish (pip skips satisfied pkgs).
#
# Usage:  bash benchmarking/setup_celloracle_env.sh
set -euo pipefail

CONDA=/opt/anaconda3/bin/conda
ENV=celloracle_env
PY=/opt/anaconda3/envs/$ENV/bin/python
PIP=/opt/anaconda3/envs/$ENV/bin/pip

# 1. Env on Python 3.10 (CellOracle supports 3.8-3.10, NOT 3.11+).
$CONDA create -y -n $ENV python=3.10

# 2. OpenMP-capable compiler for velocyto (conda-forge clang + libomp).
$CONDA install -y -n $ENV -c conda-forge clang_osx-arm64 clangxx_osx-arm64 llvm-openmp

export CC=$(ls /opt/anaconda3/envs/$ENV/bin/*-clang | head -1)
export CXX=$(ls /opt/anaconda3/envs/$ENV/bin/*-clang++ | head -1)
export SDKROOT=$(xcrun --show-sdk-path)
# Let old K&R C in gimmemotifs 0.17.2 compile under modern clang.
export CFLAGS="-Wno-implicit-int -Wno-int-conversion -Wno-implicit-function-declaration -Wno-error=implicit-int -Wno-error=int-conversion"

# 3. Build prerequisites, then the two problem C-extension deps, without build isolation
#    (so they see numpy/cython and the OpenMP compiler above).
$PIP install "numpy<2" cython "setuptools<81" wheel
$PIP install velocyto --no-build-isolation
$PIP install "gimmemotifs==0.17.2" --no-build-isolation

# 4. Remaining CellOracle runtime deps (pins from CellOracle 0.20.0 metadata; note the
#    pandas<=1.5.3 / matplotlib<3.7 downgrades relative to what velocyto pulled in).
$PIP install --no-build-isolation \
  "numpy==1.26.4" "scipy" "numba>=0.50.1" "matplotlib<3.7" "seaborn" "scikit-learn" \
  "h5py>=3.1.0" "pandas<=1.5.3,>=1.0.3" "umap-learn" "pyarrow>=0.17" "tqdm>=4.45" \
  "igraph>=0.10.1" "louvain" "anndata>=0.7.5,<=0.10.8" "scanpy>=1.6" "joblib" \
  "goatools" "genomepy>=0.8.4" ipython ipywidgets

# 5. CellOracle itself, without deps (they are satisfied above; --no-deps avoids pip
#    trying to rebuild the gimmemotifs pin, which would fail).
$PIP install celloracle --no-deps

# 6. Smoke test.
$PY - <<'PYEOF'
import celloracle as co
print("celloracle", co.__version__, "| Oracle:", hasattr(co, "Oracle"), "| Links:", hasattr(co, "Links"))
PYEOF
echo "Done. Run the benchmark with: $PY benchmarking/cell-oracle-benchmark.py"
