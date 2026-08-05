#!/usr/bin/env bash
# Reproducible setup for the `scenic_env` conda environment.
#
# pySCENIC 0.12.1 (last released 2022) does not run on a modern numpy/pandas/scikit-learn/
# dask stack -- each pin below fixed one import-time crash encountered getting
# comparison_scenic_fit_6667.py to actually run:
#   1. ctxcore imports pkg_resources -> needs setuptools<81.
#   2. pyscenic.transform uses the removed `np.object` alias -> needs numpy<1.24.
#   3. arboreto 0.1.6 (pySCENIC's GRN step) predates modern dask's `from_delayed` API and
#      silently produces zero delayed tasks on today's dask -> pin dask+distributed to the
#      contemporaneous 2021.11.2 release.
#   4. That old dask, in turn, doesn't understand pandas>=1.4's internal string-accessor
#      layout -> needs pandas<1.4 to match.
#   5. A pandas/dask this old needs a correspondingly older scipy/scikit-learn (numba, used
#      by ctxcore's aucell/recovery module, sets a numpy>=1.22 floor from the other side --
#      numpy==1.22.4 is the version that satisfies every constraint simultaneously).
#
# Usage:  bash benchmarking/setup_scenic_env.sh
set -euo pipefail

CONDA=/opt/anaconda3/bin/conda
ENV=scenic_env
PY=/opt/anaconda3/envs/$ENV/bin/python
PIP=/opt/anaconda3/envs/$ENV/bin/pip

$CONDA create -y -n $ENV python=3.10

$PIP install pyscenic
$PIP install "setuptools<81"
$PIP install "numpy==1.22.4" "pandas==1.3.5" "scipy==1.7.3" "scikit-learn==1.0.2" \
             "dask==2021.11.2" "distributed==2021.11.2" "cloudpickle<2" "tornado<6.2" "click<8.1"

# Smoke test.
/opt/anaconda3/envs/$ENV/bin/pyscenic --help > /dev/null
echo "Done. See comparison_scenic_fit_6667.py for the cisTarget database download + run commands."
