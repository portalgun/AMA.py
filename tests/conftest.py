import os
import sys

# headless plotting
os.environ.setdefault('MPLBACKEND', 'Agg')

# GPU runs: XLA's GPU reductions are otherwise nondeterministic, and tests that train twice (e.g. replaying a configuration)
# compare the results; and XLA's Triton GEMM fusions fail to compile some small dot products on newer GPUs (RTX 50xx,
# 'No supported config found for HLO ... gemm_fusion'), which cuBLAS handles
# tests that run python in a subprocess (test_package) need GPU memory too: allocate on demand instead of 75% up front
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
for _flag in ('--xla_gpu_deterministic_ops=true', '--xla_gpu_enable_triton_gemm=false'):
    if _flag.split('=')[0] not in os.environ.get('XLA_FLAGS', ''):
        os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' ' + _flag).strip()

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402

# exact comparisons against numpy/scipy references
jax.config.update('jax_enable_x64', True)
# full float32 matrix products on GPUs, which otherwise use TF32 (about 3 decimal digits) for them
jax.config.update('jax_default_matmul_precision', 'highest')

import matplotlib  # noqa: E402

# plotting tests open many figures on purpose
matplotlib.rcParams['figure.max_open_warning'] = 0

REFERENCE_MAT = os.path.join(ROOT, 'AMAdataDisparity.mat')
# burgelab's speed set (42 MB, not in the repository):
#   curl -L -o AMAdataSpeed.mat https://raw.githubusercontent.com/burgelab/AMA/master/AMAdataSpeed.mat
SPEED_MAT = os.path.join(ROOT, 'AMAdataSpeed.mat')


@pytest.fixture(scope='session')
def reference_mat():
    if not os.path.exists(REFERENCE_MAT):
        pytest.skip('AMAdataDisparity.mat not present')
    from scipy.io import loadmat
    return loadmat(REFERENCE_MAT)


@pytest.fixture(scope='session')
def speed_mat():
    if not os.path.exists(SPEED_MAT):
        pytest.skip('AMAdataSpeed.mat not present (see conftest.py)')
    from scipy.io import loadmat
    return loadmat(SPEED_MAT)
