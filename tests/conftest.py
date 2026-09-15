import os
import sys

# headless plotting
os.environ.setdefault('MPLBACKEND', 'Agg')

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402

# exact comparisons against numpy/scipy references
jax.config.update('jax_enable_x64', True)

import matplotlib  # noqa: E402

# plotting tests open many figures on purpose
matplotlib.rcParams['figure.max_open_warning'] = 0

REFERENCE_MAT = os.path.join(ROOT, 'AMAdataDisparity.mat')


@pytest.fixture(scope='session')
def reference_mat():
    if not os.path.exists(REFERENCE_MAT):
        pytest.skip('AMAdataDisparity.mat not present')
    from scipy.io import loadmat
    return loadmat(REFERENCE_MAT)
