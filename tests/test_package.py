"""
Package-level checks: import cost and the runnable example.
"""
import os
import subprocess
import sys

from conftest import ROOT, REFERENCE_MAT

import pytest


def run_python(args,**env_extra):
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([ROOT,os.environ.get('PYTHONPATH','')]),**env_extra)
    return subprocess.run([sys.executable,*args],capture_output=True,text=True,env=env,cwd=ROOT)


def test_import_does_not_load_plotting_libraries():
    code=("import sys, ama; "
          "print(','.join(m for m in ('matplotlib','sklearn','Filter','statsmodels') if m in sys.modules))")
    out=run_python(['-c',code])
    assert out.returncode==0,out.stderr
    assert out.stdout.strip()==''


@pytest.mark.skipif(not os.path.exists(REFERENCE_MAT),reason='AMAdataDisparity.mat not present')
def test_quickstart_example_runs(tmp_path):
    out=run_python([os.path.join(ROOT,'examples','quickstart.py')],
                   AMA_EXAMPLE_ITERS='20',AMA_EXAMPLE_OUT=str(tmp_path),MPLBACKEND='Agg')
    assert out.returncode==0,out.stderr[-2000:]
    assert 'held-out cost' in out.stdout
    assert (tmp_path/'disparity_unit.pkl').exists()
