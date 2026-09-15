"""
Package-level checks: import cost.
"""
import os
import subprocess
import sys

from conftest import ROOT


def test_import_does_not_load_plotting_libraries():
    code=("import sys, ama; "
          "print(','.join(m for m in ('matplotlib','sklearn','Filter','statsmodels') if m in sys.modules))")
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([ROOT,os.environ.get('PYTHONPATH','')]))
    out=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=env,check=True)
    assert out.stdout.strip()==''
