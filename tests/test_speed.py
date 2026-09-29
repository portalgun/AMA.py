"""
burgelab's second reference set, speed estimation (AMAdataSpeed.mat: 10,500 space-time stimuli, 21 speeds, and their
AMA-Gauss run with 4 filters). Skipped when the file is not present (see conftest.py).
"""
import numpy as np
import jax.numpy as jnp

import ama
from conftest import SPEED_MAT


def nrn_for(A):
    prm=A['paramRSP'][0,0]
    return ama.Nrn(fano=prm['fano'].item(),var0=prm['var0'].item(),rmax=prm['rMax'].item())


def test_reference_cost(speed_mat):
    """the AMA-Gauss cost of burgelab's filters matches their reported costs E for 1 to 4 filters"""
    A=speed_mat['AMA'][0,0]
    st=ama.Stim.load(SPEED_MAT)
    for nF in range(1,5):
        unit=ama.Unit(st,nrn_for(A),ama.Model('gss','mean'),ama.Objective('map'),ama.Optimizer(nIterMax=1))
        unit._finalize(nF,np.arange(nF),dtype=jnp.float64)
        unit.filter.out=jnp.asarray(A['f'][:,:nF])
        assert abs(float(unit.loss)-A['E'].ravel()[nF-1])<0.03


def test_training_reaches_reference_cost(speed_mat):
    """2 + 2 appended filters, refined, reach a lower cost than burgelab's 4 filters (under this cost)"""
    A=speed_mat['AMA'][0,0]
    st=ama.Stim.load(SPEED_MAT)
    ref=ama.Unit(st,nrn_for(A),ama.Model('gss','mean'),ama.Objective('map'),ama.Optimizer(nIterMax=1))
    ref._finalize(4,np.arange(4),dtype=jnp.float64)
    ref.filter.out=jnp.asarray(A['f'])
    # speed has a worse optimum (~1.85) that joint training always ends in, and that 2 + 2 appended filters reach in about
    # 5 of 8 starts (README, training schedule): try starts until one reaches the better one (~1.822)
    for seed in range(8):
        unit=ama.Unit(st,nrn_for(A),ama.Model('gss','mean'),ama.Objective('map'),
                      ama.Optimizer(nIterMax=400,lRate0=0.02,bVerbose=False),seed=666+seed)
        unit.train_new(2)
        unit.train_append(2)
        unit.train_recurse()
        if float(unit.loss)<float(ref.loss):
            break
    assert float(unit.loss)<float(ref.loss)
