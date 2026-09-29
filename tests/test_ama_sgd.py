"""
optimizerType='ama_sgd': the AMA-SGD step of Burge & Jaini (2017) / burgelab/AMA. Each filter moves a decaying
distance along its unit-normalized tangent-plane gradient, and a step is kept only if it does not raise the batch cost.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import ama
import stimuli as ts


def unit_at(f0,**opt_kw):
    """a unit whose filters are f0 [ nPix x nF ], ready to refine with train_recurse"""
    x,s,ci,Y,info=ts.gaussian_ctg()
    unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('gss','mean'),ama.Objective('map'),
                  ama.Optimizer('ama_sgd',**{'nIterMax':1,'bVerbose':False,**opt_kw}))
    unit._finalize(f0.shape[1],np.arange(f0.shape[1]),dtype=jnp.float64)
    unit.filter.out=jnp.asarray(f0/np.linalg.norm(f0,axis=0))
    return unit,info


def cost(unit,f):
    return float(unit._loss_fun({'f':jnp.asarray(f)},unit.rng,unit.stim.val,unit.stim.weights,unit.stim.yCtg,unit.stim.Y))


def expected_step(unit,f,eps):
    """the burgelab update: per-filter tangent gradient, normalized, a step of eps, renormalized; kept if not worse"""
    g=np.asarray(jax.grad(lambda f: unit._loss_fun({'f':f},unit.rng,unit.stim.val,unit.stim.weights,unit.stim.yCtg,
                                                     unit.stim.Y))(jnp.asarray(f)))
    g=g-f*np.sum(f*g,axis=0)
    g=g/np.linalg.norm(g,axis=0)
    cand=f-eps*g
    cand=cand/np.linalg.norm(cand,axis=0)
    return cand if cost(unit,cand)<=cost(unit,f) else f


class TestStep:
    @pytest.mark.parametrize('eps',[0.05,1.2])
    def test_one_step_matches_the_burgelab_update(self,eps):
        f0=np.random.default_rng(0).standard_normal((16,2))
        unit,_=unit_at(f0,lRate0=eps)
        f=np.asarray(unit.out)
        exp=expected_step(unit,f,eps)
        unit.train_recurse()
        assert np.allclose(np.asarray(unit.out),exp,atol=1e-10)

    def test_a_step_that_raises_the_cost_is_rejected(self):
        # at the optimum (the informative direction), any sizable step raises the cost
        _,info=unit_at(np.ones((16,1)))
        unit,_=unit_at(info['direction'][:,None],lRate0=0.5)
        f=np.asarray(unit.out)
        unit.train_recurse()
        assert np.array_equal(np.asarray(unit.out),f)

    def test_step_size_schedule(self):
        tx=ama._ama_sgd(0.1,0.02,0.1)
        g={'f':jnp.asarray(np.random.default_rng(1).standard_normal((5,3)))}
        state=tx.init(g)
        for t in range(25):
            upd,state=tx.update(g,state)
            assert np.allclose(np.linalg.norm(np.asarray(upd['f']),axis=0),max(0.02,0.1*0.9**t))
        assert int(state['count'])==25


class TestTraining:
    def test_full_batch_cost_never_increases(self):
        unit,_=unit_at(np.random.default_rng(2).standard_normal((16,2)),lRate0=0.1,nIterMax=60,nStepsPerChunk=20)
        unit.train_recurse()
        h=np.asarray(unit.optimizer.loss_hist)
        assert np.all(np.diff(h)<=1e-12) and h[-1]<h[0]

    def test_ama_sgd_finds_the_informative_direction(self):
        x,s,ci,Y,info=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('gss','mean'),ama.Objective('map'),
                      ama.Optimizer('ama_sgd',lRate0=0.1,stepDecay=0.01,nIterMax=300,batchSize=50,bVerbose=False))
        unit.train_new(1)
        assert abs(float(np.asarray(unit.out).ravel()@info['direction']))>0.95

    def test_settings_and_validation(self):
        opt=ama.Optimizer('ama_sgd',lRate0=0.2,stepMin=0.01,stepDecay=0.05,bVerbose=False)
        c=opt.copy()
        assert (c.optimizerType,c.lRate0,c.stepMin,c.stepDecay)==('ama_sgd',0.2,0.01,0.05)
        with pytest.raises(Exception,match='l2_sphere'):
            ama.Optimizer('ama_sgd',projectionType=['l2_ball',1])
        with pytest.raises(Exception,match='stepDecay'):
            ama.Optimizer('ama_sgd',stepDecay=1.)
        x,s,ci,Y,_=ts.sine_frequency()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('gss','mean'),ama.Objective('map'),
                      ama.Optimizer('ama_sgd',nIterMax=2,bVerbose=False))
        with pytest.raises(Exception,match='generated filters'):
            unit.train_parametric(2)
