"""
Expected cost under response noise (Model nNoiseSamples, bFixedNoise): the mean over independent noisy observations,
checked against averaging single-draw costs; its convergence; that the mean-response approximation underestimates it;
fixed noise; and training.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import ama
import stimuli as ts


def unit_for(modelType='gss',S=4,nrn=None,bFixedNoise=False,n=2,gen=ts.gaussian_ctg,responseType='basic',**opt):
    x,s,ci,Y,_=gen()
    unit=ama.Unit(ama.Stim(x,s,ci,Y),nrn or ama.Nrn(bNoise_2=True),
                  ama.Model(modelType,responseType,nNoiseSamples=S,bFixedNoise=bFixedNoise),ama.Objective('map'),
                  ama.Optimizer(**dict(dict(nIterMax=1,bVerbose=False),**opt)))
    unit._finalize(n,np.arange(n),dtype=jnp.float64)
    f=np.random.default_rng(0).standard_normal(unit.filter._shape)
    unit.filter.out=jnp.asarray(f/np.linalg.norm(f,axis=0))
    return unit


def lrn_cost(unit,key):
    flt=unit.filter
    prm={'f':flt.out_flat[flt._insert_index_jx]}
    return float(unit._loss_fun_lrn(prm,key,flt.prepped_jx,flt._insert_index_jx,unit.stim.val,unit.stim.weights,
                                    unit.stim.yCtg,unit.stim.Y))


class TestNoiseSamples:
    @pytest.mark.parametrize('modelType',['gss','full'])
    @pytest.mark.parametrize('nrn_kw',[dict(bNoise_2=True),dict(bNoise_1=True,normalizeType='broad')])
    def test_mean_of_single_draws(self,modelType,nrn_kw):
        S=5
        unit=unit_for(modelType,S=S,nrn=ama.Nrn(**nrn_kw))
        one=unit_for(modelType,S=1,nrn=ama.Nrn(**nrn_kw))
        key=jax.random.key(4)
        ref=np.mean([lrn_cost(one,k) for k in jax.random.split(key,S)])
        assert np.isclose(lrn_cost(unit,key),ref,rtol=1e-10)
        # unit.loss and evaluate average the same way (with the unit's key)
        ref=np.mean([float(one._loss_fun(one._params_out(),k,one.stim.val,one.stim.weights,one.stim.yCtg,one.stim.Y))
                     for k in jax.random.split(unit.rng,S)])
        assert np.isclose(float(unit.loss),ref,rtol=1e-10)
        held=np.mean([float(one._loss_fun_heldout(one._params_out(),k,one.stim.val,one.stim.weights,one.stim.yCtg,one.stim.Y,
                                                  one.stim.val,one.stim.weights,one.stim.yCtg)) for k in jax.random.split(unit.rng,S)])
        assert np.isclose(unit.evaluate(unit.stim),held,rtol=1e-10)

    def test_mean_response_approximation_is_optimistic(self):
        """decoding noise-free responses underestimates the expected cost of decoding noisy ones"""
        nrn=ama.Nrn(fano=2.,var0=0.5,bNoise_2=True)
        expected=float(unit_for(S=200,nrn=nrn).loss)
        mean=float(unit_for(S=1,nrn=nrn,responseType='mean').loss)
        assert mean<expected-0.05

    def test_converges(self):
        nrn=ama.Nrn(bNoise_2=True)
        costs=[float(unit_for(S=S,nrn=nrn).loss) for S in (100,400)]
        assert abs(costs[0]-costs[1])<0.02

    def test_fixed_noise_is_deterministic(self):
        unit=unit_for(bFixedNoise=True)
        assert lrn_cost(unit,jax.random.key(0))==lrn_cost(unit,jax.random.key(1))
        free=unit_for(bFixedNoise=False)
        assert lrn_cost(free,jax.random.key(0))!=lrn_cost(free,jax.random.key(1))

    def test_gradient_matches_finite_difference(self):
        unit=unit_for(bFixedNoise=True)
        flt=unit.filter
        lf=lambda f: unit._loss_fun_lrn({'f':f},jax.random.key(0),flt.prepped_jx,flt._insert_index_jx,unit.stim.val,
                                        unit.stim.weights,unit.stim.yCtg,unit.stim.Y)
        f=jnp.asarray(np.random.default_rng(2).standard_normal(flt._shape))
        g=jax.grad(lf)(f)
        for idx in [(0,0),(9,1)]:
            e=np.zeros(f.shape); e[idx]=1e-6
            assert np.isclose(g[idx],(lf(f+e)-lf(f-e))/2e-6,rtol=1e-4,atol=1e-7)

    @pytest.mark.parametrize('opt',[dict(),dict(optimizerType='lbfgs',nIterMax=30)])
    def test_training(self,opt):
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(bNoise_2=True),ama.Model('gss','basic',nNoiseSamples=8,bFixedNoise=True),
                      ama.Objective('map'),ama.Optimizer(**dict(dict(nIterMax=100,lRate0=0.05,bVerbose=False),**opt)))
        unit.train_new(1)
        h=np.asarray(unit.optimizer.loss_hist)
        assert np.all(np.isfinite(h)) and h[-1]<h[0]

    def test_validation(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        for nrn,model,match in [(ama.Nrn(bNoise_2=True),ama.Model('gss','mean',nNoiseSamples=4),'basic'),
                                (ama.Nrn(),ama.Model('gss','basic',nNoiseSamples=4),'bNoise')]:
            unit=ama.Unit(ama.Stim(x,s,ci,Y),nrn,model,ama.Objective('map'),ama.Optimizer(nIterMax=1,bVerbose=False))
            with pytest.raises(Exception,match=match):
                unit.train_new(1)
        with pytest.raises(Exception,match='nNoiseSamples'):
            ama.Model(nNoiseSamples=0)
