"""
Response budget (Nrn respBudget): each final response dimension has a fixed root mean square over the training stimuli,
so the size of the responses (rmax, a normalization pool's scale) can not buy signal-to-noise ratio against the fixed
additive noise. Checked against direct computations.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import ama
import stimuli as ts


def unit_for(nrn,n=2,modelType='gss',responseType='mean',gen=ts.gaussian_ctg,stim=None,seed=0,**model_kw):
    if stim is None:
        x,s,ci,Y,_=gen()
        stim=ama.Stim(x,s,ci,Y)
    unit=ama.Unit(stim,nrn,ama.Model(modelType,responseType,**model_kw),ama.Objective('map'),
                  ama.Optimizer(nIterMax=1,bVerbose=False))
    unit._finalize(n,np.arange(n),dtype=jnp.float64)
    f=np.random.default_rng(seed).standard_normal(unit.filter._shape)
    unit.filter.out=jnp.asarray(f/np.linalg.norm(f,axis=0))
    return unit


class TestBudget:
    @pytest.mark.parametrize('nrn_kw',[dict(),dict(normalizeType='gen'),dict(bSplitNegatives=True)])
    def test_root_mean_square(self,nrn_kw):
        unit=unit_for(ama.Nrn(respBudget=1.7,**nrn_kw),gen=ts.unequal_counts)
        R=np.asarray(unit._nrn_out()[2])
        w=np.asarray(unit.stim.weights)>0
        rms=np.sqrt(np.mean(np.abs(R[...,w])**2,axis=-1))
        assert np.allclose(rms,1.7,rtol=1e-10)

    @pytest.mark.parametrize('modelType',['gss','full'])
    def test_cost_does_not_depend_on_the_response_scale(self,modelType):
        costs={}
        for budget in (None,2.):
            for rmax in (5.7,11.4):
                unit=unit_for(ama.Nrn(rmax=rmax,respBudget=budget),modelType=modelType)
                costs[budget,rmax]=float(unit.loss)
        assert np.isclose(costs[2.,5.7],costs[2.,11.4],rtol=1e-10)
        assert costs[None,11.4]<costs[None,5.7]-0.01                     # without it, larger responses beat the noise

    def test_normalization_pool_scale_is_irrelevant(self):
        M=np.abs(np.random.default_rng(1).standard_normal((2,2)))
        costs=[float(unit_for(ama.Nrn(normalizeType='gen',normPool=c*M,eps=1e-12,respBudget=1.)).loss) for c in (1.,0.1)]
        assert np.isclose(costs[0],costs[1],rtol=1e-8)

    def test_stage_one_variance_is_scaled(self):
        kw=dict(bNoise_1=True,normalizeType='broad')
        plain=unit_for(ama.Nrn(**kw))
        budget=unit_for(ama.Nrn(respBudget=1.3,**kw))
        R0,V0=[np.asarray(a) for a in plain._nrn_out()[2::2]]
        R1,V1=[np.asarray(a) for a in budget._nrn_out()[2::2]]
        w=np.asarray(plain.stim.weights)>0
        G=1.3/np.sqrt(np.mean(R0[...,w]**2,axis=-1))[:,None,None]
        assert np.allclose(R1,G*R0,rtol=1e-10)
        assert np.allclose(V1,G**2*V0,rtol=1e-10)

    def test_held_out_stimuli_use_the_training_gains(self):
        x,s,ci,Y,_=ts.gaussian_ctg(nStimPerCtg=60)
        train,test=ama.Stim(x,s,ci,Y).train_test(0.3)
        unit=unit_for(ama.Nrn(respBudget=1.),stim=train)
        f=unit.filter.out_flat
        Gtr=unit.nrn.gain(unit.stim.val,f,unit.stim.weights)
        tst=unit._prepare_stim(test)
        assert not np.allclose(np.asarray(Gtr),np.asarray(unit.nrn.gain(tst.val,f,tst.weights)))
        # evaluate = the model of the training responses, applied to the test responses scaled by the training gains
        obs=unit.nrn.main(unit.rng,tst.val,f,tst.weights,None,None,Gtr)
        ref=unit.nrn.main(unit.rng,unit.stim.val,f,unit.stim.weights,None,None,Gtr)
        lAll,_=unit._lik_parts(unit._likelihoods_heldout(obs,ref,unit.stim.weights,unit.stim.Y))
        ref_cost=float(unit.objective.lrn_main(lAll,tst.weights,tst.yCtg,tst.Y,unit.stim.weights))
        assert np.isclose(unit.evaluate(test),ref_cost,rtol=1e-12)
        assert np.isclose(unit.evaluate(train),float(unit.loss),rtol=1e-10)

    def test_gradient_matches_finite_difference(self):
        unit=unit_for(ama.Nrn(normalizeType='gen',respBudget=1.,bLearnNormPool=True))
        flt=unit.filter
        p=unit.nrn.params0()
        lf=lambda f: unit._loss_fun_lrn({'f':f,'p':p},unit.rng,flt.prepped_jx,flt._insert_index_jx,unit.stim.val,
                                        unit.stim.weights,unit.stim.yCtg,unit.stim.Y)
        f=jnp.asarray(np.random.default_rng(2).standard_normal(flt._shape))
        g=jax.grad(lf)(f)
        for idx in [(0,0),(9,1)]:
            e=np.zeros(f.shape); e[idx]=1e-6
            assert np.isclose(g[idx],(lf(f+e)-lf(f-e))/2e-6,rtol=1e-4,atol=1e-7)

    def test_training_with_a_learned_pool(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(normalizeType='gen',bLearnNormPool=True,respBudget=1.),ama.Model('gss','mean'),
                      ama.Objective('map'),ama.Optimizer(nIterMax=100,lRate0=0.05,bVerbose=False))
        unit.train_new(2)
        h=np.asarray(unit.optimizer.loss_hist)
        assert np.all(np.isfinite(h)) and h[-1]<h[0]

    def test_validation_and_copy(self):
        with pytest.raises(Exception,match='respBudget'):
            ama.Nrn(respBudget=0.)
        assert ama.Nrn(respBudget=2.).copy().respBudget==2.
        assert ama.Nrn(respBudget=2.)._key()!=ama.Nrn(respBudget=3.)._key()
