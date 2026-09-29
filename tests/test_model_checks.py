"""
Stimulus and batch checks, stage-1 noise in the likelihood, and leave-one-out full AMA.
"""
import warnings

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.special import logsumexp

import ama
import stimuli as ts
from test_extended import unit_from, ungroup


#- contrast normalization

class TestContrastNormalization:
    def test_helper_matches_test_generator(self):
        s=np.random.default_rng(0).standard_normal((6,5,7))
        assert np.allclose(ama.contrast_normalize(s),ts.contrast_normalize(s))

    @pytest.mark.parametrize('transform',[lambda s: 3*s, lambda s: s+0.1])
    def test_warns_for_unnormalized_stimuli(self,transform):
        x,s,ci,Y,_=ts.gaussian_ctg()
        with pytest.warns(UserWarning,match='contrast normalized'):
            ama.Stim(x,transform(s),ci,Y)

    def test_normalized_stimuli_do_not_warn(self,recwarn):
        x,s,ci,Y,_=ts.gaussian_ctg()
        ama.Stim(x,s,ci,Y)
        ama.Stim.gen_test()
        assert not [w for w in recwarn if 'contrast normalized' in str(w.message)]

    def test_bContrastNormalize(self,recwarn):
        x,s,ci,Y,_=ts.gaussian_ctg()
        a=ama.Stim(x,3*s+0.1,ci,Y,bContrastNormalize=True)
        assert not [w for w in recwarn if 'contrast normalized' in str(w.message)]
        assert np.allclose(np.asarray(a.val),np.asarray(ama.Stim(x,s,ci,Y).val))


#- batches

class TestBatchChecks:
    def batch_unit(self,**opt_kw):
        x,s,ci,Y,_=ts.gaussian_ctg()                                        # 5 categories x 40 stimuli
        return ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('gss','mean'),ama.Objective('map'),
                        ama.Optimizer(nIterMax=2,batchSize=10,bVerbose=False,**opt_kw))

    def test_small_batches_warn_for_gss(self):
        unit=self.batch_unit()
        with pytest.warns(UserWarning,match='full-rank'):
            unit.train_new(2)

    def test_nBatchMinCtg_sets_the_minimum(self,recwarn):
        unit=self.batch_unit(nBatchMinCtg=3)
        _,mask=unit.optimizer._batch_plan(unit.stim.weights)
        assert np.all(np.asarray(mask).sum(0)>=3)
        unit.train_new(2)
        assert not [w for w in recwarn if 'full-rank' in str(w.message)]


#- stage-1 noise in the likelihood

@pytest.mark.parametrize('bNoise_2',[False,True])
@pytest.mark.parametrize('normalizeType',['broad','gen'])
def test_likelihood_variance_matches_sampled_noise(normalizeType,bNoise_2):
    x,s,ci,Y,_=ts.unequal_counts()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        stim=ama.Stim(x,2*s,ci,Y)
    nrn=ama.Nrn(bNoise_1=True,bNoise_2=bNoise_2,normalizeType=normalizeType,eps=0.01,fano=0.1,var0=0.01)
    unit=unit_from(stim,nrn,responseType='basic')
    f=unit.filter.out_flat
    outs=jax.vmap(lambda k: unit.nrn.main(k,unit.stim.val,f))(jax.random.split(jax.random.key(0),6000))
    RNs=np.asarray(outs[3])
    RVar=np.asarray(outs[4][0])
    w=np.asarray(unit.stim.weights)>0
    assert np.all(RVar[:,w]>0)
    rel=np.abs(RNs.var(0)[:,w]/RVar[:,w]-1)
    if normalizeType=='broad':
        assert rel.max()<0.1                                                # exact up to sampling error
    else:
        # first order: underestimates by ~5-10% at this noise level (1/D^2 is convex), more for some responses
        assert np.median(rel)<0.10 and np.quantile(rel,0.9)<0.30


def test_stage1_noise_with_gen_normalization_rejects_quadrature_pairs():
    with pytest.raises(Exception,match='real responses'):
        unit_from(ama.Stim(*ts.sine_frequency()[:4]),ama.Nrn(bNoise_1=True,normalizeType='gen'),
                  fourierType=2,dtype=jnp.complex128)


#- leave-one-out full AMA

def loo_reference(unit):
    """numpy: Eq 5 posterior with the decoded stimulus removed from its own category"""
    nrn_out=unit.nrn.main(unit.rng,unit.stim.val,unit.filter.out_flat,unit.stim.weights)
    R,Rm,V=[np.asarray(ama._flatten_responses(x)) for x in unit.model._response_fun(*nrn_out)]
    w=unit.stim.weights
    R,lab=ungroup(R,w)
    Rm,_=ungroup(Rm,w)
    V,_=ungroup(V,w)
    LL=(-0.5*(((R[:,:,None]-Rm[:,None,:])**2)/V[:,None,:]).sum(0)
        -0.5*np.log(2*np.pi*V).sum(0)[None,:])                               # [ observed l x stimulus j ]
    np.fill_diagonal(LL,-np.inf)
    L=np.stack([logsumexp(LL[:,lab==i],axis=1) for i in range(lab.max()+1)],axis=1)
    lp=L-logsumexp(L,1,keepdims=True)
    return -lp[np.arange(len(lab)),lab].mean()


def loo_unit(nrn=None,responseType='mean',bLeaveOneOut=True,**kw):
    unit=unit_from(ama.Stim(*ts.unequal_counts()[:4]),nrn,'full',responseType,**kw)
    unit.model.bLeaveOneOut=bLeaveOneOut
    return unit


class TestLeaveOneOut:
    @pytest.mark.parametrize('responseType',['mean','basic'])
    @pytest.mark.parametrize('rho',[0.,0.3])
    def test_matches_reference(self,responseType,rho):
        unit=loo_unit(ama.Nrn(bNoise_2=responseType=='basic',rho=rho),responseType)
        if rho==0:
            assert np.isclose(float(unit.loss),loo_reference(unit),rtol=1e-8)
        else:
            assert np.isfinite(float(unit.loss))

    def test_self_match_makes_cost_optimistic(self):
        nrn=ama.Nrn(fano=0.,var0=1e-3)
        plain=float(loo_unit(nrn,bLeaveOneOut=False).loss)
        loo=float(loo_unit(ama.Nrn(fano=0.,var0=1e-3)).loss)
        assert loo>plain+0.1

    def test_gradients_are_finite(self):
        unit=loo_unit()
        g=jax.grad(lambda f: unit._loss_fun({'f':f},unit.rng,unit.stim.val,unit.stim.weights,unit.stim.yCtg,unit.stim.Y))(unit.filter.out_flat)
        assert np.all(np.isfinite(np.asarray(g)))

    def test_learning_with_batches(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('full','mean',bLeaveOneOut=True),ama.Objective('map'),
                      ama.Optimizer(nIterMax=60,lRate0=0.05,batchSize=60,bVerbose=False))
        unit.train_new(2)
        h=unit.optimizer.loss_hist
        assert np.all(np.isfinite(h))
        assert np.isfinite(float(unit.loss))

    @pytest.mark.parametrize('modelType,errType,counts,match',[
        ('gss','map',(2,30,20),'at least 3'),
        ('full','mle',(10,30,20),'mle'),
        ('full','map',(1,30,20),'at least 2'),
    ])
    def test_validation(self,modelType,errType,counts,match):
        x,s,ci,Y,_=ts.unequal_counts(counts=counts)
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model(modelType,'mean',bLeaveOneOut=True),ama.Objective(errType),
                      ama.Optimizer(nIterMax=1,bVerbose=False))
        with pytest.raises(Exception,match=match):
            unit._finalize(1,[0])
