"""
Factor-analysis (low-rank plus diagonal) category covariances (Model covRank): the EM fit against sklearn, the
likelihoods, leave-one-out, mixtures, gradients, and held-out decoding with few stimuli per response dimension.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.stats import multivariate_normal as smvn
from sklearn.decomposition import FactorAnalysis

import ama
import stimuli as ts
from test_likelihood_extensions import model_inputs
from test_leave_one_out import deleted_reference, PICKS


def fa_data(n=500,p=8,r=2,seed=0):
    rng=np.random.default_rng(seed)
    L=rng.standard_normal((p,r))
    psi=rng.uniform(0.2,1.,p)
    X=rng.standard_normal((n,r))@L.T + rng.standard_normal((n,p))*np.sqrt(psi)
    return X


class TestFactorAnalysis:
    def test_matches_sklearn(self):
        X=fa_data()
        S=np.cov(X.T,bias=True)                                     # maximum likelihood, as sklearn fits
        C=np.asarray(ama.Model._low_rank(jnp.asarray(S),ama.Model('gss',covRank=2,nFA=3000)))
        ref=FactorAnalysis(2,tol=1e-10,max_iter=10000,svd_method='lapack').fit(X).get_covariance()
        assert np.allclose(C,ref,atol=1e-3*np.abs(ref).max())
        # the fit is the maximum likelihood one: at least as likely as sklearn's
        ll=lambda C: smvn(X.mean(0),C).logpdf(X).sum()
        assert ll(C)>=ll(ref)-1e-3

    def test_rank_zero_is_diagonal_and_batches(self):
        S=np.stack([np.cov(fa_data(seed=s).T) for s in range(3)])
        D=np.asarray(ama.Model._low_rank(jnp.asarray(S),ama.Model('gss',covRank=0)))
        assert np.allclose(D,np.stack([np.diag(np.diag(x)) for x in S]))
        C=np.asarray(ama.Model._low_rank(jnp.asarray(S),ama.Model('gss',covRank=2)))
        C1=np.asarray(ama.Model._low_rank(jnp.asarray(S[1]),ama.Model('gss',covRank=2)))
        assert np.allclose(C[1],C1)


class TestModels:
    def test_gss_likelihood(self):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=4)
        m=ama.Model('gss',covRank=1)
        w=unit.stim.weights
        lAll=np.asarray(ama.Model._model__gss(R,Rm,RVar,noiseCov,None,w,False,m,unit.stim.Y))
        mu,cov=ama.Model._ctg_stats(Rm,w,ama.Model('gss'),unit.stim.Y)
        C=np.asarray(ama.Model._low_rank(cov,m))
        N,Rn,mu=np.asarray(noiseCov),np.asarray(R),np.asarray(mu)
        for (l,k,i) in [(0,0,0),(5,1,3),(9,4,2)]:
            assert np.isclose(lAll[l,k,i],smvn(mu[:,i],C[i]+N[i]).logpdf(Rn[:,l,k]))

    @pytest.mark.parametrize('kw',[dict(),dict(ctgPoolWidth=0.7)])
    def test_leave_one_out_matches_refitting(self,kw):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=3,covRank=1,**kw)
        w=unit.stim.weights
        loo=np.asarray(ama.Model._model__gss(R,Rm,RVar,noiseCov,None,w,True,unit.model,unit.stim.Y))
        for c,k in PICKS[:2]:
            ref=deleted_reference(ama.Model._model__gss,R,Rm,RVar,noiseCov,w,unit,c,k)
            assert np.allclose(loo[c,k][k] if not kw else loo[c,k],ref[k] if not kw else ref,rtol=1e-6,atol=1e-6)

    def test_mixture_components(self):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=4)
        w=unit.stim.weights
        g=np.asarray(ama.Model._model__gss(R,Rm,RVar,noiseCov,None,w,False,ama.Model('gss',covRank=1),unit.stim.Y))
        m=np.asarray(ama.Model._model__mix(R,Rm,RVar,noiseCov,None,w,False,ama.Model('mix',nMix=1,mixReg=0.,covRank=1),unit.stim.Y))
        assert np.allclose(g,m,rtol=1e-6,atol=1e-6)

    def test_gradient_matches_finite_difference(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('gss','mean',covRank=1),ama.Objective('map'),
                      ama.Optimizer(nIterMax=1,bVerbose=False))
        unit._finalize(3,np.arange(3),dtype=jnp.float64)
        lf=lambda f: unit._loss_fun_lrn({'f':f},unit.rng,unit.filter.prepped_jx,unit.filter._insert_index_jx,
                                        unit.stim.val,unit.stim.weights,unit.stim.yCtg,unit.stim.Y)
        f=jnp.asarray(np.random.default_rng(2).standard_normal(unit.filter._shape))
        g=jax.grad(lf)(f)
        for idx in [(0,0),(9,2)]:
            e=np.zeros(f.shape); e[idx]=1e-6
            assert np.isclose(g[idx],(lf(f+e)-lf(f-e))/2e-6,rtol=1e-4,atol=1e-7)

    @pytest.mark.filterwarnings('ignore:stimuli are not contrast normalized')
    def test_few_stimuli_per_dimension_decode_better(self):
        """8 response dimensions, 12 training stimuli per category, responses with a rank-2 shared structure: the
        factor model's covariances generalize better than the sample covariances"""
        rng=np.random.default_rng(0)
        nPix,nCtg,nF=24,4,8
        basis=np.linalg.qr(rng.standard_normal((nPix,nPix)))[0]
        F=basis[:,:nF]
        A=basis[:,:nF]@rng.standard_normal((nF,2))*0.25
        M=basis[:,:nF]@rng.standard_normal((nF,nCtg))*0.02         # overlapping categories
        def gen(n,seed):
            r=np.random.default_rng(seed)
            ci=np.repeat(np.arange(nCtg),n)
            s=M[:,ci]+A@r.standard_normal((2,len(ci)))+0.03*r.standard_normal((nPix,len(ci)))
            return ama.Stim(np.arange(nPix),s,ci,np.arange(nCtg,dtype=float)),ci
        train,_=gen(12,1)
        test,_=gen(300,2)
        held={}
        for name,kw in [('full',{}),('rank2',{'covRank':2})]:
            unit=ama.Unit(train,ama.Nrn(var0=0.01,fano=0.01),ama.Model('gss','mean',**kw),ama.Objective('map'),
                          ama.Optimizer(nIterMax=1,bVerbose=False))
            unit._finalize(nF,np.arange(nF),dtype=jnp.float64)
            unit.filter.out=jnp.asarray(F)
            held[name]=unit.evaluate(test)
        assert held['rank2']<held['full']

    def test_validation(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        for model,match in [(ama.Model('gss','mean',covRank=2),'below the number'),(ama.Model('full','mean',covRank=1),'covRank')]:
            unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),model,ama.Objective('map'),ama.Optimizer(nIterMax=1,bVerbose=False))
            with pytest.raises(Exception,match=match):
                unit._finalize(2,np.arange(2))
