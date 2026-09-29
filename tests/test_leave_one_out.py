"""
Leave-one-out likelihoods for the parametric models ('gss', 'student', 'circ'): each stimulus is scored against its own
category's statistics without it. Checked against recomputing the statistics with the stimulus deleted.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.stats import multivariate_normal as smvn, multivariate_t as smvt

import ama
import stimuli as ts
from test_likelihood_extensions import model_inputs


def loo_reference(Rm,weights,c,k,centered=True,covShrink=0.,covTarget='diag'):
    """mean and covariance of category k without stimulus c, by brute force"""
    Rm,w=np.asarray(Rm),np.asarray(weights)>0
    scat,dofs=[],[]
    for i in range(w.shape[1]):
        keep=w[:,i].copy()
        if i==k:
            keep[c]=False
        X=Rm[:,keep,i]
        mu=X.mean(1) if centered else np.zeros(X.shape[0],dtype=X.dtype)
        D=X-mu[:,None]
        scat.append(D@D.conj().T); dofs.append(X.shape[1]-(1 if centered else 0))
        if i==k:
            muk=mu
    cov=scat[k]/dofs[k]
    if covShrink>0:
        T=sum(scat)/sum(dofs) if covTarget=='pooled' else np.diag(np.diag(cov))
        cov=(1-covShrink)*cov+covShrink*T
    return muk,cov


PICKS=[(0,0),(7,1),(19,2),(3,4)]


def deleted_reference(fun,R,Rm,RVar,noiseCov,w,unit,c,k):
    """
    lAll[c,k,:] of the plain model with stimulus (c,k) given weight 0: statistics and the noise covariance without it,
    and its own category's prior without it (log (N_k-1)/N_k relative to the others)
    """
    w0=np.asarray(w).copy()
    w0[c,k]=0
    w0=jnp.asarray(w0)
    out=np.array(fun(R,Rm,RVar,unit.nrn._corr_fun(RVar,w0,unit.nrn.rho),None,w0,False,unit.model,unit.stim.Y))[c,k]
    n=float(np.asarray(w)[:,k].sum())
    out[k]+=np.log((n-1)/n)
    return out


def loo_noise_prior(RVar,w,c,k):
    """the own category's (independent) noise covariance and log prior ratio without stimulus (c,k)"""
    keep=np.asarray(w)[:,k]>0
    keep[c]=False
    v=np.asarray(RVar)[:,keep,k].mean(1)
    n=keep.sum()+1
    return np.diag(v),np.log((n-1)/n)


POOLED=[dict(ctgPoolWidth=0.7),dict(ctgPoolWidth=0.7,bPoolMeans=True),dict(covShrink=0.4,covTarget='pooled'),
        dict(ctgPoolWidth=0.5,bPoolMeans=True,covShrink=0.3,covTarget='pooled'),dict(ctgPoolWidth=0.5,covShrink=0.3)]


class TestPooledLeaveOneOut:
    @pytest.mark.parametrize('kw',POOLED)
    def test_gss_matches_refitting_without_the_stimulus(self,kw):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=3,**kw)
        w=unit.stim.weights
        loo=np.asarray(ama.Model._model__gss(R,Rm,RVar,noiseCov,None,w,True,unit.model,unit.stim.Y))
        for c,k in PICKS:
            ref=deleted_reference(ama.Model._model__gss,R,Rm,RVar,noiseCov,w,unit,c,k)
            assert np.allclose(loo[c,k],ref,rtol=1e-9,atol=1e-9)

    def test_student_matches_refitting(self):
        unit,R,Rm,RVar,noiseCov=model_inputs('student',n=2,df=6.,ctgPoolWidth=0.7,bPoolMeans=True)
        w=unit.stim.weights
        loo=np.asarray(ama.Model._model__student(R,Rm,RVar,noiseCov,None,w,True,unit.model,unit.stim.Y))
        for c,k in PICKS:
            assert np.allclose(loo[c,k],deleted_reference(ama.Model._model__student,R,Rm,RVar,noiseCov,w,unit,c,k),rtol=1e-9,atol=1e-9)

    @pytest.mark.parametrize('circMean',['estimate','zero'])
    def test_circ_matches_refitting(self,circMean):
        unit,R,Rm,RVar,noiseCov=model_inputs('circ',gen=ts.sine_frequency,n=2,fourierType=2,circMean=circMean,
                                             ctgPoolWidth=2.,covShrink=0.2,covTarget='pooled')
        w=unit.stim.weights
        loo=np.asarray(ama.Model._model__circ(R,Rm,RVar,noiseCov,None,w,True,unit.model,unit.stim.Y))
        for c,k in [(0,0),(5,1),(11,2)]:
            assert np.allclose(loo[c,k],deleted_reference(ama.Model._model__circ,R,Rm,RVar,noiseCov,w,unit,c,k),rtol=1e-9,atol=1e-9)

    def test_unpooled_path_matches_refitting(self):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=3,covShrink=0.3)
        w=unit.stim.weights
        loo=np.asarray(ama.Model._model__gss(R,Rm,RVar,noiseCov,None,w,True,unit.model,unit.stim.Y))
        for c,k in PICKS:
            assert np.allclose(loo[c,k],deleted_reference(ama.Model._model__gss,R,Rm,RVar,noiseCov,w,unit,c,k),rtol=1e-9,atol=1e-9)

    def test_training_with_pooling(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('gss','mean',bLeaveOneOut=True,ctgPoolWidth=0.5,bPoolMeans=True),
                      ama.Objective('map'),ama.Optimizer(nIterMax=60,lRate0=0.05,batchSize=40,bVerbose=False))
        unit.train_new(2)
        h=np.asarray(unit.optimizer.loss_hist)
        assert np.all(np.isfinite(h)) and np.isfinite(float(unit.loss))


class TestParametricLeaveOneOut:
    @pytest.mark.parametrize('kw',[dict(),dict(covShrink=0.3)])
    def test_gss_matches_deletion(self,kw):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=3,**kw)
        w=unit.stim.weights
        loo=np.asarray(ama.Model._model__gss(R,Rm,RVar,noiseCov,None,w,True,unit.model,unit.stim.Y))
        plain=np.asarray(ama.Model._model__gss(R,Rm,RVar,noiseCov,None,w,False,unit.model,unit.stim.Y))
        N,Rn=np.asarray(noiseCov),np.asarray(R)
        for c,k in PICKS:
            mu,cov=loo_reference(Rm,w,c,k,**kw)
            NL,lp=loo_noise_prior(RVar,w,c,k)
            assert np.isclose(loo[c,k,k],smvn(mu,cov+NL).logpdf(Rn[:,c,k])+lp,rtol=1e-10)
        off=~np.eye(loo.shape[-1],dtype=bool)[None]&np.ones_like(loo,dtype=bool)
        assert np.allclose(loo[off],plain[off])                            # other categories are unchanged
        valid=np.asarray(w)>0
        assert np.mean(np.diagonal(loo,axis1=1,axis2=2)[valid]<np.diagonal(plain,axis1=1,axis2=2)[valid])>0.9

    def test_student_matches_deletion(self):
        unit,R,Rm,RVar,noiseCov=model_inputs('student',n=2,df=6.)
        loo=np.asarray(ama.Model._model__student(R,Rm,RVar,noiseCov,None,unit.stim.weights,True,unit.model,unit.stim.Y))
        N,Rn=np.asarray(noiseCov),np.asarray(R)
        for c,k in PICKS:
            mu,cov=loo_reference(Rm,unit.stim.weights,c,k)
            NL,lp=loo_noise_prior(RVar,unit.stim.weights,c,k)
            ref=smvt(mu,(cov+NL)*4/6,df=6.).logpdf(Rn[:,c,k])+lp
            assert np.isclose(loo[c,k,k],ref,rtol=1e-10)

    @pytest.mark.parametrize('circMean',['estimate','zero'])
    def test_circ_matches_deletion(self,circMean):
        unit,R,Rm,RVar,noiseCov=model_inputs('circ',gen=ts.sine_frequency,n=2,fourierType=2,circMean=circMean)
        loo=np.asarray(ama.Model._model__circ(R,Rm,RVar,noiseCov,None,unit.stim.weights,True,unit.model,unit.stim.Y))
        h=np.asarray(R).shape[0]//2
        Rc=np.asarray(R)[:h]+1j*np.asarray(R)[h:]
        Rmc=np.asarray(Rm)[:h]+1j*np.asarray(Rm)[h:]
        N=np.asarray(noiseCov)
        for c,k in [(0,0),(5,1),(11,2)]:
            mu,cov=loo_reference(Rmc,unit.stim.weights,c,k,centered=circMean=='estimate')
            NL,lp=loo_noise_prior(RVar,unit.stim.weights,c,k)
            C=cov+NL[:h,:h]+NL[h:,h:]
            z=Rc[:,c,k]-mu
            ref=-np.real(z.conj()@np.linalg.solve(C,z)) - h*np.log(np.pi) - np.log(np.real(np.linalg.det(C))) + lp
            assert np.isclose(loo[c,k,k],ref,rtol=1e-10)

    def test_padding_is_ignored(self):
        x,s,ci,Y,_=ts.unequal_counts()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('gss','mean',bLeaveOneOut=True),ama.Objective('map'),
                      ama.Optimizer(nIterMax=1,bVerbose=False))
        unit._finalize(2,np.arange(2),dtype=jnp.float64)
        f=np.random.default_rng(0).standard_normal(unit.filter._shape)
        unit.filter.out=jnp.asarray(f/np.linalg.norm(f,axis=0))
        assert np.isfinite(float(unit.loss))

    @pytest.mark.parametrize('modelType',['gss','student'])
    def test_gradient_matches_finite_difference(self,modelType):
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model(modelType,'mean',bLeaveOneOut=True),ama.Objective('map'),
                      ama.Optimizer(nIterMax=1,bVerbose=False))
        unit._finalize(2,np.arange(2),dtype=jnp.float64)
        lf=lambda f: unit._loss_fun_lrn({'f':f},unit.rng,unit.filter.prepped_jx,unit.filter._insert_index_jx,
                                        unit.stim.val,unit.stim.weights,unit.stim.yCtg,unit.stim.Y)
        f=jnp.asarray(np.random.default_rng(2).standard_normal(unit.filter._shape))
        g=jax.grad(lf)(f)
        for idx in [(0,0),(9,1)]:
            e=np.zeros(f.shape)
            e[idx]=1e-6
            assert np.isclose(g[idx],(lf(f+e)-lf(f-e))/2e-6,rtol=1e-4,atol=1e-7)

    def test_training_cost_tracks_held_out_cost(self):
        """with small categories and several filters, the in-sample cost is optimistic; leave-one-out is not"""
        x,s,ci,Y,_=ts.gaussian_ctg(nStimPerCtg=16,nCtg=5)
        stim=ama.Stim(x,s,ci,Y)
        xt,st,cit,_,_=ts.gaussian_ctg(nStimPerCtg=400,nCtg=5,seed=9)
        test=ama.Stim(xt,st,cit,Y)
        costs={}
        for loo in (False,True):
            unit=ama.Unit(stim,ama.Nrn(),ama.Model('gss','mean',bLeaveOneOut=loo),ama.Objective('map'),
                          ama.Optimizer(nIterMax=1,bVerbose=False))
            unit._finalize(4,np.arange(4),dtype=jnp.float64)
            f=np.random.default_rng(0).standard_normal(unit.filter._shape)
            unit.filter.out=jnp.asarray(f/np.linalg.norm(f,axis=0))
            costs[loo]=float(unit.loss)
            held=unit.evaluate(test)
        assert costs[False] < held
        assert abs(costs[True]-held) < abs(costs[False]-held)

    def test_learning_with_batches(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('gss','mean',bLeaveOneOut=True),ama.Objective('map'),
                      ama.Optimizer(nIterMax=60,lRate0=0.05,batchSize=60,nBatchMinCtg=3,bVerbose=False))
        unit.train_new(2)
        h=np.asarray(unit.optimizer.loss_hist)
        assert np.all(np.isfinite(h)) and h[-1]<h[0]

    def test_validation(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('gss','mean',bLeaveOneOut=True),ama.Objective('map'),
                      ama.Optimizer(nIterMax=1,batchSize=5,bVerbose=False))
        with pytest.raises(Exception,match='nBatchMinCtg'):
            unit.train_new(1)


MODELS=[('gss',dict(),{}),('student',dict(df=6.),{}),
        ('circ',dict(circMean='estimate'),dict(gen=ts.sine_frequency,n=2,fourierType=2)),
        ('circ',dict(circMean='zero'),dict(gen=ts.sine_frequency,n=2,fourierType=2))]


class TestRankOneLeaveOneOut:
    """bLooNoise=False: the own category's noise covariance is kept, and the left-out likelihood is a rank-one downdate"""

    @staticmethod
    def inputs(modelType,kw,gen_kw,seed=5):
        unit,R,Rm,RVar,noiseCov=model_inputs(modelType,bLeaveOneOut=True,bLooNoise=False,**kw,**gen_kw)
        rng=np.random.default_rng(seed)
        R=R+0.3*jnp.asarray(rng.standard_normal(R.shape))                   # observed responses away from the means
        return unit,R,Rm,RVar,noiseCov

    @pytest.mark.parametrize('modelType,kw,gen_kw',MODELS)
    def test_matches_deletion(self,modelType,kw,gen_kw):
        unit,R,Rm,RVar,noiseCov=self.inputs(modelType,kw,gen_kw)
        assert ama.Model._loo_rank_one(unit.model)
        w=unit.stim.weights
        fun=getattr(ama.Model,'_model__'+modelType)
        loo=np.asarray(fun(R,Rm,RVar,noiseCov,None,w,True,unit.model,unit.stim.Y))
        Rn,N=np.asarray(R),np.asarray(noiseCov)
        picks=PICKS if modelType!='circ' else [(0,0),(5,1),(11,2)]
        for c,k in picks:
            _,lp=loo_noise_prior(RVar,w,c,k)
            if modelType=='circ':
                h=Rn.shape[0]//2
                Rmc=np.asarray(Rm)[:h]+1j*np.asarray(Rm)[h:]
                mu,cov=loo_reference(Rmc,w,c,k,centered=kw['circMean']=='estimate')
                C=cov+N[k,:h,:h]+N[k,h:,h:]
                z=Rn[:h,c,k]+1j*Rn[h:,c,k]-mu
                ref=-np.real(z.conj()@np.linalg.solve(C,z))-h*np.log(np.pi)-np.log(np.real(np.linalg.det(C)))
            else:
                mu,cov=loo_reference(Rm,w,c,k)
                ref=(smvn(mu,cov+N[k]).logpdf(Rn[:,c,k]) if modelType=='gss' else
                     smvt(mu,(cov+N[k])*(kw['df']-2)/kw['df'],df=kw['df']).logpdf(Rn[:,c,k]))
            assert np.isclose(loo[c,k,k],ref+lp,rtol=1e-10)

    @pytest.mark.parametrize('modelType,kw,gen_kw',MODELS[:2]+[(m,k,dict(g,gen=ts.unequal_counts) if m!='circ' else g) for m,k,g in MODELS[:1]])
    def test_matches_general_path_with_correlated_noise(self,modelType,kw,gen_kw,monkeypatch):
        unit,R,Rm,RVar,_=self.inputs(modelType,kw,gen_kw)
        n=R.shape[0]
        A=np.random.default_rng(1).standard_normal((n,n))
        noiseCov=jnp.asarray(A@A.T/n+0.2*np.eye(n))
        w=unit.stim.weights
        fun=getattr(ama.Model,'_model__'+modelType)
        fast=np.asarray(fun(R,Rm,RVar,noiseCov,None,w,True,unit.model,unit.stim.Y))
        monkeypatch.setattr(ama.Model,'_loo_rank_one',staticmethod(lambda m: False))
        slow=np.asarray(fun(R,Rm,RVar,noiseCov,None,w,True,unit.model,unit.stim.Y))
        valid=np.asarray(w)>0
        assert np.all(np.isfinite(fast))
        assert np.allclose(fast[valid],slow[valid],rtol=1e-10,atol=1e-10)

    def test_general_path_otherwise(self):
        for kw in (dict(covShrink=0.2),dict(covRank=1),dict(ctgPoolWidth=0.5),dict(bLooNoise=True)):
            m=ama.Model('gss','mean',bLeaveOneOut=True,**dict(dict(bLooNoise=False),**kw))
            assert not ama.Model._loo_rank_one(m)

    def test_equals_exact_noise_without_response_dependent_noise(self):
        """with fano=0 the noise variance does not depend on the stimulus, so bLooNoise does not change the likelihoods"""
        x,s,ci,Y,_=ts.gaussian_ctg()
        stim=ama.Stim(x,s,ci,Y)
        costs=[]
        for bLooNoise in (True,False):
            unit=ama.Unit(stim,ama.Nrn(fano=0.),ama.Model('gss','mean',bLeaveOneOut=True,bLooNoise=bLooNoise),ama.Objective('map'),
                          ama.Optimizer(nIterMax=1,bVerbose=False))
            unit._finalize(3,np.arange(3),dtype=jnp.float64)
            f=np.random.default_rng(0).standard_normal(unit.filter._shape)
            unit.filter.out=jnp.asarray(f/np.linalg.norm(f,axis=0))
            costs.append(float(unit.loss))
        assert np.isclose(costs[0],costs[1],rtol=1e-12)

    @pytest.mark.parametrize('modelType',['gss','student'])
    def test_gradient_matches_finite_difference(self,modelType):
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model(modelType,'mean',bLeaveOneOut=True,bLooNoise=False),
                      ama.Objective('map'),ama.Optimizer(nIterMax=1,bVerbose=False))
        unit._finalize(2,np.arange(2),dtype=jnp.float64)
        lf=lambda f: unit._loss_fun_lrn({'f':f},unit.rng,unit.filter.prepped_jx,unit.filter._insert_index_jx,
                                        unit.stim.val,unit.stim.weights,unit.stim.yCtg,unit.stim.Y)
        f=jnp.asarray(np.random.default_rng(2).standard_normal(unit.filter._shape))
        g=jax.grad(lf)(f)
        for idx in [(0,0),(9,1)]:
            e=np.zeros(f.shape)
            e[idx]=1e-6
            assert np.isclose(g[idx],(lf(f+e)-lf(f-e))/2e-6,rtol=1e-4,atol=1e-7)

    def test_learning_with_batches(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('gss','mean',bLeaveOneOut=True,bLooNoise=False),ama.Objective('map'),
                      ama.Optimizer(nIterMax=60,lRate0=0.05,batchSize=60,nBatchMinCtg=3,bVerbose=False))
        unit.train_new(2)
        h=np.asarray(unit.optimizer.loss_hist)
        assert np.all(np.isfinite(h)) and h[-1]<h[0]
