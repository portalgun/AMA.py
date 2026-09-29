"""
Continuous latent variables: stimuli with their own latent values (Stim y, Stim.binned), errors and gaussian targets
measured from them, and continuous estimates within the categories (Model bWithin), against independent computations.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.special import logsumexp

import ama
import stimuli as ts


def binned(nBins=5,period=None,**kw):
    x,s,y,info=ts.continuous_gaussian(period=period,**kw)
    return ama.Stim.binned(x,s,y,nBins=nBins,Yperiod=period),y,info


def unit_for(stim,modelType='gss',errType='l2',nrn=None,n=1,seed=1,objective=None,**model_kw):
    unit=ama.Unit(stim,nrn or ama.Nrn(),ama.Model(modelType,'mean',**model_kw),objective or ama.Objective(errType),
                  ama.Optimizer(nIterMax=1,bVerbose=False))
    unit._finalize(n,np.arange(n),dtype=jnp.float64)
    f=np.random.default_rng(seed).standard_normal(unit.filter._shape)
    unit.filter.out=jnp.asarray(f/np.linalg.norm(f,axis=0))
    return unit


def flat_responses(unit):
    out=unit.nrn.main(unit.rng,unit.stim.val,unit.filter.out_flat,unit.stim.weights)
    return [np.asarray(ama._flatten_responses(v)) for v in unit.model._response_fun(*out)]


def valid(unit):
    w=np.asarray(unit.stim.weights)>0
    return w,np.asarray(unit.stim.yCtg)


class TestStim:
    def test_binned_levels_and_values(self):
        x,s,y,_=ts.continuous_gaussian()
        st=ama.Stim.binned(x,s,y,nBins=4)
        w,yc=np.asarray(st.weights)>0,np.asarray(st.yCtg)
        assert st.nCtg==4 and st.bContinuous
        assert np.allclose(np.sort(yc[w]),np.sort(y))                    # every stimulus keeps its own value
        for c in range(4):
            assert np.isclose(st.Y[c],yc[w[:,c],c].mean())                 # levels are the bin means
            assert np.all(yc[w[:,c],c]>=st.binEdges[0][c]-1e-12)
        counts=w.sum(0)
        assert counts.max()-counts.min()<=1                                # quantile bins: equal counts

    def test_uniform_bins_and_like(self):
        x,s,y,_=ts.continuous_gaussian()
        st=ama.Stim.binned(x,s,y,nBins=4,binType='uniform')
        assert np.allclose(np.diff(st.binEdges[0]),np.diff(st.binEdges[0])[0])
        train,test=st.train_test(0.3)
        # a separately binned copy with the same edges and levels can be evaluated with a unit trained on st
        again=ama.Stim.binned(x,s,y,like=st)
        assert np.array_equal(np.asarray(again.Y),np.asarray(st.Y)) and np.array_equal(np.asarray(again.yCtg),np.asarray(st.yCtg))

    def test_circular_bins_wrap(self):
        x,s,y,_=ts.continuous_gaussian(period=180.)
        st=ama.Stim.binned(x,s,y,nBins=6,Yperiod=180.)
        w,yc=np.asarray(st.weights)>0,np.asarray(st.yCtg)
        # the first bin is centered at 0: values just below 180 belong to it
        c0=int(np.argmin(np.abs(ama._wrap(jnp.asarray(st.Y),(180.,)))))
        near=y[(y>180-15)|(y<15)]
        assert np.allclose(np.sort(yc[w[:,c0],c0]),np.sort(near))

    def test_several_dimensions(self):
        x,s,y,_=ts.continuous_gaussian()
        y2=np.stack([y,np.random.default_rng(0).uniform(0,1,len(y))],1)
        st=ama.Stim.binned(x,s,y2,nBins=(3,2))
        assert st.nCtg==6 and np.asarray(st.yCtg).shape[-1]==2

    def test_validation(self):
        x,s,y,_=ts.continuous_gaussian()
        with pytest.raises(Exception,match='one latent value per stimulus'):
            ama.Stim(x,s,np.zeros(len(y),dtype=int),[0.],y=y[:-1])
        st=ama.Stim.binned(x,s,y,nBins=3)
        with pytest.raises(Exception,match='same bins'):
            ama.Stim.binned(x,s[:,y<0],y[y<0],like=st)
        disc=ama.Stim(*ts.gaussian_ctg()[:4])
        with pytest.raises(Exception,match='bWithin'):
            unit_for(disc,bWithin=True)


class TestErrors:
    def test_levels_as_values_change_nothing(self):
        # stimuli whose own values are their levels give the costs of stimuli without values
        x,s,ci,Y,_=ts.unequal_counts()
        idx=np.unique(ci,return_inverse=True)[1]
        for obj in (ama.Objective('l2'),ama.Objective('l1'),ama.Objective('xent',targetSigma=0.7),
                    ama.Objective('wasserstein',targetSigma=0.7)):
            a=unit_for(ama.Stim(x,s,ci,Y),objective=obj)
            b=unit_for(ama.Stim(x,s,ci,Y,y=Y[idx]),objective=obj.copy())
            assert np.isclose(float(a.loss),float(b.loss),rtol=1e-12)

    def test_l2_from_own_values(self):
        st,_,_=binned()
        unit=unit_for(st,errType='l2')
        lpost=np.asarray(unit.posterior)
        est=np.exp(lpost)@np.asarray(st.Y)
        w,yc=valid(unit)
        assert np.isclose(float(unit.loss),np.mean((est[w]-yc[w])**2),rtol=1e-10)

    @pytest.mark.parametrize('period',[None,2.])
    def test_gaussian_target_at_own_value(self,period):
        st,_,_=binned(period=period)
        sig=0.3
        unit=unit_for(st,objective=ama.Objective('xent',targetSigma=sig))
        lpost=np.asarray(unit.posterior)
        w,yc=valid(unit)
        Y=np.asarray(st.Y)
        d=Y[None,None,:]-yc[...,None]
        if period is not None:
            d=d-period*np.floor(d/period+0.5)
        lq=-d**2/(2*sig**2)
        q=np.exp(lq-logsumexp(lq,axis=-1,keepdims=True))
        ref=-np.sum(q*lpost,axis=-1)
        assert np.isclose(float(unit.loss),ref[w].mean(),rtol=1e-10)

    def test_performance_from_own_values(self):
        st,_,_=binned()
        unit=unit_for(st)
        p=unit.performance('mean')
        w,yc=valid(unit)
        est=np.asarray(p['estimates'])
        assert np.isclose(p['rmseAll'],np.sqrt(np.mean((est[w]-yc[w])**2)))
        c=2
        assert np.isclose(p['bias'][c],np.mean(est[w[:,c],c]-yc[w[:,c],c]))


class TestWithin:
    @pytest.mark.parametrize('modelType',['gss','student','mix'])
    def test_linear_within_category(self,modelType):
        st,_,_=binned()
        kw={'nMix':2} if modelType=='mix' else {}
        unit=unit_for(st,modelType,n=2,bWithin=True,**kw)
        R,Rm,RVar=flat_responses(unit)
        w,yc=valid(unit)
        Y=np.asarray(st.Y)
        noiseCov=np.asarray(unit.nrn._corr_fun(jnp.asarray(RVar),unit.stim.weights,unit.nrn.rho))
        _,Yc=unit._lik_parts(unit._likelihoods(unit._nrn_out(),unit.stim.weights,unit.stim.Y,yRef=unit.stim.yCtg))
        Yc=np.asarray(Yc)
        for i in range(st.nCtg):
            Ri=Rm[:,w[:,i],i]
            yi=yc[w[:,i],i]
            X=np.cov(Ri)
            c=np.array([np.cov(Ri[f],yi)[0,1] for f in range(Ri.shape[0])])
            beta=np.linalg.solve(X+noiseCov[i],c)
            for (l,k) in [(0,0),(3,1),(5,i)]:
                ref=yi.mean()+beta@(R[:,l,k]-Ri.mean(1))
                assert np.isclose(Yc[l,k,i],ref,rtol=1e-8)

    @pytest.mark.parametrize('bLeaveOneOut',[False,True])
    def test_full_kernel_regression(self,bLeaveOneOut):
        st,_,_=binned(nStim=80)
        unit=unit_for(st,'full',n=2,bWithin=True,bLeaveOneOut=bLeaveOneOut)
        R,Rm,RVar=flat_responses(unit)
        w,yc=valid(unit)
        lAll,Yc=[np.asarray(a) for a in unit._lik_parts(unit._likelihoods(unit._nrn_out(),unit.stim.weights,unit.stim.Y,
                                                                         yRef=unit.stim.yCtg))]
        for (l,k,i) in [(0,0,0),(3,1,1),(7,2,0),(2,4,3)]:
            js=[j for j in np.flatnonzero(w[:,i]) if not (bLeaveOneOut and i==k and j==l)]
            t=np.array([-0.5*np.sum((R[:,l,k]-Rm[:,j,i])**2/RVar[:,j,i])-0.5*np.sum(np.log(2*np.pi*RVar[:,j,i])) for j in js])
            p=np.exp(t-logsumexp(t))
            assert np.isclose(Yc[l,k,i],p@yc[js,i],rtol=1e-8)
            assert np.isclose(lAll[l,k,i],logsumexp(t)-np.log(w[:,i].sum()),rtol=1e-10)

    def test_circular_within(self):
        P=2.
        st,_,_=binned(period=P,nBins=4)
        unit=unit_for(st,n=2,bWithin=True)
        R,Rm,RVar=flat_responses(unit)
        w,yc=valid(unit)
        Y=np.asarray(st.Y)
        noiseCov=np.asarray(unit.nrn._corr_fun(jnp.asarray(RVar),unit.stim.weights,unit.nrn.rho))
        _,Yc=unit._lik_parts(unit._likelihoods(unit._nrn_out(),unit.stim.weights,unit.stim.Y,yRef=unit.stim.yCtg))
        i=1
        d=yc[w[:,i],i]-Y[i]
        d=d-P*np.floor(d/P+0.5)                                           # offsets from the level, wrapped
        Ri=Rm[:,w[:,i],i]
        c=np.array([np.cov(Ri[f],d)[0,1] for f in range(2)])
        ref=Y[i]+d.mean()+np.linalg.solve(np.cov(Ri)+noiseCov[i],c)@(R[:,0,2]-Ri.mean(1))
        assert np.isclose(np.asarray(Yc)[0,2,i],ref,rtol=1e-8)
        # the circular mean estimate combines them on the circle
        lpost=np.asarray(unit.posterior)
        est=np.asarray(ama.Objective._est__mean(jnp.asarray(lpost),st.Y,st.Yperiod,Yc))
        p=np.exp(lpost[0,2])
        ang=np.angle(np.sum(p*np.exp(2j*np.pi*np.asarray(Yc)[0,2]/P)))*P/(2*np.pi)
        assert np.isclose(est[0,2],ang)

    def test_estimators_with_within_values(self):
        rng=np.random.default_rng(0)
        lpost=jnp.asarray(np.log(rng.dirichlet(np.ones(5),(3,4))))
        Y=jnp.arange(5.)
        Yc=jnp.asarray(Y[None,None]+rng.uniform(-0.4,0.4,(3,4,5)))
        p=np.exp(np.asarray(lpost))
        assert np.allclose(ama.Objective._est__mean(lpost,Y,None,Yc),np.sum(p*np.asarray(Yc),-1))
        mode=np.take_along_axis(np.asarray(Yc),np.argmax(p,-1)[...,None],-1)[...,0]
        assert np.allclose(ama.Objective._est__mode(lpost,Y,None,Yc),mode)
        med=np.asarray(ama.Objective._est__median(lpost,Y,None,Yc))
        for a in range(3):
            for b in range(4):
                o=np.argsort(np.asarray(Yc)[a,b])
                assert np.isclose(med[a,b],np.interp(0.5,np.cumsum(p[a,b][o])-p[a,b][o]/2,np.asarray(Yc)[a,b][o]))

    def test_heldout_estimates_match_evaluate(self):
        st,_,_=binned(nStim=200)
        train,test=st.train_test(0.3)
        unit=unit_for(train,n=2,bWithin=True)
        est=unit.estimates('mean',stim=test)
        w=np.asarray(test.weights)>0
        yc=np.asarray(test.yCtg)
        assert np.isclose(unit.evaluate(test),np.mean((est[w]-yc[w])**2),rtol=1e-8)

    def test_within_estimates_resolve_below_the_bins(self):
        # low noise and few bins: the category levels alone are off by the bin width, within-bin regression is not
        x,s,y,_=ts.continuous_gaussian(nStim=400,signal=0.9)
        st=ama.Stim.binned(x,s,y,nBins=3)
        rmse={}
        for bW in (False,True):
            unit=ama.Unit(st,ama.Nrn(fano=0.05,var0=0.01),ama.Model('gss','mean',bWithin=bW),ama.Objective('l2'),
                          ama.Optimizer(nIterMax=150,lRate0=0.05,bVerbose=False))
            unit.train_new(1)
            rmse[bW]=unit.performance('mean')['rmseAll']
        assert rmse[True]<0.5*rmse[False]

    def test_training_gradient_is_finite(self):
        st,_,_=binned()
        for mt in ('gss','full'):
            unit=ama.Unit(st,ama.Nrn(),ama.Model(mt,'mean',bWithin=True),ama.Objective('l2'),
                          ama.Optimizer(nIterMax=40,lRate0=0.05,bVerbose=False))
            unit.train_new(2)
            h=np.asarray(unit.optimizer.loss_hist)
            assert np.all(np.isfinite(h)) and h[-1]<h[0]


class TestWithinLeaveOneOut:
    @pytest.mark.parametrize('modelType,kw',[('gss',{}),('student',{}),('mix',{'nMix':2}),('gss',{'bLooNoise':False})])
    def test_own_category_without_the_stimulus(self,modelType,kw):
        st,_,_=binned()
        unit=unit_for(st,modelType,n=2,bWithin=True,bLeaveOneOut=True,**kw)
        R,Rm,RVar=flat_responses(unit)
        w,yc=valid(unit)
        noiseCov=np.asarray(unit.nrn._corr_fun(jnp.asarray(RVar),unit.stim.weights,unit.nrn.rho))
        _,Yc=unit._lik_parts(unit._likelihoods(unit._nrn_out(),unit.stim.weights,unit.stim.Y,yRef=unit.stim.yCtg))
        Yc=np.asarray(Yc)
        for (l,k) in [(0,0),(3,1),(5,2),(1,4)]:
            keep=w[:,k].copy()
            keep[l]=False
            Ri=Rm[:,keep,k]
            yi=yc[keep,k]
            N=np.diag(RVar[:,keep,k].mean(1)) if kw.get('bLooNoise',True) else noiseCov[k]
            c=np.array([np.cov(Ri[f],yi)[0,1] for f in range(Ri.shape[0])])
            ref=yi.mean()+np.linalg.solve(np.cov(Ri)+N,c)@(R[:,l,k]-Ri.mean(1))
            assert np.isclose(Yc[l,k,k],ref,rtol=1e-8)
            i=(k+1)%st.nCtg                                                # other categories keep all their stimuli
            Ri=Rm[:,w[:,i],i]
            yi=yc[w[:,i],i]
            c=np.array([np.cov(Ri[f],yi)[0,1] for f in range(Ri.shape[0])])
            ref=yi.mean()+np.linalg.solve(np.cov(Ri)+noiseCov[i],c)@(R[:,l,k]-Ri.mean(1))
            assert np.isclose(Yc[l,k,i],ref,rtol=1e-8)

    def test_two_dimensional_latent(self):
        rng=np.random.default_rng(4)
        x,s,ci,Y,_=ts.gaussian_ctg(nCtg=4,nStimPerCtg=30)
        y=np.stack([rng.standard_normal(s.shape[1]),rng.standard_normal(s.shape[1])],1)+np.asarray(ci)[:,None]
        st=ama.Stim(x,s,ci,np.stack([np.arange(1.,5),np.arange(1.,5)],1),y=y)
        unit=unit_for(st,n=2,bWithin=True,bLeaveOneOut=True)
        R,Rm,RVar=flat_responses(unit)
        w,yc=valid(unit)
        _,Yc=unit._lik_parts(unit._likelihoods(unit._nrn_out(),unit.stim.weights,unit.stim.Y,yRef=unit.stim.yCtg))
        l,k=2,1
        keep=w[:,k].copy(); keep[l]=False
        Ri,yi=Rm[:,keep,k],yc[keep,k]
        N=np.diag(RVar[:,keep,k].mean(1))
        C=np.stack([[np.cov(Ri[f],yi[:,dd])[0,1] for dd in range(2)] for f in range(Ri.shape[0])])
        ref=yi.mean(0)+(R[:,l,k]-Ri.mean(1))@np.linalg.solve(np.cov(Ri)+N,C)
        assert np.allclose(np.asarray(Yc)[l,k,k],ref,rtol=1e-8)


@pytest.mark.parametrize('nLast,loo',[(1,False),(2,True)])
def test_within_rejects_categories_too_small_for_its_regression(nLast,loo):
    x,s,y,_=ts.continuous_gaussian(nStim=200)
    ys=np.sort(y)
    edges=np.r_[np.quantile(y,[0,.25,.5,.75]),(ys[-nLast-1]+ys[-nLast])/2,ys[-1]+1]
    st=ama.Stim.binned(x,s,y,edges=edges)
    u=ama.Unit(st,ama.Nrn(),ama.Model('gss','mean',ctgPoolWidth=0.5,bLeaveOneOut=loo,bWithin=True),ama.Objective('l2'),
               ama.Optimizer(nIterMax=1,bVerbose=False))
    with pytest.raises(Exception,match='bWithin needs at least'):
        u._finalize(2,np.arange(2),dtype=jnp.float64)
