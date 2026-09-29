"""
Divergence costs between the posterior over levels and a (one-hot or gaussian) target: cross-entropy, Jensen-Shannon,
1-Wasserstein and Fisher. Each is checked against an independent computation, and through gradients and training.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

import ama
import stimuli as ts

# unsorted and unevenly spaced levels for gaussian_ctg's 5 categories
Y_UNSORTED=np.array([0.3,-1.,2.5,0.,1.])
CASES=[(err,sig) for err in ('xent','js','wasserstein') for sig in (None,0.7)] + [('fisher',0.7),('fisher',2.)]


def div_unit(errType,targetSigma=None,modelType='gss',n=2,seed=1,Y=Y_UNSORTED,**opt_kw):
    x,s,ci,Y0,_=ts.gaussian_ctg()
    nrn=ama.Nrn(rho=0.) if modelType=='full' else ama.Nrn()
    unit=ama.Unit(ama.Stim(x,s,ci,Y0 if Y is None else Y),nrn,ama.Model(modelType,'mean'),
                  ama.Objective(errType,targetSigma=targetSigma),ama.Optimizer(**{'nIterMax':1,'bVerbose':False,**opt_kw}))
    unit._finalize(n,np.arange(n),dtype=jnp.float64)
    f=np.random.default_rng(seed).standard_normal(unit.filter._shape)
    unit.filter.out=jnp.reshape(jnp.asarray(f/np.linalg.norm(f,axis=0)),unit.filter._shape_exp)
    return unit


def target(Y,k,sigma):
    if sigma is None:
        return (np.arange(len(Y))==k).astype(float)
    q=np.exp(-(Y-Y[k])**2/(2*sigma**2))
    return q/q.sum()


def reference_error(errType,p,q,Y):
    """one stimulus: posterior p and target q over levels Y"""
    if errType=='xent':
        return -np.sum(q[q>0]*np.log(p[q>0]))
    if errType=='js':
        return jensenshannon(p,q)**2                                        # natural log
    if errType=='wasserstein':
        return wasserstein_distance(Y,Y,p,q)
    # fisher: scores by finite differences between neighbouring levels
    o=np.argsort(Y)
    Ys,lp,lq=Y[o],np.log(p[o]),np.log(q[o])
    dY=np.diff(Ys)
    w=(q[o][1:]+q[o][:-1])/2
    return np.sum(w*(np.diff(lp)/dY-np.diff(lq)/dY)**2)/np.sum(w)


def loss_of(unit,f):
    return unit._loss_fun_lrn({'f':f},unit.rng,unit.filter.prepped_jx,unit.filter._insert_index_jx,
                              unit.stim.val,unit.stim.weights,unit.stim.yCtg,unit.stim.Y)


class TestDivergences:
    @pytest.mark.parametrize('modelType',['gss','full'])
    @pytest.mark.parametrize('errType,sigma',CASES)
    def test_matches_reference(self,errType,sigma,modelType):
        unit=div_unit(errType,sigma,modelType)
        P=np.exp(np.asarray(unit.posterior))                                # [ nStim_Ctg x nCtg x nCtg ]
        err=np.asarray(unit.error)
        w=np.asarray(unit.stim.weights)
        Y=np.asarray(unit.stim.Y)
        ref=np.zeros_like(err)
        for j,k in zip(*np.nonzero(w>0)):
            ref[j,k]=reference_error(errType,P[j,k],target(Y,k,sigma),Y)
        assert np.allclose(err[w>0],ref[w>0],rtol=1e-8,atol=1e-10)
        assert np.isclose(float(unit.loss),np.sum(ref*w)/np.sum(w),rtol=1e-8)

    def test_one_hot_cross_entropy_is_map(self):
        assert np.isclose(float(div_unit('xent').loss),float(div_unit('map').loss),rtol=1e-12)

    def test_one_hot_wasserstein_is_expected_absolute_error(self):
        unit=div_unit('wasserstein')
        P=np.exp(np.asarray(unit.posterior))
        Y=np.asarray(unit.stim.Y)
        ref=np.einsum('jki,ki->jk',P,np.abs(Y[None,:]-Y[:,None]))
        w=np.asarray(unit.stim.weights)>0
        assert np.allclose(np.asarray(unit.error)[w],ref[w])

    def test_js_is_bounded(self):
        err=np.asarray(div_unit('js',seed=4).error)
        assert np.all(err>=-1e-12) and np.all(err<=np.log(2)+1e-12)

    def test_fisher_ignores_posterior_normalization(self):
        lQ=jnp.asarray(np.log(np.stack([target(Y_UNSORTED,k,0.7) for k in range(5)])))
        lp=jnp.asarray(np.random.default_rng(0).standard_normal((3,5,5)))
        Y=jnp.asarray(Y_UNSORTED)
        assert np.allclose(ama.Objective._err__fisher(lp,None,lQ,Y),ama.Objective._err__fisher(lp+3.,None,lQ,Y))

    def test_zero_at_the_target(self):
        Y=jnp.asarray(Y_UNSORTED)
        for errType,sigma in CASES:
            lQ=ama.Objective(errType,targetSigma=sigma).log_target(Y)
            err=np.asarray(getattr(ama.Objective,'_err__'+errType)(lQ[None],None,lQ,Y))
            if errType=='xent':                                             # the target's entropy
                q=np.exp(np.asarray(lQ))
                assert np.allclose(err,-np.sum(np.where(q>0,q*np.log(np.where(q>0,q,1)),0),axis=-1))
            else:
                assert np.allclose(err,0,atol=1e-12)

    @pytest.mark.parametrize('errType,sigma',CASES)
    def test_gradient_matches_finite_difference(self,errType,sigma):
        unit=div_unit(errType,sigma)
        f=jnp.asarray(np.random.default_rng(2).standard_normal(unit.filter._shape))
        g=jax.grad(lambda f: loss_of(unit,f))(f)
        assert np.all(np.isfinite(np.asarray(g)))
        for idx in [(0,0),(9,1)]:
            e=np.zeros(f.shape)
            e[idx]=1e-6
            fd=(loss_of(unit,f+e)-loss_of(unit,f-e))/2e-6
            assert np.isclose(g[idx],fd,rtol=1e-4,atol=1e-7)

    @pytest.mark.parametrize('errType,sigma',CASES)
    def test_training_lowers_cost(self,errType,sigma):
        unit=div_unit(errType,sigma,nIterMax=100,lRate0=0.05)
        unit.train_new(2)
        hist=np.asarray(unit.optimizer.loss_hist)
        assert np.all(np.isfinite(hist)) and hist[-1]<hist[0]

    def test_float32(self):
        x,s,ci,_,_=ts.gaussian_ctg()
        for errType,sigma in CASES:
            unit=ama.Unit(ama.Stim(x,s,ci,Y_UNSORTED),ama.Nrn(),ama.Model('gss','mean'),ama.Objective(errType,targetSigma=sigma),
                          ama.Optimizer(nIterMax=20,bVerbose=False))
            unit.train_new(2,dtype=jnp.float32)
            assert np.all(np.isfinite(np.asarray(unit.optimizer.loss_hist)))


class TestSettings:
    def test_validation(self):
        for kw in [dict(errType='fisher'),                                  # no score for the one-hot target
                   dict(errType='map',targetSigma=1.),dict(errType='l2',targetSigma=1.),
                   dict(errType='js',targetSigma=0.),dict(errType='xent',estType='mean'),
                   dict(errType='wasserstein',bPosterior=False)]:
            with pytest.raises(Exception):
                ama.Objective(**kw)
        assert ama.Objective('js').bPosterior is True

    def test_target_sigma_is_in_the_key(self):
        assert ama.Objective('js')._key()!=ama.Objective('js',targetSigma=1.)._key()
        assert ama.Objective('js',targetSigma=1.)._key()!=ama.Objective('js',targetSigma=2.)._key()
        # a changed target is not served from a stale trace
        a,b=div_unit('xent',0.5),div_unit('xent',2.)
        assert not np.isclose(float(a.loss),float(b.loss))

    def test_copy_and_config(self):
        unit=div_unit('wasserstein',0.7)
        assert unit.objective.copy().targetSigma==0.7
        cfg=unit.config()
        assert cfg['objective']['targetSigma']==0.7
        assert ama.Unit.from_config(cfg,unit.stim).objective._key()==unit.objective._key()
