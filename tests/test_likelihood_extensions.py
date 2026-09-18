"""
Likelihood extensions: covariance shrinkage and pooling over categories, the multivariate student t, and the circular
complex gaussian for quadrature-pair responses. Each is checked against an independent computation.
"""
import numpy as np
import pytest
import jax.numpy as jnp
from scipy.stats import multivariate_normal as smvn, multivariate_t as smvt

import ama
import stimuli as ts


def model_inputs(modelType='gss',gen=ts.gaussian_ctg,n=3,fourierType=0,**model_kw):
    """a finalized unit with random filters, and the inputs its Model receives"""
    x,s,ci,Y,_=gen()
    unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model(modelType,'mean',**model_kw),ama.Objective('map'),
                  ama.Optimizer(nIterMax=1,bVerbose=False))
    dtype=jnp.complex128 if fourierType else jnp.float64
    unit._finalize(n,np.arange(n),fourierType=fourierType,dtype=dtype)
    rng=np.random.default_rng(3)
    f=rng.standard_normal(unit.filter._shape)
    if fourierType:
        f=f+1j*rng.standard_normal(unit.filter._shape)
    f=f/np.linalg.norm(f.reshape(-1,n),axis=0)
    unit.filter.out=jnp.reshape(jnp.asarray(f,dtype=dtype),unit.filter._shape_exp)
    nrn_out=unit.nrn.main(unit.rng,unit.stim.val,unit.filter.out_flat)
    R,Rm,RVar=[ama._flatten_responses(v) for v in unit.model._response_fun(*nrn_out)]
    noiseCov=unit.nrn._corr_fun(RVar,unit.stim.weights,unit.nrn.rho)
    return unit,R,Rm,RVar,noiseCov


def sample_stats(Rm,weights,centered=True):
    """per-category means and (hermitian) covariances by brute force"""
    Rm,w=np.asarray(Rm),np.asarray(weights)>0
    mus,covs,dofs,scat=[],[],[],[]
    for c in range(w.shape[1]):
        X=Rm[:,w[:,c],c]
        mu=X.mean(1)
        D=X-mu[:,None] if centered else X
        S=D@D.conj().T
        dof=X.shape[1]-1 if centered else X.shape[1]
        mus.append(mu); scat.append(S); dofs.append(dof); covs.append(S/dof)
    return np.array(mus).T,np.array(covs),np.array(dofs),np.array(scat)


class TestCategoryStatistics:
    def test_no_options_is_the_sample_covariance(self):
        unit,R,Rm,RVar,noiseCov=model_inputs()
        mu,cov=ama.Model._ctg_stats(Rm,unit.stim.weights,unit.model,unit.stim.Y)
        mu0,cov0,_,_=sample_stats(Rm,unit.stim.weights)
        assert np.allclose(mu,mu0) and np.allclose(cov,cov0)

    def test_shrinkage_toward_diagonal_and_pooled(self):
        unit,R,Rm,RVar,noiseCov=model_inputs(covShrink=0.3)
        _,cov0,dofs,scat=sample_stats(Rm,unit.stim.weights)
        _,cov=ama.Model._ctg_stats(Rm,unit.stim.weights,unit.model,unit.stim.Y)
        diag=np.array([np.diag(np.diag(c)) for c in cov0])
        assert np.allclose(cov,0.7*cov0+0.3*diag)
        m=ama.Model('gss',covShrink=1.,covTarget='pooled')
        _,covp=ama.Model._ctg_stats(Rm,unit.stim.weights,m,unit.stim.Y)
        pooled=scat.sum(0)/dofs.sum()
        assert np.allclose(covp,np.broadcast_to(pooled,covp.shape))

    def test_category_pooling_limits(self):
        unit,R,Rm,RVar,noiseCov=model_inputs()
        _,cov0,dofs,scat=sample_stats(Rm,unit.stim.weights)
        Y=unit.stim.Y
        narrow=ama.Model('gss',ctgPoolWidth=1e-3)
        _,c1=ama.Model._ctg_stats(Rm,unit.stim.weights,narrow,Y)
        assert np.allclose(c1,cov0)                                 # tiny kernel: each category on its own
        wide=ama.Model('gss',ctgPoolWidth=1e6)
        _,c2=ama.Model._ctg_stats(Rm,unit.stim.weights,wide,Y)
        assert np.allclose(c2,np.broadcast_to(scat.sum(0)/dofs.sum(),c2.shape))   # huge kernel: fully pooled

    def test_pooled_means(self):
        unit,R,Rm,RVar,noiseCov=model_inputs()
        wide=ama.Model('gss',ctgPoolWidth=1e6,bPoolMeans=True)
        mu,_=ama.Model._ctg_stats(Rm,unit.stim.weights,wide,unit.stim.Y)
        mu0,_,_,_=sample_stats(Rm,unit.stim.weights)
        counts=np.asarray(unit.stim.weights).sum(0)
        grand=(mu0*counts).sum(1)/counts.sum()
        assert np.allclose(mu,np.broadcast_to(grand[:,None],mu.shape))

    def test_invalid_options_raise(self):
        with pytest.raises(Exception):
            ama.Model('gss',covShrink=1.5)
        with pytest.raises(Exception):
            ama.Model('gss',covTarget='identity')
        with pytest.raises(Exception):
            ama.Model('circ',circMean='other')


class TestStudent:
    def test_lmvt0_matches_scipy(self):
        rng=np.random.default_rng(0)
        A=rng.standard_normal((3,3))
        S=A@A.T+np.eye(3)
        x=rng.standard_normal((5,3))
        for df in (3.,7.5,40.):
            assert np.allclose(np.asarray(ama.lmvt0(jnp.asarray(x),jnp.asarray(S)[None],df)),
                               smvt(np.zeros(3),S,df=df).logpdf(x))

    def test_student_likelihood_matches_reference_and_approaches_gauss(self):
        unit,R,Rm,RVar,noiseCov=model_inputs('student',df=6.)
        lAll=np.asarray(ama.Model._model__student(R,Rm,RVar,noiseCov,None,unit.stim.weights,False,unit.model,unit.stim.Y))
        mu0,cov0,_,_=sample_stats(Rm,unit.stim.weights)
        Rn=np.asarray(R)
        c,k,i=1,2,0
        C=cov0[i]+np.asarray(noiseCov)[i]
        ref=smvt(mu0[:,i],C*4/6,df=6.).logpdf(Rn[:,c,k])
        assert np.isclose(lAll[c,k,i],ref)
        big=ama.Model('student','mean',df=1e7)
        lt=np.asarray(ama.Model._model__student(R,Rm,RVar,noiseCov,None,unit.stim.weights,False,big,unit.stim.Y))
        lg=np.asarray(ama.Model._model__gss(R,Rm,RVar,noiseCov,None,unit.stim.weights,False,ama.Model(),unit.stim.Y))
        assert np.allclose(lt,lg,atol=1e-4)

    def test_df_must_exceed_two(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('student',df=2.),ama.Objective('map'),ama.Optimizer(nIterMax=1,bVerbose=False))
        with pytest.raises(Exception,match='df > 2'):
            unit.train_new(1)


class TestCirc:
    def test_lcn0_matches_real_representation(self):
        rng=np.random.default_rng(1)
        n=3
        A=rng.standard_normal((n,n))+1j*rng.standard_normal((n,n))
        C=A@A.conj().T+np.eye(n)
        z=rng.standard_normal((4,n))+1j*rng.standard_normal((4,n))
        # a proper complex gaussian CN(0,C) is the real gaussian N(0, 1/2 [[Re C, -Im C],[Im C, Re C]]) on (Re z, Im z)
        Creal=0.5*np.block([[C.real,-C.imag],[C.imag,C.real]])
        ref=smvn(np.zeros(2*n),Creal).logpdf(np.concatenate((z.real,z.imag),axis=1))
        assert np.allclose(np.asarray(ama.lcn0(jnp.asarray(z),jnp.asarray(C)[None])),ref)

    @pytest.mark.parametrize('circMean',['estimate','zero'])
    def test_circ_likelihood_matches_reference(self,circMean):
        unit,R,Rm,RVar,noiseCov=model_inputs('circ',gen=ts.sine_frequency,n=2,fourierType=2,circMean=circMean)
        lAll=np.asarray(ama.Model._model__circ(R,Rm,RVar,noiseCov,None,unit.stim.weights,False,unit.model,unit.stim.Y))
        h=np.asarray(R).shape[0]//2
        Rc=np.asarray(R)[:h]+1j*np.asarray(R)[h:]
        Rmc=np.asarray(Rm)[:h]+1j*np.asarray(Rm)[h:]
        mu0,cov0,_,_=sample_stats(Rmc,unit.stim.weights,centered=circMean=='estimate')
        if circMean=='zero':
            mu0=np.zeros_like(mu0)
        N=np.asarray(noiseCov)
        for (c,k,i) in [(0,0,0),(3,1,2),(5,2,1)]:
            C=cov0[i]+N[i,:h,:h]+N[i,h:,h:]
            z=Rc[:,c,k]-mu0[:,i]
            ref=-np.real(z.conj()@np.linalg.solve(C,z)) - h*np.log(np.pi) - np.log(np.real(np.linalg.det(C)))
            assert np.isclose(lAll[c,k,i],ref)

    def test_circ_requires_quadrature_pairs(self):
        x,s,ci,Y,_=ts.sine_frequency()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('circ'),ama.Objective('map'),ama.Optimizer(nIterMax=1,bVerbose=False))
        with pytest.raises(Exception,match='fourierType=2'):
            unit.train_new(1,fourierType=1)


@pytest.mark.parametrize('model_kw',[dict(modelType='gss',covShrink=0.5),
                                     dict(modelType='student',df=5.),
                                     dict(modelType='circ',circMean='zero'),
                                     dict(modelType='gss',ctgPoolWidth=2.,covTarget='pooled',covShrink=0.2)])
def test_learning_with_extensions(model_kw):
    x,s,ci,Y,_=ts.sine_frequency()
    m=dict(model_kw)
    unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model(m.pop('modelType'),'mean',**m),ama.Objective('map'),
                  ama.Optimizer(nIterMax=120,lRate0=0.05,bVerbose=False))
    unit.train_new(2,fourierType=2)
    hist=unit.optimizer.loss_hist
    assert np.all(np.isfinite(hist)) and hist[-1]<hist[0]
    x2,s2,ci2,Y2,_=ts.sine_frequency(seed=5)
    assert np.isfinite(unit.evaluate(ama.Stim(x2,s2,ci2,Y2)))
