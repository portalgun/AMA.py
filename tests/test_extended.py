"""
Noise, complex gradients, feature combinations, edge cases, precision, rarely used paths, plotting,
determinism, and training against the burgelab reference.
"""
import warnings

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as smvn

import ama
import stimuli as ts
from test_ama import make_unit, loss_of, train, real_filters_and_half_spectra, ama_path


#- helpers

def unit_from(stim,nrn=None,modelType='gss',responseType='mean',errType='map',n=2,f=None,seed=1,**finalize_kw):
    unit=ama.Unit(stim,nrn or ama.Nrn(),ama.Model(modelType,responseType),ama.Objective(errType),ama.Optimizer(nIterMax=1,bVerbose=False))
    finalize_kw.setdefault('dtype',jnp.float64)
    unit._finalize(n,np.arange(n),**finalize_kw)
    if f is None:
        f=np.random.default_rng(seed).standard_normal(unit.filter._shape)
        if jnp.issubdtype(finalize_kw['dtype'],jnp.complexfloating):
            f=f+1j*np.random.default_rng(seed+1).standard_normal(unit.filter._shape)
        f=f/np.linalg.norm(f.reshape(-1,n),axis=0)
    unit.filter.out=jnp.reshape(jnp.asarray(f,dtype=finalize_kw['dtype']),unit.filter._shape_exp)
    return unit


def ungroup(X,weights):
    """[ nDim x nStim_Ctg x nCtg ] grouped by category -> [ nDim x N ] valid stimuli, and their category labels"""
    X=np.asarray(X)
    w=np.asarray(weights)>0
    cols=[X[:,w[:,c],c] for c in range(w.shape[1])]
    lab=np.concatenate([np.full(col.shape[-1],c) for c,col in enumerate(cols)])
    return np.concatenate(cols,axis=-1),lab


def reference_loss(unit,rho=0.):
    """brute-force numpy MAP cost from the unit's own flattened responses (observed R, means Rm, noise variance V)"""
    nrn_out=unit.nrn.main(unit.rng,unit.stim.val,unit.filter.out_flat)
    R,Rm,V=[np.asarray(ama._flatten_responses(x)) for x in unit.model._response_fun(*nrn_out)]
    R,lab=ungroup(R,unit.stim.weights)
    Rm,_=ungroup(Rm,unit.stim.weights)
    V,_=ungroup(V,unit.stim.weights)
    nDim,N=R.shape
    P=rho+(1-rho)*np.eye(nDim)
    L=np.zeros((N,lab.max()+1))
    for i in range(L.shape[1]):
        m=lab==i
        if unit.model.modelType=='gss':
            sd=np.sqrt(V[:,m].mean(1))
            C=np.atleast_2d(np.cov(Rm[:,m]))+np.outer(sd,sd)*P
            L[:,i]=smvn(Rm[:,m].mean(1),C).logpdf(R.T)+np.log(m.sum()/N)
        else:
            ll=np.stack([smvn(Rm[:,j],np.outer(np.sqrt(V[:,j]),np.sqrt(V[:,j]))*P).logpdf(R.T) for j in np.where(m)[0]])
            L[:,i]=logsumexp(ll,0)-np.log(N)
    lp=L-logsumexp(L,1,keepdims=True)
    return -lp[np.arange(N),lab].mean()


def complex_param_shape(unit):
    return np.broadcast_shapes(*[np.shape(i) for i in unit.filter._insert_index_jx])


#- 1. noise

class TestNoise:
    def test_noisy_response_statistics(self):
        unit=unit_from(ama.Stim(*ts.unequal_counts()[:4]),ama.Nrn(bNoise_2=True),responseType='basic')
        f=unit.filter.out_flat
        outs=jax.vmap(lambda k: unit.nrn.main(k,unit.stim.val,f))(jax.random.split(jax.random.key(0),4000))
        R,RNs=np.asarray(outs[2]),np.asarray(outs[3])
        assert np.allclose(R.std(0),0)                       # mean responses do not depend on the noise draw
        R=R[0]
        assert np.allclose(RNs.mean(0),R,atol=0.2)
        assert np.allclose(RNs.var(0),1.36*np.abs(R)+0.23,rtol=0.12)

    def test_noise_before_normalization(self):
        # unnormalized stimuli (||s||=2) with broadband normalization: stage-1 noise is divided by ||s|| too
        x,s,ci,Y,_=ts.unequal_counts()
        with pytest.warns(UserWarning,match='contrast normalized'):
            stim=ama.Stim(x,2*s,ci,Y)
        unit=unit_from(stim,ama.Nrn(bNoise_1=True,normalizeType='broad',eps=0.),responseType='basic')
        f=unit.filter.out_flat
        outs=jax.vmap(lambda k: unit.nrn.main(k,unit.stim.val,f))(jax.random.split(jax.random.key(1),4000))
        r,R,RNs=np.asarray(outs[0][0]),np.asarray(outs[2][0]),np.asarray(outs[3])
        w=np.asarray(unit.stim.weights)>0
        assert np.allclose(R[:,w],r[:,w]/2)
        assert np.allclose(RNs.mean(0)[:,w],R[:,w],atol=0.1)
        assert np.allclose(RNs.var(0)[:,w],(1.36*np.abs(r[:,w])+0.23)/4,rtol=0.12)
        RVar=np.asarray(outs[4][0])
        assert np.allclose(RVar[:,w],(1.36*np.abs(r[:,w])+0.23)/4)          # the likelihood accounts for stage-1 noise

    def test_noise_is_redrawn_each_iteration(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        for bNoise in [True,False]:
            unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(bNoise_2=bNoise),ama.Model('gss','basic'),ama.Objective('map'),
                          ama.Optimizer(optimizerType='sgd',lRate0=0.,nIterMax=5,bVerbose=False))
            unit.train_new(1)
            h=np.array(unit.optimizer.loss_hist)
            assert np.allclose(h,h[0])!=bNoise

    @pytest.mark.parametrize('averageType,factor',[('mean',1/15),('median',np.pi/2/15)])
    def test_sample_averaging_variance(self,averageType,factor):
        R=jnp.full((1,20000,1),2.)
        smp=ama.Nrn._noise__true(R,1.36,0.23,15,jax.random.key(2),0)
        avg=np.asarray(getattr(ama.Nrn,'_average__'+averageType)(smp))
        assert np.isclose(avg.var(),(1.36*2+0.23)*factor,rtol=0.15)

    def test_log_mean_average(self):
        R=jnp.array([[[3.],[-3.]]])
        smp=ama.Nrn._noise__true(R,0.1,0.01,9,jax.random.key(3),0)
        avg=np.asarray(ama.Nrn._average__log_mean(smp))
        assert np.all(np.isfinite(avg))
        assert np.array_equal(np.sign(avg),np.sign(np.asarray(R)))
        assert np.allclose(np.abs(avg),3,rtol=0.2)

    @pytest.mark.parametrize('modelType',['gss','full'])
    @pytest.mark.parametrize('responseType',['mean','basic'])
    @pytest.mark.parametrize('rho',[0.,0.3])
    def test_cost_matches_reference(self,modelType,responseType,rho):
        unit=unit_from(ama.Stim(*ts.unequal_counts()[:4]),ama.Nrn(bNoise_2=responseType=='basic',rho=rho),modelType,responseType)
        assert np.isclose(float(unit.loss),reference_loss(unit,rho),rtol=1e-8)

    @pytest.mark.parametrize('bComplex',[False,True])
    def test_correlated_noise_samples(self,bComplex):
        R=jnp.zeros((3,40000,1),dtype=jnp.complex128 if bComplex else jnp.float64)
        eta=np.asarray(ama.Nrn._noise__true(R,0.,1.,1,jax.random.key(4),0.4))[:,:,0,0]
        dims=np.concatenate((eta.real,eta.imag)) if bComplex else eta
        C=np.corrcoef(dims)
        off=C[~np.eye(len(C),dtype=bool)]
        assert np.allclose(off,0.4,atol=0.03)

    def test_complex_noise_components(self):
        R=jnp.full((1,40000,1),3.-1j)
        eta=np.asarray(ama.Nrn._noise__true(R,1.36,0.23,1,jax.random.key(5),0))[0,:,0,0]-(3-1j)
        assert np.isclose(eta.real.var(),1.36*3+0.23,rtol=0.05)
        assert np.isclose(eta.imag.var(),1.36*1+0.23,rtol=0.05)
        assert abs(np.corrcoef(eta.real,eta.imag)[0,1])<0.03


#- 2. complex (fourier-domain) gradients

class TestComplexGradient:
    @pytest.mark.parametrize('fourierType',[1,2])
    @pytest.mark.parametrize('modelType',['gss','full'])
    def test_matches_finite_differences(self,fourierType,modelType):
        x,s,ci,Y,_=ts.sine_frequency(nPix=16,nStimPerCtg=8)
        unit=unit_from(ama.Stim(x,s,ci,Y),modelType=modelType,fourierType=fourierType,dtype=jnp.complex128)
        rng=np.random.default_rng(0)
        shape=complex_param_shape(unit)
        f=rng.standard_normal(shape)+1j*rng.standard_normal(shape)
        L=lambda f: float(loss_of(unit,jnp.asarray(f)))
        g=np.conj(np.asarray(jax.grad(lambda f: loss_of(unit,f))(jnp.asarray(f))))
        h=1e-6
        for idx in [(0,0),(3,1)]:
            e=np.zeros(shape,dtype=complex)
            e[idx]=h
            dx=(L(f+e)-L(f-e))/(2*h)
            dy=(L(f+1j*e)-L(f-1j*e))/(2*h)
            assert np.isclose(g[idx],dx+1j*dy,rtol=1e-4,atol=1e-7)

    def test_optimizer_steps_along_conjugate_gradient(self):
        x,s,ci,Y,_=ts.sine_frequency(nPix=16,nStimPerCtg=8)
        unit=unit_from(ama.Stim(x,s,ci,Y),fourierType=2,dtype=jnp.complex128)
        unit.optimizer=ama.Optimizer(optimizerType='sgd',lRate0=1e-3,nIterMax=1,projectionType=['l2_ball',1e6],bVerbose=False)
        rng=np.random.default_rng(1)
        shape=complex_param_shape(unit)
        f0=jnp.asarray(rng.standard_normal(shape)+1j*rng.standard_normal(shape))
        params,*_=unit.optimizer.minimize(f0,jax.random.key(0),unit.stim,unit.filter,unit._loss_fun_lrn)
        g=jax.grad(lambda f: loss_of(unit,f))(f0)
        assert np.allclose(params['f'],f0-1e-3*jnp.conj(g))


#- 3. feature combinations

def split_units(g,fourierType=0,modelType='gss'):
    x,s,ci,Y,_=ts.binocular_shift()
    kw=dict(bSplit=True)
    if fourierType:
        kw.update(fourierType=fourierType,dtype=jnp.complex128)
    return unit_from(ama.Stim(x,s,ci,Y,nSplit=2),modelType=modelType,n=g.shape[-1],f=g,**kw)


class TestCombinations:
    @pytest.mark.parametrize('modelType',['gss','full'])
    def test_split_equals_unsplit_subfilters(self,modelType):
        x,s,ci,Y,info=ts.binocular_shift()
        n=info['nPixEye']
        g=np.random.default_rng(0).standard_normal((n,2,2))
        g/=np.linalg.norm(g.reshape(-1,2),axis=0)
        split=split_units(g,modelType=modelType)
        # each (filter, eye) as its own filter that is zero on the other eye
        sub=np.zeros((2*n,4))
        for k in range(2):
            for e in range(2):
                sub[e*n:(e+1)*n,2*k+e]=g[:,e,k]
        unsplit=unit_from(ama.Stim(x,s,ci,Y),modelType=modelType,n=4,f=sub)
        assert np.isclose(float(split.loss),float(unsplit.loss),rtol=1e-10)

    @pytest.mark.parametrize('modelType',['gss','full'])
    def test_split_fourier_equals_split_spatial(self,modelType):
        n=ts.binocular_shift()[4]['nPixEye']
        g,F=real_filters_and_half_spectra((n,),4)
        g=np.stack((g[:,:2],g[:,2:]),axis=1)/np.sqrt(2)          # [ pix x eye x filter ]
        F=np.stack((F[:,:2],F[:,2:]),axis=1)/np.sqrt(2)
        spatial=split_units(g,modelType=modelType)
        fourier=split_units(F,fourierType=1,modelType=modelType)
        assert np.allclose(np.asarray(fourier.responses.R),np.asarray(spatial.responses.R))
        assert np.isclose(float(fourier.loss),float(spatial.loss),rtol=1e-10)

    @pytest.mark.parametrize('modelType,fourierType',[('full',0),('full',2),('gss',2)])
    def test_split_learning(self,modelType,fourierType):
        unit,info=train(ts.binocular_shift,n=2,nIterMax=60,stim_kw={'nSplit':2},
                        train_kw={'bSplit':True,'fourierType':fourierType},modelType=modelType)
        hist=unit.optimizer.loss_hist
        assert np.all(np.isfinite(hist)) and hist[-1]<hist[0]

    def test_broadband_normalization_in_pipeline(self):
        x,s,ci,Y,_=ts.unequal_counts()
        for scale in [1,2]:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                stim=ama.Stim(x,scale*s,ci,Y)
            unit=unit_from(stim,ama.Nrn(normalizeType='broad',eps=0.))
            r,_,R,_,_=unit.nrn.main(unit.rng,unit.stim.val,unit.filter.out_flat)
            w=np.asarray(unit.stim.weights)>0
            assert np.allclose(np.asarray(R)[:,w],np.asarray(r)[:,w]/scale)
            assert np.isfinite(float(unit.loss))

    def test_generic_normalization_in_pipeline(self):
        unit=unit_from(ama.Stim(*ts.unequal_counts()[:4]),ama.Nrn(normalizeType='gen',eps=0.1))
        r,_,R,_,_=[np.asarray(v) for v in unit.nrn.main(unit.rng,unit.stim.val,unit.filter.out_flat)]
        assert np.allclose(R,r/(0.1+np.abs(r).sum(0,keepdims=True)))
        assert np.isfinite(float(unit.loss))

    @pytest.mark.parametrize('fourierType',[1,2])
    def test_narrowband_normalization_is_bounded_by_rmax(self,fourierType):
        x,s,ci,Y,_=ts.sine_frequency(nPix=16)
        unit=unit_from(ama.Stim(x,s,ci,Y),ama.Nrn(normalizeType='narrow',eps=0.),fourierType=fourierType,dtype=jnp.complex128)
        _,_,R,_,_=unit.nrn.main(unit.rng,unit.stim.val,unit.filter.out_flat)
        w=np.asarray(unit.stim.weights)>0
        assert np.all(np.abs(np.asarray(R)[:,w])<=5.7*(1+1e-9))
        assert np.isfinite(float(unit.loss))
        g=jax.grad(lambda f: loss_of(unit,f))(jnp.asarray(np.ones(complex_param_shape(unit),dtype=complex)))
        assert np.all(np.isfinite(np.asarray(g)))

    def test_narrowband_learning(self):
        unit,_=train(ts.sine_frequency,n=2,nIterMax=60,train_kw={'fourierType':1})
        unit.nrn.normalizeType='narrow'
        unit.train_recurse()
        hist=unit.optimizer.loss_hist
        assert np.all(np.isfinite(hist))


#- 4. edge cases

class TestEdgeCases:
    def test_gss_needs_two_stimuli_per_category(self):
        x,s,ci,Y,_=ts.unequal_counts(counts=(1,5,5))
        with pytest.raises(Exception,match='at least 2'):
            unit_from(ama.Stim(x,s,ci,Y))
        unit=unit_from(ama.Stim(x,s,ci,Y),modelType='full')
        assert np.isfinite(float(unit.loss))

    def test_unit_split_is_validated(self):
        unit,_=train(ts.unequal_counts,nIterMax=5)
        assert unit.split(stimInd=np.arange(3)).stim.nStim_Ctg==3
        with pytest.raises(Exception,match='at least 2'):
            unit.split(stimInd=[0])

    def test_full_model_requires_noise(self):
        with pytest.raises(Exception,match='needs response noise'):
            unit_from(ama.Stim(*ts.unequal_counts()[:4]),ama.Nrn(rho=None),modelType='full')

    @pytest.mark.parametrize('rho',[-0.5,1.])
    def test_invalid_noise_correlation_raises(self,rho):
        with pytest.raises(Exception,match='rho must be'):
            unit_from(ama.Stim(*ts.unequal_counts()[:4]),ama.Nrn(rho=rho),n=4)

    def test_more_filters_than_stimuli_per_category(self):
        x,s,ci,Y,_=ts.unequal_counts(nPix=12,counts=(3,3,3))
        unit=unit_from(ama.Stim(x,s,ci,Y),n=6)
        assert np.isfinite(float(unit.loss))
        g=jax.grad(lambda f: loss_of(unit,f))(unit.filter.out_flat)
        assert np.all(np.isfinite(np.asarray(g)))

    @pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64])
    @pytest.mark.parametrize('modelType',['gss','full'])
    def test_separable_data_with_saturated_posterior(self,dtype,modelType):
        nPix,nCtg,nPer=8,4,6
        rng=np.random.default_rng(0)
        base=np.eye(nPix)[:nCtg].T
        s=np.repeat(base,nPer,axis=1)+1e-3*rng.standard_normal((nPix,nCtg*nPer))
        s=ts.contrast_normalize(s)
        ci=np.repeat(np.arange(nCtg),nPer)
        x=ama.filt.X(ndim=1,n=nPix,totS=1)
        f=ts.contrast_normalize(base[:,:2])
        unit=unit_from(ama.Stim(x,s,ci,np.arange(nCtg)),ama.Nrn(rmax=1000.),modelType,dtype=dtype,f=f)
        L=float(unit.loss)
        assert np.isfinite(L) and L>=0
        g=jax.grad(lambda f: loss_of(unit,f))(unit.filter.out_flat)
        assert np.all(np.isfinite(np.asarray(g)))

    def test_median_estimate_does_not_need_sorted_Y(self):
        lp=jnp.asarray(np.log(np.random.default_rng(0).dirichlet(np.ones(4),size=(5,3))))
        Y=jnp.array([3.,0.,2.,1.])
        order=np.argsort(np.asarray(Y))
        assert np.allclose(ama.Objective._est__median(lp,Y),ama.Objective._est__median(lp[...,order],Y[order]))


#- 5. precision

class TestPrecision:
    @pytest.mark.parametrize('modelType',['gss','full'])
    def test_float32_matches_float64_on_reference(self,reference_mat,modelType):
        f=reference_mat['AMA'][0,0]['f']
        out={}
        for dtype in (jnp.float32,jnp.float64):
            st=ama.Stim.load(ama_path('AMAdataDisparity.mat'))
            unit=ama.Unit(st,ama.Nrn(fano=0.5,var0=0.23,rmax=5.7),ama.Model(modelType,'mean'),ama.Objective('map'),ama.Optimizer(nIterMax=1))
            unit._finalize(4,np.arange(4),dtype=dtype,stimInd=np.arange(100) if modelType=='full' else None)
            unit.filter.out=jnp.asarray(f,dtype=dtype)
            L=lambda g: unit._loss_fun({'f':g},unit.rng,unit.stim.val,unit.stim.weights,unit.stim.yCtg,unit.stim.Y)
            v,g=jax.value_and_grad(L)(unit.filter.out_flat)
            assert v.dtype==dtype
            out[dtype]=(float(v),np.asarray(g,dtype=float))
        (v32,g32),(v64,g64)=out[jnp.float32],out[jnp.float64]
        assert np.isclose(v32,v64,rtol=1e-5)
        assert np.linalg.norm(g32-g64)/np.linalg.norm(g64)<1e-3

    @pytest.mark.parametrize('modelType',['gss','full'])
    def test_float32_many_filters_and_categories(self,modelType):
        x,s,ci,Y,_=ts.gaussian_ctg(nPix=32,nCtg=19,nStimPerCtg=30)
        f=np.random.default_rng(0).standard_normal((32,8))
        vals=[float(unit_from(ama.Stim(x,s,ci,Y),ama.Nrn(rmax=20.),modelType,n=8,f=f/np.linalg.norm(f,axis=0),dtype=dt).loss)
              for dt in (jnp.float32,jnp.float64)]
        assert np.isclose(vals[0],vals[1],rtol=1e-4)


#- 7. rarely used paths

class TestRarePaths:
    def test_unit_error(self):
        for errType in ['map','l2']:
            unit,*_=make_unit(errType=errType)
            err=np.asarray(unit.error)
            w=np.asarray(unit.stim.weights)>0
            assert err.shape==w.shape
            assert np.isclose(err[w].mean(),float(unit.loss))

    def test_gen_test(self):
        st=ama.Stim.gen_test()
        assert st.nCtg==2 and st.val.shape==(72,51,2)

    def test_load_in(self):
        unit,*_=make_unit()
        before=float(unit.loss)
        unit.filter.load_in(jnp.asarray(np.asarray(unit.filter.out)),False)
        assert np.isclose(float(unit.loss),before)

    def test_circular_mean_estimator(self):
        Y=jnp.array([-2.5,-1.,0.5,2.,3.])
        p=np.random.default_rng(0).dirichlet(np.ones(5),size=(4,2))
        est=np.asarray(ama.Objective._est__cmean(jnp.log(jnp.asarray(p)),Y))
        assert np.allclose(est,np.angle(p@np.exp(1j*np.asarray(Y))))

    @pytest.mark.parametrize('name,fun',[
        ('abs',np.abs),
        ('swish2',lambda x: x*(1+np.tanh(x))/2),
        ('gauss',lambda x: np.exp(-x**2)),
        ('igauss',lambda x: 1-np.exp(-x**2)),
    ])
    def test_remaining_activations(self,name,fun):
        R=np.linspace(-3,3,7).reshape(1,7,1)
        assert np.allclose(getattr(ama.Nrn,'_activation__'+name)(jnp.asarray(R),False),fun(R))

    @pytest.mark.parametrize('method',['train_new','train_recurse','train_append'])
    def test_train_methods_accept_optimizer(self,method):
        unit,_=train(ts.gaussian_ctg,nIterMax=5)
        opt=ama.Optimizer(nIterMax=7,lRate0=0.05,bVerbose=False)
        getattr(unit,method)(1) if method!='train_recurse' else None
        if method=='train_recurse':
            unit.train_recurse(optimizer=opt)
        else:
            getattr(unit,method)(1,optimizer=opt)
        assert unit.optimizer is opt and len(opt.loss_hist)==7

    def test_missing_optimizer_raises(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model(),ama.Objective())
        with pytest.raises(Exception,match='Optimizer'):
            unit.train_new(1)

    def test_fourier_requires_complex_dtype(self):
        with pytest.raises(Exception,match='complex'):
            unit_from(ama.Stim(*ts.sine_frequency()[:4]),fourierType=1,dtype=jnp.float64)

    def test_objective_argument_errors(self):
        for kw in [dict(errType='mle',bPosterior=True),dict(errType='map',bPosterior=False),dict(errType=3),dict(errType='mle',estType='mean')]:
            with pytest.raises(Exception):
                ama.Objective(**kw)
        assert ama.Objective(2).errType=='l2'

    def test_no_noise_gss(self):
        unit=unit_from(ama.Stim(*ts.unequal_counts()[:4]),ama.Nrn(rho=None))
        assert np.isfinite(float(unit.loss))


#- 6. plotting

@pytest.fixture
def figures():
    plt.close('all')
    yield
    assert len(plt.get_fignums())>0
    plt.close('all')


class TestPlotting:
    @pytest.mark.parametrize('gen,bSplit',[(ts.gaussian_ctg,False),(ts.image_orientation,False),(ts.binocular_shift,True)])
    def test_stim_plots(self,figures,gen,bSplit):
        st=ama.Stim(*gen()[:4],nSplit=2 if bSplit else 0)
        for dtype,bFourier in [(jnp.float64,False),(jnp.complex128,True)]:
            st._finalize(dtype,None,bFourier,bSplit)
            st.plot()
            st.plot(bFourier=True)

    @pytest.mark.parametrize('gen,stim_kw,train_kw',[
        (ts.gaussian_ctg,{},{}),
        (ts.image_orientation,{},{}),
        (ts.sine_frequency,{},{'fourierType':1}),
        (ts.sine_frequency,{},{'fourierType':2}),
        (ts.binocular_shift,{'nSplit':2},{'bSplit':True}),
        (ts.binocular_shift,{'nSplit':2},{'bSplit':True,'fourierType':2}),
    ])
    def test_filter_and_response_plots(self,figures,gen,stim_kw,train_kw):
        unit,_=train(gen,n=2,nIterMax=5,stim_kw=stim_kw,train_kw=train_kw)
        unit.plot_out()
        unit.plot_out(bFourier=not unit.filter.bIsFourier)
        unit.filter.plot_fprepped()
        unit.filter.plot_indices()
        unit.train_append(1)
        unit.plot_last()
        r=unit.responses
        r.plot_marginal()
        r.plot_joint()
        r.plot_tsne(perplexity=5)
        r.plot_tsne(n_components=3,perplexity=5)

    def test_implied_spatial_filters(self):
        g,F=real_filters_and_half_spectra((16,),1)
        for fourierType in [1,2]:
            unit,_=train(ts.gaussian_ctg,nIterMax=1)
            unit=unit_from(unit.stim_full,fourierType=fourierType,dtype=jnp.complex128,n=1,f=F)
            spatial=unit.filter.implied_spatial()
            assert np.allclose(np.real(spatial),g)
            assert np.iscomplexobj(spatial)==(fourierType==2)

    def test_plot_last_without_previous_raises(self):
        unit,_=train(ts.gaussian_ctg,nIterMax=2)
        with pytest.raises(Exception,match='no previous'):
            unit.plot_last()


#- 9. determinism

def test_same_seed_gives_same_filters():
    x,s,ci,Y,_=ts.gaussian_ctg()
    def run(seed):
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(bNoise_2=True),ama.Model('gss','basic'),ama.Objective('map'),
                      ama.Optimizer(nIterMax=20,lRate0=0.05,bVerbose=False),seed=seed)
        unit.train_new(2)
        return np.asarray(unit.out)
    assert np.array_equal(run(None),run(None))
    assert not np.allclose(run(None),run(1))


#- 8. training on the burgelab reference

def test_training_reaches_burgelab_cost(reference_mat):
    """learning 2 AMA-Gauss filters on the disparity set reaches a lower cost than burgelab's own filters"""
    A=reference_mat['AMA'][0,0]
    prm=A['paramRSP'][0,0]
    nrn_kw=dict(fano=prm['fano'].item(),var0=prm['var0'].item(),rmax=prm['rMax'].item())
    st=ama.Stim.load(ama_path('AMAdataDisparity.mat'))

    ref=unit_from(st,ama.Nrn(**nrn_kw),n=2,f=A['f'][:,:2])
    ref_cost=float(ref.loss)

    unit=ama.Unit(st,ama.Nrn(**nrn_kw),ama.Model('gss','mean'),ama.Objective('map'),
                  ama.Optimizer(nIterMax=400,lRate0=0.02,bVerbose=False))
    unit.train_new(2)
    assert float(unit.loss)<ref_cost


def test_adam_refinement_does_not_increase_cost(reference_mat):
    """adam on the unit sphere needs tangent-plane gradients (Burge & Jaini 2017, Eq 19); without them it climbs away from good filters"""
    A=reference_mat['AMA'][0,0]
    prm=A['paramRSP'][0,0]
    st=ama.Stim.load(ama_path('AMAdataDisparity.mat'))
    nrn=ama.Nrn(fano=prm['fano'].item(),var0=prm['var0'].item(),rmax=prm['rMax'].item())
    unit=unit_from(st,nrn,n=2,f=A['f'][:,:2],dtype=jnp.float32)
    unit.optimizer=ama.Optimizer(nIterMax=300,lRate0=0.005,bVerbose=False)
    unit.train_recurse()
    h=np.array(unit.optimizer.loss_hist)
    assert h[-1]<h[0]
    assert h[-1]<=h.min()+1e-3


@pytest.mark.parametrize('bComplex',[False,True])
def test_tangent_projection_removes_radial_component(bComplex):
    rng=np.random.default_rng(0)
    shape=(6,2,3)
    f=rng.standard_normal(shape)
    g=rng.standard_normal(shape)
    if bComplex:
        f=f+1j*rng.standard_normal(shape)
        g=g+1j*rng.standard_normal(shape)
    f=f/np.sqrt((np.abs(f)**2).sum((0,1)))
    t=np.asarray(ama.Optimizer._tangent(jnp.asarray(g),jnp.asarray(f)))
    assert np.allclose(np.real(np.conj(f)*t).sum((0,1)),0)              # orthogonal to each filter
    assert np.allclose(g-t,f*np.real(np.conj(f)*g).sum((0,1)))          # only the radial part was removed
