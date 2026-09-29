"""
Run from the repo root:  python -m pytest tests
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as smvn

import ama
import stimuli as ts


GENERATORS=[ts.gaussian_ctg,ts.sine_frequency,ts.binocular_shift,ts.image_orientation,ts.unequal_counts]


#- helpers

def make_unit(gen=ts.unequal_counts,n=2,modelType='gss',responseType='mean',errType='map',nrn=None,f=None,seed=1,**kw):
    x,s,ci,Y,_=gen()
    unit=ama.Unit(ama.Stim(x,s,ci,Y),nrn or ama.Nrn(),ama.Model(modelType,responseType),ama.Objective(errType),ama.Optimizer(nIterMax=1,**kw))
    unit._finalize(n,np.arange(n),dtype=jnp.float64)
    if f is None:
        f=np.random.default_rng(seed).standard_normal(unit.filter._shape)
        f=f/np.linalg.norm(f,axis=0)
    unit.filter.out=jnp.reshape(jnp.asarray(f),unit.filter._shape_exp)
    return unit,(s,ci,Y),np.asarray(f)


def reference_log_posterior(model,f,s,ci,nY,fano=1.36,var0=0.23,rmax=5.7):
    """brute-force numpy: AMA-Gauss (Jaini & Burge 2017) and full AMA (Burge & Jaini 2017, Eq 5)"""
    lab=np.unique(ci,return_inverse=True)[1]
    N=len(lab)
    r=rmax*f.T@s
    v=fano*np.abs(r)+var0
    L=np.zeros((N,nY))
    for i in range(nY):
        m=lab==i
        if model=='gss':
            C=np.atleast_2d(np.cov(r[:,m]))+np.diag(v[:,m].mean(1))
            L[:,i]=smvn(r[:,m].mean(1),C).logpdf(r.T)+np.log(m.sum()/N)
        else:
            ll=np.stack([-0.5*(((r-r[:,[j]])**2)/v[:,[j]]).sum(0) - 0.5*np.log(2*np.pi*v[:,j]).sum() for j in np.where(m)[0]])
            L[:,i]=logsumexp(ll,0)-np.log(N)
    return L-logsumexp(L,1,keepdims=True),lab


def loss_of(unit,f):
    return unit._loss_fun_lrn({'f':f},unit.rng,unit.filter.prepped_jx,unit.filter._insert_index_jx,
                              unit.stim.val,unit.stim.weights,unit.stim.yCtg,unit.stim.Y)


#- stimuli

@pytest.mark.parametrize('gen',GENERATORS)
def test_stimuli_are_contrast_normalized(gen):
    x,s,ci,Y,_=gen()
    flat=s.reshape(-1,s.shape[-1])
    assert np.allclose(flat.mean(0),0,atol=1e-12)
    assert np.allclose(np.linalg.norm(flat,axis=0),1)
    assert len(np.unique(ci))==len(Y)
    assert s.shape[-1]==len(ci)


def test_binocular_shift_right_eye_is_shifted_left_eye():
    x,s,ci,Y,info=ts.binocular_shift(disparities=(0,),smooth=1.0)
    n=info['nPixEye']
    assert np.allclose(s[:n],s[n:])


#- math primitives

def test_lmvn0_matches_scipy():
    rng=np.random.default_rng(0)
    B=rng.standard_normal((4,4))
    C=B@B.T+np.eye(4)
    x=rng.standard_normal((7,4))
    assert np.allclose(ama.lmvn0(jnp.asarray(x),jnp.asarray(C)),smvn(np.zeros(4),C).logpdf(x))


#- Stim

class TestStim:
    def test_relabels_and_pads_unequal_categories(self):
        x,s,ci,Y,info=ts.unequal_counts()
        st=ama.Stim(x,s,ci,Y)
        assert ci.min()==1
        assert st.nCtg==3
        assert st.nStim_Ctg==info['counts'].max()
        assert np.array_equal(np.asarray(st.weights).sum(0),info['counts'])
        assert np.allclose(np.asarray(st.yCtg)[0],Y)
        # grouped stimuli are the originals
        c=1
        assert np.allclose(np.asarray(st.val)[:,:info['counts'][c],c],s[:,ci==c+1])

    def test_Y_length_mismatch_raises(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        with pytest.raises(Exception):
            ama.Stim(x,s,ci,Y[:-1])

    def test_subset(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        sub=ama.Stim(x,s,ci,Y)[np.arange(5)]
        assert sub.val.shape[-2:]==(5,len(Y))
        assert sub.weights.shape==(5,len(Y))
        assert sub.nStim==5*len(Y)

    def test_split_roundtrip_and_layout(self):
        x,s,ci,Y,info=ts.binocular_shift()
        st=ama.Stim(x,s,ci,Y,nSplit=2)
        val=np.asarray(st.val)
        st.split()
        n=info['nPixEye']
        assert st.val.shape==(n,2,st.nStim_Ctg,st.nCtg)
        assert np.allclose(np.asarray(st.val)[:,0],val[:n])
        assert np.allclose(np.asarray(st.val)[:,1],val[n:])
        st.unsplit()
        assert np.allclose(np.asarray(st.val),val)

    @pytest.mark.parametrize('gen,bSplit',[(ts.gaussian_ctg,False),(ts.image_orientation,False),(ts.binocular_shift,True)])
    def test_fourier_roundtrip(self,gen,bSplit):
        x,s,ci,Y,_=gen()
        st=ama.Stim(x,s,ci,Y,nSplit=2 if bSplit else 0)
        st._finalize(jnp.float64,None,False,bSplit)
        val=np.asarray(st.val)
        st._finalize(jnp.complex128,None,True,bSplit)
        assert jnp.iscomplexobj(st.val)
        st._finalize(jnp.float64,None,False,bSplit)
        assert np.allclose(np.asarray(st.val),val)

    def test_fourier_preserves_dot_products(self):
        # Parseval: responses in the fourier domain are proportional to spatial-domain responses
        x,s,ci,Y,_=ts.gaussian_ctg()
        st=ama.Stim(x,s,ci,Y)
        a=np.asarray(st.val)[:,0,0]
        b=np.asarray(st.val)[:,1,0]
        st._finalize(jnp.complex128,None,True,False)
        A=np.asarray(st.val)[:,0,0]
        B=np.asarray(st.val)[:,1,0]
        assert np.allclose(np.vdot(A,B),a@b)   # orthonormal transform


#- response model

class TestNrn:
    def test_respond_is_scaled_dot_product(self):
        rng=np.random.default_rng(0)
        f=rng.standard_normal((6,3))
        s=rng.standard_normal((6,4,2))
        R=ama.Nrn.respond(jnp.asarray(f),jnp.asarray(s),5.7,False)
        assert np.allclose(R,5.7*np.einsum('pf,pnc->fnc',f,s))

    def test_variance_is_scaled_additive(self):
        R=jnp.array([-2.,0.,3.])
        assert np.allclose(ama.Nrn.variance(R,1.36,0.23),1.36*np.abs(R)+0.23)

    def test_noise_statistics(self):
        R=jnp.array([[-3.,0.,4.]])
        smp=np.asarray(ama.Nrn._noise__true(R,1.36,0.23,200000,jax.random.key(0),False))
        assert np.allclose(smp.mean(-1),R,atol=0.02)
        assert np.allclose(smp.var(-1),1.36*np.abs(R)+0.23,rtol=0.02)

    def test_noise_off_is_identity(self):
        R=jnp.ones((2,3,4))
        assert np.allclose(ama.Nrn._average__full(ama.Nrn._noise__none(R)),R)

    @pytest.mark.parametrize('name,fun',[
        ('relu',lambda x: np.maximum(x,0)),
        ('logistic',lambda x: 1/(1+np.exp(-x))),
        ('swish',lambda x: x/(1+np.exp(-x))),
        ('softplus',lambda x: np.log1p(np.exp(x))),
        ('tanh',np.tanh),
    ])
    def test_activations(self,name,fun):
        R=np.linspace(-3,3,7).reshape(1,7,1)
        out=getattr(ama.Nrn,'_activation__'+name)(jnp.asarray(R),False)
        assert out.shape==R.shape
        assert np.allclose(out,fun(R))

    def test_activation_complex_is_componentwise(self):
        R=jnp.array([-1.+2j,3.-4j])
        assert np.allclose(ama.Nrn._activation__relu(R,True),np.array([0.+2j,3.+0j]))

    def test_broadband_normalization(self):
        rng=np.random.default_rng(0)
        s=rng.standard_normal((5,3,2))
        R=rng.standard_normal((4,3,2))
        out=ama.Nrn._normalize__broad(jnp.asarray(R),None,jnp.asarray(s),0.,False)
        assert np.allclose(out,R/np.linalg.norm(s,axis=0))

    def test_noise_covariance(self):
        RVar=jnp.asarray(np.random.default_rng(0).uniform(1,2,(3,4,2)))
        w=jnp.ones((4,2)).at[3,1].set(0)
        vbar=np.array([np.asarray(RVar)[:,:4,0].mean(1),np.asarray(RVar)[:,:3,1].mean(1)])
        unc=np.asarray(ama.Nrn._corr__uncorr(RVar,w,0))
        assert np.allclose(unc,vbar[:,:,None]*np.eye(3))
        cor=np.asarray(ama.Nrn._corr__corr(RVar,w,0.3))
        sd=np.sqrt(vbar)
        assert np.allclose(np.diagonal(cor,axis1=1,axis2=2),vbar)
        assert np.isclose(cor[1,0,2],0.3*sd[1,0]*sd[1,2])

    def test_rho_selects_corr_type(self):
        assert ama.Nrn(rho=None).corrType=='None'
        assert ama.Nrn(rho=0).corrType=='uncorr'
        assert ama.Nrn(rho=0.2).corrType=='corr'

    def test_narrow_normalization_requires_fourier(self):
        with pytest.raises(Exception):
            make_unit(nrn=ama.Nrn(normalizeType='narrow'))


#- likelihood, posterior, cost

class TestLikelihood:
    @pytest.mark.parametrize('model',['gss','full'])
    def test_map_cost_matches_reference(self,model):
        unit,(s,ci,Y),f=make_unit(modelType=model)
        lp,lab=reference_log_posterior(model,f,s,ci,len(Y))
        assert np.isclose(float(unit.loss),-lp[np.arange(len(lab)),lab].mean(),rtol=1e-10)

    @pytest.mark.parametrize('model',['gss','full'])
    def test_posterior_matches_reference_and_normalizes(self,model):
        unit,(s,ci,Y),f=make_unit(modelType=model)
        lp_ref,lab=reference_log_posterior(model,f,s,ci,len(Y))
        post=np.asarray(unit.posterior)            # [ nStim_Ctg x nCtg x nCtg ]
        w=np.asarray(unit.stim.weights)>0
        assert np.allclose(np.exp(post[w]).sum(-1),1)
        for c in range(len(Y)):
            assert np.allclose(post[:w[:,c].sum(),c],lp_ref[lab==c])

    def test_prior_uses_category_counts(self):
        unit,*_=make_unit()
        lAll=jnp.zeros((unit.stim.nStim_Ctg,unit.stim.nCtg,unit.stim.nCtg))
        post=np.exp(np.asarray(ama.Objective._posterior__true(lAll,unit.stim.weights)))
        counts=np.asarray(unit.stim.weights).sum(0)
        assert np.allclose(post[0,0],counts/counts.sum())

    def test_mle_error_is_negative_log_likelihood(self):
        lAll=jnp.log(jnp.array([[[0.5,0.25],[0.1,0.2]]]))
        assert np.allclose(ama.Objective._err__mle(lAll,None),-np.log([[0.5,0.2]]))

    def test_padding_does_not_change_loss(self):
        err=jnp.array([[1.,2.],[3.,jnp.nan]])
        w=jnp.array([[1.,1.],[1.,0.]])
        assert np.isclose(ama.Objective._loss__mean(err,w),2.)
        assert np.isclose(ama.Objective._loss__median(err,w),2.)

    def test_estimators(self):
        Y=jnp.array([0.,1.,2.,3.])
        p=jnp.array([0.1,0.2,0.3,0.4])
        lp=jnp.log(p)[None]
        assert np.isclose(ama.Objective._est__mean(lp,Y)[0],p@Y)
        assert np.isclose(ama.Objective._est__mode(lp,Y)[0],3.)
        assert np.isclose(ama.Objective._est__median(lp,Y)[0],np.interp(0.5,np.cumsum(p),Y))

    def test_objective_defaults_follow_cost(self):
        assert ama.Objective('l2').estType=='mean'
        assert ama.Objective('l1').estType=='median'
        assert ama.Objective('map').bPosterior is True
        assert ama.Objective('mle').bPosterior is False
        with pytest.raises(Exception):
            ama.Objective('map',estType='mean')

    @pytest.mark.parametrize('model,err',[('gss','map'),('gss','l2'),('full','map'),('full','l2')])
    def test_gradient_matches_finite_difference(self,model,err):
        unit,*_=make_unit(modelType=model,errType=err)
        f=jnp.asarray(np.random.default_rng(2).standard_normal(unit.filter._shape))
        g=jax.grad(lambda f: loss_of(unit,f))(f)
        for idx in [(0,0),(3,1)]:
            e=np.zeros(f.shape)
            e[idx]=1e-6
            fd=(loss_of(unit,f+e)-loss_of(unit,f-e))/2e-6
            assert np.isclose(g[idx],fd,rtol=1e-4,atol=1e-7)


def test_burgelab_reference_cost(reference_mat):
    """AMA-Gauss cost of the burgelab filters on their disparity training set matches their reported cost"""
    A=reference_mat['AMA'][0,0]
    prm=A['paramRSP'][0,0]
    st=ama.Stim.load(ama_path('AMAdataDisparity.mat'))
    for nF,tol in [(1,0.005),(2,0.03)]:
        nrn=ama.Nrn(fano=prm['fano'].item(),var0=prm['var0'].item(),rmax=prm['rMax'].item())
        unit=ama.Unit(st,nrn,ama.Model('gss','mean'),ama.Objective('map'),ama.Optimizer(nIterMax=1))
        unit._finalize(nF,np.arange(nF),dtype=jnp.float64)
        unit.filter.out=jnp.asarray(A['f'][:,:nF])
        assert abs(float(unit.loss)-A['E'].ravel()[nF-1])<tol


def ama_path(name):
    import os
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ama.__file__))),name)


#- filter indexing

class TestIndex:
    def test_overlapping_indices_raise(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        with pytest.raises(Exception):
            ama.Filter()._finalize(ama.Stim(x,s,ci,Y),jnp.float64,2,[0,1],[1])

    def test_out_of_range_raises(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        with pytest.raises(Exception):
            ama.Filter()._finalize(ama.Stim(x,s,ci,Y),jnp.float64,2,[0,2])

    def test_fourier_index_1d_is_positive_frequencies(self):
        x,s,ci,Y,_=ts.sine_frequency(nPix=8)
        st=ama.Stim(x,s,ci,Y)
        st._finalize(jnp.complex128,None,True,False)
        filt=ama.Filter()
        filt._finalize(st,jnp.complex128,1,[0],[],bAnalytic=True)
        assert np.array_equal(filt.index.pix,np.where(ama._centered_freqs(8)>0)[0])

    @pytest.mark.parametrize('size',[(8,8),(7,6)])
    def test_fourier_index_2d_is_half_plane(self,size):
        x,s,ci,Y,_=ts.image_orientation(size=size)
        st=ama.Stim(x,s,ci,Y)
        st._finalize(jnp.complex128,None,True,False)
        filt=ama.Filter()
        filt._finalize(st,jnp.complex128,1,[0],[],bAnalytic=True)
        kept=np.zeros(size,dtype=bool)
        kept.flat[filt.index.pix]=True
        f0,f1=np.meshgrid(ama._centered_freqs(size[0]),ama._centered_freqs(size[1]),indexing='ij')
        for i,j in np.ndindex(*size):
            k=(f0[i,j],f1[i,j])
            if k==(0,0):
                assert not kept[i,j]
                continue
            m=np.isclose(f0,-k[0]) & np.isclose(f1,-k[1])
            if m.any():
                # -k is on the grid (not aliased through nyquist): exactly one of the conjugate pair is learned
                assert kept[i,j] != kept[m][0]
        # both diagonal orientations are representable
        assert kept[(f0>0)&(f1>0)].any() and kept[(f0>0)&(f1<0)].any()


#- learning

def train(gen,n=1,nIterMax=150,stim_kw={},train_kw={},**unit_kw):
    x,s,ci,Y,info=gen()
    kw=dict(modelType='gss',responseType='mean',errType='map')
    kw.update(unit_kw)
    unit=ama.Unit(ama.Stim(x,s,ci,Y,**stim_kw),ama.Nrn(),ama.Model(kw['modelType'],kw['responseType']),ama.Objective(kw['errType']),
                  ama.Optimizer(nIterMax=nIterMax,lRate0=0.05,bVerbose=False))
    unit.train_new(n,**train_kw)
    return unit,info


class TestLearning:
    def test_recovers_informative_direction(self):
        unit,info=train(ts.gaussian_ctg,nIterMax=300)
        f=np.asarray(unit.out)[:,0]
        assert np.isclose(np.linalg.norm(f),1,atol=1e-5)
        assert abs(f@info['direction'])>0.95
        hist=unit.optimizer.loss_hist
        assert hist[-1]<hist[0]

    def test_append_keeps_fixed_filters_and_recurse_runs(self):
        unit,_=train(ts.gaussian_ctg,nIterMax=50)
        f1=np.asarray(unit.out)[:,0].copy()
        unit.train_append(1)
        assert np.shape(unit.out)==(16,2)
        assert np.allclose(np.asarray(unit.out)[:,0],f1)
        unit.train_recurse()
        assert np.all(np.isfinite(np.asarray(unit.out)))
        assert np.allclose(np.linalg.norm(np.asarray(unit.out),axis=0),1,atol=1e-5)

    def test_recurse_subset_keeps_others_fixed(self):
        unit,_=train(ts.gaussian_ctg,n=2,nIterMax=30)
        f0=np.asarray(unit.out)[:,0].copy()
        unit.train_recurse(ind_rec=[1])
        assert np.allclose(np.asarray(unit.out)[:,0],f0)

    @pytest.mark.parametrize('fourierType',[1,2])
    def test_fourier_learning(self,fourierType):
        unit,_=train(ts.sine_frequency,n=2,train_kw={'fourierType':fourierType})
        hist=unit.optimizer.loss_hist
        assert np.isfinite(hist[-1]) and hist[-1]<hist[0]
        assert jnp.iscomplexobj(unit.out)

    def test_split_learning(self):
        unit,info=train(ts.binocular_shift,n=2,stim_kw={'nSplit':2},train_kw={'bSplit':True})
        assert np.shape(unit.out)==(info['nPixEye'],2,2)
        hist=unit.optimizer.loss_hist
        assert hist[-1]<hist[0]

    @pytest.mark.parametrize('model,response,err',[('full','basic','l2'),('gss','basic','l1'),('gss','mean','mle')])
    def test_other_configurations_on_2d_images(self,model,response,err):
        unit,_=train(ts.image_orientation,n=2,nIterMax=40,modelType=model,responseType=response,errType=err)
        assert np.shape(unit.out)==(8,8,2)
        assert np.isfinite(unit.optimizer.loss_hist[-1])

    def test_unit_split_and_copies(self):
        unit,_=train(ts.gaussian_ctg,nIterMax=20)
        sub=unit.split(stimInd=np.arange(10))
        assert sub.stim.nStim_Ctg==10
        assert np.isfinite(float(sub.loss))
        assert np.allclose(np.asarray(sub.out),np.asarray(unit.out))
        r=unit.responses
        assert r.R.shape==(1,unit.stim.nStim_Ctg,unit.stim.nCtg)


#- mini-batches (AMA-SGD) and compiled-trace reuse

class TestBatch:
    def test_batch_plan_preserves_category_proportions(self):
        x,s,ci,Y,info=ts.unequal_counts(counts=(10,30,20))
        st=ama.Stim(x,s,ci,Y)
        mMax,mask=ama.Optimizer(batchSize=30)._batch_plan(st.weights)
        assert mMax==15
        assert np.array_equal(np.asarray(mask).sum(0),[5,15,10])

    def test_batch_plan_keeps_two_per_category(self):
        x,s,ci,Y,_=ts.unequal_counts(counts=(10,30,20))
        _,mask=ama.Optimizer(batchSize=3)._batch_plan(ama.Stim(x,s,ci,Y).weights)
        assert np.all(np.asarray(mask).sum(0)>=2)

    def test_sample_batch_draws_distinct_valid_stimuli(self):
        x,s,ci,Y,info=ts.unequal_counts(counts=(10,30,20))
        st=ama.Stim(x,s,ci,Y)
        mMax,mask=ama.Optimizer(batchSize=30)._batch_plan(st.weights)
        val,w,y=ama.Optimizer._sample_batch(jax.random.key(3),mMax,mask,st.val,st.weights,st.yCtg)
        val,w,y=np.asarray(val),np.asarray(w),np.asarray(y)
        full=np.asarray(st.val)
        for c,n in enumerate([5,15,10]):
            assert w[:,c].sum()==n
            cols=[np.where(np.all(np.isclose(full[:,:info['counts'][c],c],val[:,[b],c]),axis=0))[0] for b in range(n)]
            assert all(len(i)==1 for i in cols)                   # each drawn stimulus is a real stimulus of this category
            assert len({int(i[0]) for i in cols})==n              # without replacement
            assert np.allclose(y[:n,c],Y[c])

    def test_batch_learning_recovers_informative_direction(self):
        x,s,ci,Y,info=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('gss','mean'),ama.Objective('map'),
                      ama.Optimizer(nIterMax=400,lRate0=0.05,batchSize=60,bVerbose=False))
        unit.train_new(1)
        f=np.asarray(unit.out)[:,0]
        assert abs(f@info['direction'])>0.95

    def test_full_model_batch_learning(self):
        x,s,ci,Y,_=ts.image_orientation()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('full','mean'),ama.Objective('map'),
                      ama.Optimizer(nIterMax=60,lRate0=0.05,batchSize=40,bVerbose=False))
        unit.train_new(2)
        assert np.all(np.isfinite(unit.optimizer.loss_hist))
        assert np.all(np.isfinite(np.asarray(unit.out)))

    def test_chunking_does_not_change_result(self):
        outs=[]
        for chunk in [130,50]:
            unit,_=train(ts.gaussian_ctg,nIterMax=130)
            unit.optimizer.nStepsPerChunk=chunk
            unit.train_new(1)
            assert len(unit.optimizer.loss_hist)==130
            outs.append(np.asarray(unit.out))
        assert np.allclose(outs[0],outs[1],atol=1e-6)

    def test_changed_configuration_is_not_stale(self):
        unit,(s,ci,Y),f=make_unit()
        before=float(unit.loss)
        unit.nrn.fano=0.5
        after=float(unit.loss)
        fresh,*_=make_unit(nrn=ama.Nrn(fano=0.5),f=f)
        assert not np.isclose(before,after)
        assert np.isclose(after,float(fresh.loss))

    def test_retraining_reuses_compiled_chunk(self):
        unit,_=train(ts.gaussian_ctg,nIterMax=20)
        n=ama.Optimizer._run_chunk._cache_size()
        unit.train_recurse()
        unit.train_recurse()
        assert ama.Optimizer._run_chunk._cache_size()==n


#- fourier-domain filters

def centered_ft(v,ndim,inverse=False):
    axes=tuple(range(ndim))
    fun=np.fft.ifftn if inverse else np.fft.fftn
    return np.fft.fftshift(fun(np.fft.ifftshift(v,axes=axes),axes=axes,norm='ortho'),axes=axes)


def real_filters_and_half_spectra(dims,n,seed=0):
    """
    unit-norm real spatial filters g [ *dims x n ] (no DC, no nyquist-only content) and the half spectra
    F = sqrt(2) G on the learned half-space, which fourierType=1 must turn back into exactly g
    """
    ndim=len(dims)
    grids=np.meshgrid(*[ama._centered_freqs(d) for d in dims],indexing='ij')
    keep=np.zeros(dims,dtype=bool)
    undecided=np.ones(dims,dtype=bool)
    for g in grids:
        keep|=undecided & (g>0)
        undecided&=(g==0)
    rng=np.random.default_rng(seed)
    g=rng.standard_normal(tuple(dims)+(n,))
    G=centered_ft(g,ndim)
    conj_of_kept=np.zeros(dims,dtype=bool)
    for idx in zip(*np.where(keep)):
        k=[grids[a][idx] for a in range(ndim)]
        m=np.ones(dims,dtype=bool)
        for a in range(ndim):
            m&=np.isclose(grids[a],-k[a])
        conj_of_kept|=m
    G[~(keep|conj_of_kept)]=0
    g=np.real(centered_ft(G,ndim,inverse=True))
    g/=np.linalg.norm(g.reshape(-1,n),axis=0)
    G=centered_ft(g,ndim)
    F=np.where(keep[...,None],np.sqrt(2)*G,0)
    return g,F


def fourier_unit(gen,fourierType,F,modelType='gss'):
    x,s,ci,Y,_=gen()
    unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model(modelType,'mean'),ama.Objective('map'),ama.Optimizer(nIterMax=1))
    unit._finalize(F.shape[-1],np.arange(F.shape[-1]),fourierType=fourierType,dtype=jnp.complex128)
    unit.filter.out=jnp.asarray(F.reshape(unit.filter._shape_exp))
    return unit


class TestFourierFilters:
    @pytest.mark.parametrize('gen,dims',[(ts.gaussian_ctg,(16,)),(ts.image_orientation,(8,8))])
    @pytest.mark.parametrize('modelType',['gss','full'])
    def test_fourierType1_equals_spatial_filter(self,gen,dims,modelType):
        g,F=real_filters_and_half_spectra(dims,2)
        assert np.isclose(np.linalg.norm(F[...,0]),1)
        spatial,*_=make_unit(gen=gen,modelType=modelType,f=g.reshape(-1,2))
        fourier=fourier_unit(gen,1,F,modelType)
        assert np.allclose(np.asarray(fourier.responses.R),np.asarray(spatial.responses.R))
        assert np.isclose(float(fourier.loss),float(spatial.loss),rtol=1e-10)

    def test_fourierType2_is_unit_norm_hilbert_pair_1d(self):
        from scipy.signal import hilbert
        g,F=real_filters_and_half_spectra((16,),1)
        unit=fourier_unit(ts.gaussian_ctg,2,F)
        R=np.asarray(unit.responses.R)[0]
        x,s,ci,Y,_=ts.gaussian_ctg()
        sp=np.asarray(ama.Stim(x,s,ci,Y).val)
        h=np.imag(hilbert(g[:,0]))
        assert np.isclose(np.linalg.norm(h),1) and np.isclose(g[:,0]@h,0)
        assert np.allclose(R.real,5.7*np.einsum('p,pnc->nc',g[:,0],sp))
        assert np.allclose(R.imag,-5.7*np.einsum('p,pnc->nc',h,sp))

    def test_fourierType2_pair_2d_unit_norm_and_orthogonal(self):
        g,F=real_filters_and_half_spectra((8,8),1)
        gc=centered_ft(np.sqrt(2)*F[...,0],2,inverse=True)   # response = <gc, s>
        assert np.allclose(gc.real,g[...,0])
        assert np.isclose(np.linalg.norm(gc.imag),1)
        assert np.isclose(np.sum(gc.real*gc.imag),0)
        unit=fourier_unit(ts.image_orientation,2,F)
        assert np.all(np.isfinite(float(unit.loss)))
