"""
Response whitening (Nrn whitenType), intended for full AMA: the mean responses are decorrelated before noise.
  'gram'     - by the Gram matrix of the (real, implied spatial) filters: equivalent to orthonormalizing the filters
  'response' - by the covariance of the mean responses over the stimuli: decorrelates the responses themselves
whitenMethod 'zca' (symmetric) or 'chol' (Gram-Schmidt order); both keep each dimension's variance (filter norm).
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.linalg import sqrtm, cholesky

import ama
import stimuli as ts
from test_ama import loss_of
from test_extended import unit_from, ungroup


METHODS=['zca','chol']

CONFIGS=[
    (ts.gaussian_ctg,{},{}),
    (ts.image_orientation,{},{}),
    (ts.sine_frequency,{},{'fourierType':1,'dtype':jnp.complex128}),
    (ts.sine_frequency,{},{'fourierType':2,'dtype':jnp.complex128}),
    (ts.binocular_shift,{'nSplit':2},{'bSplit':True}),
    (ts.binocular_shift,{'nSplit':2},{'bSplit':True,'fourierType':2,'dtype':jnp.complex128}),
]


#- helpers

def flat(x):
    return np.asarray(ama._flatten_responses(x))


def numpy_whitening(M,method):
    d=np.sqrt(np.diag(M))
    Minvh=np.linalg.inv(np.real(sqrtm(M))) if method=='zca' else np.linalg.inv(cholesky(M,lower=True))
    return d[:,None]*Minvh


def valid_cov(R,weights):
    X,_=ungroup(R,weights)
    return np.cov(X)


def config_unit(gen,stim_kw,fin_kw,nrn=None,modelType='full',n=2):
    return unit_from(ama.Stim(*gen()[:4],**stim_kw),nrn,modelType,n=n,**fin_kw)


def spatial_stimuli(gen,stim_kw,fin_kw):
    """spatial-domain stimuli [ (nSplit x) nPix x nStim_Ctg x nCtg ], parts flattened in (part, pixel) order"""
    bSplit=fin_kw.get('bSplit',False)
    st=ama.Stim(*gen()[:4],**stim_kw)
    st._finalize(jnp.float64,None,False,bSplit)
    v=np.asarray(st.val)
    if bSplit:
        v=np.moveaxis(v,1,0).reshape(-1,*v.shape[2:])
    return v


def implied_rows(unit):
    """one real spatial filter per flattened response dimension, over (part, pixel), zero outside its part"""
    imp=np.asarray(unit.filter.implied_spatial())
    S=unit.filter.nSplit if unit.filter.bSplit else 1
    imp=imp.reshape(-1,S,imp.shape[-1])                                     # [ pixels x parts x filters ]
    comps=[np.real,np.imag] if np.iscomplexobj(imp) else [np.real]
    rows=[]
    for comp in comps:
        for i in range(imp.shape[-1]):
            for e in range(S):
                row=np.zeros((S,imp.shape[0]))
                row[e]=comp(imp[:,e,i])
                rows.append(row.ravel())
    return np.array(rows)


#- gram whitening

@pytest.mark.parametrize('method',METHODS)
def test_gram_whitening_orthonormalizes_filters(method):
    x,s,ci,Y,_=ts.unequal_counts()
    f=np.random.default_rng(0).standard_normal((12,3))*np.array([1.,2.,0.5])
    f[:,1]+=f[:,0]                                                          # correlated, unequal norms
    unit=unit_from(ama.Stim(x,s,ci,Y),ama.Nrn(whitenType='gram',whitenMethod=method,whitenEps=0.),'full',n=3,f=f)
    W=numpy_whitening(f.T@f,method)
    r=5.7*np.einsum('pf,pnc->fnc',f,np.asarray(unit.stim.val))
    assert np.allclose(flat(unit.responses.R),np.einsum('ij,jnc->inc',W,r))
    Fw=f@W.T                                                                # the filters actually applied
    assert np.allclose(Fw.T@Fw,np.diag(np.diag(f.T@f)))
    assert np.allclose(np.asarray(unit.nrn.whitening(unit.stim.val,unit.filter.out_flat,unit.stim.weights)),W)


@pytest.mark.parametrize('gen,stim_kw,fin_kw',CONFIGS)
def test_gram_and_responses_follow_implied_filters(gen,stim_kw,fin_kw):
    unit=config_unit(gen,stim_kw,fin_kw)
    E=implied_rows(unit)
    assert np.allclose(np.asarray(unit.nrn.gram(unit.filter.out_flat)),E@E.T)
    assert np.allclose(flat(unit.responses.R),5.7*np.einsum('dp,pnc->dnc',E,spatial_stimuli(gen,stim_kw,fin_kw)))


@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('gen,stim_kw,fin_kw',CONFIGS)
def test_gram_whitening_all_configurations(gen,stim_kw,fin_kw,method):
    unit=config_unit(gen,stim_kw,fin_kw,ama.Nrn(whitenType='gram',whitenMethod=method,whitenEps=0.))
    G=np.asarray(unit.nrn.gram(unit.filter.out_flat))
    W=np.asarray(unit.nrn.whitening(unit.stim.val,unit.filter.out_flat,unit.stim.weights))
    assert np.allclose(W@G@W.T,np.diag(np.diag(G)))
    E=implied_rows(unit)
    assert np.allclose(flat(unit.responses.R),5.7*np.einsum('dp,pnc->dnc',W@E,spatial_stimuli(gen,stim_kw,fin_kw)))


@pytest.mark.parametrize('method',METHODS)
def test_gram_whitening_of_orthonormal_filters_changes_nothing(method):
    x,s,ci,Y,_=ts.gaussian_ctg()
    q,_=np.linalg.qr(np.random.default_rng(0).standard_normal((16,3)))
    plain=unit_from(ama.Stim(x,s,ci,Y),None,'full',n=3,f=q)
    white=unit_from(ama.Stim(x,s,ci,Y),ama.Nrn(whitenType='gram',whitenMethod=method,whitenEps=0.),'full',n=3,f=q)
    assert np.isclose(float(white.loss),float(plain.loss),rtol=1e-9)


#- response whitening

@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('gen,stim_kw,fin_kw',[(ts.unequal_counts,{},{})]+CONFIGS[2:])
def test_response_whitening_decorrelates_responses(gen,stim_kw,fin_kw,method):
    plain=config_unit(gen,stim_kw,fin_kw)
    white=config_unit(gen,stim_kw,fin_kw,ama.Nrn(whitenType='response',whitenMethod=method,whitenEps=0.))
    C0=valid_cov(flat(plain.responses.R),plain.stim.weights)
    C=valid_cov(flat(white.responses.R),white.stim.weights)
    assert np.allclose(C,np.diag(np.diag(C0)))                             # decorrelated, variances kept
    W=numpy_whitening(C0,method)
    assert np.allclose(flat(white.responses.R),np.einsum('ij,jnc->inc',W,flat(plain.responses.R)))


def test_noise_is_added_after_whitening():
    unit=unit_from(ama.Stim(*ts.unequal_counts()[:4]),ama.Nrn(whitenType='response',bNoise_2=True),'full','basic',n=3)
    f=unit.filter.out_flat
    w=unit.stim.weights
    outs=jax.vmap(lambda k: unit.nrn.main(k,unit.stim.val,f,w))(jax.random.split(jax.random.key(0),3000))
    R=np.asarray(outs[2][0])
    RNs=np.asarray(outs[3])
    valid=np.asarray(w)>0
    z=((RNs-R)/np.sqrt(1.36*np.abs(R)+0.23))[:,:,valid]                     # standardized noise [ draws x dims x stimuli ]
    C=np.corrcoef(np.moveaxis(z,1,0).reshape(3,-1))
    assert np.allclose(C,np.eye(3),atol=0.02)


#- gradients and learning

@pytest.mark.parametrize('whitenType',['gram','response'])
@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('bOrthonormal',[False,True])
def test_whitening_gradients_match_finite_differences(whitenType,method,bOrthonormal):
    x,s,ci,Y,_=ts.unequal_counts()
    f=np.random.default_rng(0).standard_normal((12,3))
    f=np.linalg.qr(f)[0] if bOrthonormal else f/np.linalg.norm(f,axis=0)
    unit=unit_from(ama.Stim(x,s,ci,Y),ama.Nrn(whitenType=whitenType,whitenMethod=method),'full',n=3,f=f)
    fj=jnp.asarray(f)
    g=np.asarray(jax.grad(lambda f: loss_of(unit,f))(fj))
    assert np.all(np.isfinite(g))
    for idx in [(0,0),(5,2)]:
        e=np.zeros(f.shape)
        e[idx]=1e-6
        fd=(float(loss_of(unit,fj+e))-float(loss_of(unit,fj-e)))/2e-6
        assert np.isclose(g[idx],fd,rtol=1e-4,atol=1e-6)


@pytest.mark.parametrize('whitenType,method',[('gram','zca'),('gram','chol'),('response','zca'),('response','chol')])
def test_full_ama_learning_with_whitening(whitenType,method):
    x,s,ci,Y,_=ts.gaussian_ctg()
    unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(whitenType=whitenType,whitenMethod=method),ama.Model('full','mean'),ama.Objective('map'),
                  ama.Optimizer(nIterMax=60,lRate0=0.05,bVerbose=False))
    unit.train_new(2)
    h=unit.optimizer.loss_hist
    assert np.all(np.isfinite(h)) and h[-1]<h[0]
    f=unit.filter.out_flat
    W=np.asarray(unit.nrn.whitening(unit.stim.val,f,unit.stim.weights))
    M=np.asarray(unit.nrn.gram(f)) if whitenType=='gram' else valid_cov(flat(ama.Nrn._respond(unit.nrn,f,unit.stim.val)),unit.stim.weights)
    Mw=W@M@W.T
    corr=Mw/np.sqrt(np.outer(np.diag(Mw),np.diag(Mw)))
    # decorrelated up to the whitenEps ridge, which matters most when M is nearly singular
    assert np.abs(corr[~np.eye(len(corr),dtype=bool)]).max()<0.02
    assert np.allclose(np.diag(Mw),np.diag(M),rtol=0.01)


#- frozen whitening, validation, configuration changes

def test_frozen_whitening_for_held_out_stimuli():
    x,s,ci,Y,_=ts.gaussian_ctg()
    unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(whitenType='response'),ama.Model('full','mean'),ama.Objective('map'),
                  ama.Optimizer(nIterMax=20,lRate0=0.05,bVerbose=False))
    unit.train_new(2)
    unit.freeze_whitening()
    W=np.asarray(unit.nrn._W)
    sub=unit.split(stimInd=np.arange(10))
    assert np.allclose(np.asarray(sub.nrn._W),W)
    R_full=flat(unit.responses.R)
    assert np.allclose(flat(sub.responses.R),R_full[:,:10])
    assert np.isfinite(float(sub.loss))
    sub.unfreeze_whitening()
    assert not np.allclose(flat(sub.responses.R),R_full[:,:10])           # re-estimated on the subset
    unit.train_recurse()
    assert unit.nrn._W is None


def test_whitening_validation():
    with pytest.raises(Exception,match='narrow'):
        unit_from(ama.Stim(*ts.sine_frequency()[:4]),ama.Nrn(whitenType='gram',normalizeType='narrow'),'full',fourierType=1,dtype=jnp.complex128)
    x,s,ci,Y,_=ts.unequal_counts(nPix=12,counts=(3,3,3))
    with pytest.raises(Exception,match='more stimuli'):
        unit_from(ama.Stim(x,s,ci,Y),ama.Nrn(whitenType='response'),'full',n=9)
    with pytest.raises(Exception,match='whitenMethod'):
        unit_from(ama.Stim(x,s,ci,Y),ama.Nrn(whitenType='gram',whitenMethod='pca'),'full')
    unit=unit_from(ama.Stim(x,s,ci,Y),None,'full')
    with pytest.raises(Exception,match='freeze_whitening'):
        unit.freeze_whitening()


def test_changing_whitening_settings_is_not_stale():
    x,s,ci,Y,_=ts.unequal_counts()
    f=np.random.default_rng(3).standard_normal((12,3))
    unit=unit_from(ama.Stim(x,s,ci,Y),ama.Nrn(whitenType='gram',whitenMethod='zca'),'full',n=3,f=f)
    before=float(unit.loss)
    unit.nrn.whitenMethod='chol'
    fresh=unit_from(ama.Stim(x,s,ci,Y),ama.Nrn(whitenType='gram',whitenMethod='chol'),'full',n=3,f=f)
    assert not np.isclose(before,float(unit.loss))
    assert np.isclose(float(unit.loss),float(fresh.loss))
    unit.nrn.whitenType='None'
    plain=unit_from(ama.Stim(x,s,ci,Y),None,'full',n=3,f=f)
    assert np.isclose(float(unit.loss),float(plain.loss))
    assert ama.Nrn(whitenType='response',whitenMethod='chol').copy()._key()==ama.Nrn(whitenType='response',whitenMethod='chol')._key()
