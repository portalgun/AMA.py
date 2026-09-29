"""
Shared imports and helpers of the ama package: densities, latent geometry, the _Static/_TypeFunc machinery, and
configuration utilities. Every module does `from ._base import *`, which (by __all__) includes the private names.
"""


import copy
import functools
import importlib
import inspect
import pickle
import sys
import warnings
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jxrandom
import numpy.random as random
import optax
from jax import jit,vmap,lax,value_and_grad,tree_util
from jax.scipy.special import logsumexp
from jax._src.numpy.util import promote_dtypes_inexact
from functools import partial
from scipy.io import loadmat
from itertools import combinations, product


class _LazyModule:
    """imports a module on first attribute access, so that importing ama does not load plotting libraries"""
    def __init__(self, name):
        self._name=name

    def __getattr__(self, attr):
        return getattr(importlib.import_module(self._name), attr)

plt=_LazyModule('matplotlib.pyplot')
cm=_LazyModule('matplotlib.cm')
filt=_LazyModule('Filter')

# optax transformations by (optimizerType, lRate0), shared by all Optimizers (see Optimizer.tx)
_TX_CACHE={}


def contrast_normalize(stimuli):
    """mean-subtract each stimulus (last axis) and scale it to unit L2 norm, as AMA assumes (Burge & Jaini 2017)"""
    s=np.asarray(stimuli,dtype=float)
    flat=np.reshape(s,(-1,s.shape[-1]))
    flat=flat-flat.mean(axis=0,keepdims=True)
    norm=np.linalg.norm(flat,axis=0,keepdims=True)
    return np.reshape(flat/np.where(norm>0,norm,1),s.shape)

def _user_stacklevel():
    # the warnings.warn stacklevel of the first caller outside this package, for the function that calls this one: a
    # warning then points at the user's line whichever entry point (train_new, load, from_config, ...) led to it
    import os
    d=os.path.dirname(os.path.abspath(__file__))
    f,level=sys._getframe(1),1
    while f is not None and os.path.dirname(os.path.abspath(f.f_code.co_filename))==d:
        f,level=f.f_back,level+1
    return level

def _warn_if_not_contrast_normalized(stimuli,tol=1e-3):
    # stimuli [ nPix x nStim ]; the mean component's norm is |mean|*sqrt(nPix)
    norm=np.linalg.norm(stimuli,axis=0)
    mean=np.abs(stimuli.mean(axis=0))*np.sqrt(stimuli.shape[0])
    if np.any(np.abs(norm-1)>tol) or np.any(mean>tol):
        warnings.warn('stimuli are not contrast normalized (zero mean, unit norm), which AMA assumes; '
                      'pass bContrastNormalize=True to Stim or use ama.contrast_normalize',stacklevel=_user_stacklevel())

def _get_copy_dict(instance,excl=[]):
    flds = [attr for attr in dir(instance) if not attr.startswith('_') and attr not in excl and not callable(getattr(instance,attr))]

    dict={}
    for fld in flds:
        val=getattr(instance,fld)
        if hasattr(val,'val'):
            dict[fld]=val.val
        else:
            dict[fld]=val
    return dict


#- yaml configuration

CONFIG_VERSION=1
_NRN_EXCL=['filter','corrType','bAnalytic','bFinalized','bSplit','bFourier','dtype','bReadout','bPooledReadout','bParams',
           'bComplexOut','nIn','nChan','nSub','nOut','nDimOut']
_OPT_EXCL=['loss_hist','tx','val_hist','best_step','stop_reason']


def _yaml_safe(v):
    """plain python (yaml-safe) version of a config value: numpy/jax scalars and arrays, tuples, dtypes"""
    if v is None or isinstance(v,(bool,int,float,str)):
        return v
    if isinstance(v,(np.bool_,)):
        return bool(v)
    if isinstance(v,np.integer):
        return int(v)
    if isinstance(v,np.floating):
        return float(v)
    if isinstance(v,dict):
        return {str(k):_yaml_safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):
        return [_yaml_safe(x) for x in v]
    if isinstance(v,(np.ndarray,jnp.ndarray)) or hasattr(v,'__array__'):
        a=np.asarray(v)
        if np.iscomplexobj(a):
            return {'real':_yaml_safe(a.real.tolist()),'imag':_yaml_safe(a.imag.tolist())}
        return a.tolist()
    if isinstance(v,(np.dtype,type)) or hasattr(v,'dtype') is False and 'dtype' in type(v).__name__.lower():
        return np.dtype(v).name
    if callable(v):
        return getattr(v,'__name__',str(v))
    return str(v)


def _components_config(nrn,model,objective,optimizer):
    return {'nrn':_yaml_safe(_get_copy_dict(nrn,_NRN_EXCL)),
            'model':_yaml_safe(_get_copy_dict(model)),
            'objective':_yaml_safe(_get_copy_dict(objective)),
            'optimizer':None if optimizer is None else _yaml_safe(_get_copy_dict(optimizer,_OPT_EXCL))}


def _logged_training(fn):
    """record a training call (method and arguments) in Unit.train_log, for Unit.config and Unit.from_config"""
    sig=inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(self,*args,**kwargs):
        if getattr(self,'_bTrainLogging',False):
            return fn(self,*args,**kwargs)
        bound=sig.bind(self,*args,**kwargs)
        bound.apply_defaults()                 # record every argument: a config stays reproducible if defaults change
        rec={}
        for k,v in list(bound.arguments.items())[1:]:
            if k=='stimVal':
                rec[k]=None if v is None else {'given':True,'nStim':int(np.sum(np.asarray(v.weights)>0))}
            elif k=='optimizer':
                # the optimizer in effect (the unit's own when none is given), so a replay does not use a later one
                v=v if v is not None else getattr(self,'optimizer',None)
                rec[k]=None if v is None else _yaml_safe(_get_copy_dict(v,_OPT_EXCL))
            elif k=='dtype':
                rec[k]=None if v is None else np.dtype(v).name
            else:
                rec[k]=_yaml_safe(v)
        self._bTrainLogging=True
        try:
            out=fn(self,*args,**kwargs)
        finally:
            self._bTrainLogging=False
        if not hasattr(self,'train_log') or self.train_log is None:
            self.train_log=[]
        self.train_log.append({'method':fn.__name__,'args':rec})
        return out
    return wrapper


def source_sha256():
    """sha256 of the package's source files (names and contents, in order): identifies the code a unit was trained or
    configured with (optimization paths can differ numerically between versions)"""
    import hashlib
    import os
    h=hashlib.sha256()
    d=os.path.dirname(os.path.abspath(__file__))
    for fn in sorted(os.listdir(d)):
        if fn.endswith('.py'):
            with open(os.path.join(d,fn),'rb') as fh:
                h.update(fn.encode()+b'\0'+fh.read())
    return h.hexdigest()


def load_config(fname):
    """read an AMA yaml configuration (see Unit.config)"""
    import yaml
    with open(fname) as fh:
        return yaml.safe_load(fh)


def config_from_saved(fname):
    """the yaml configuration (Unit.config) of a unit saved with Unit.save, without loading stimuli"""
    with open(fname,'rb') as fh:
        state=pickle.load(fh)
    fin=state['finalize']
    cfg={'ama_config_version':CONFIG_VERSION,'ama_source_sha256':state.get('ama_source_sha256'),'name':state.get('name'),
         'seed':_yaml_safe(state['seed']),
         'nrn':_yaml_safe(state['nrn']),'model':_yaml_safe(state['model']),'objective':_yaml_safe(state['objective']),
         'optimizer':_yaml_safe(state['optimizer']),
         'filters':{'n':int(fin['n']),'fourierType':(2 if fin['bAnalytic'] else 1) if fin['bFourier'] else 0,
                    'dtype':str(fin['dtype']),'bSplit':bool(fin['bSplit']),
                    'shape':list(np.shape(state['out'])),'bPoolWeights':state.get('pool_p') is not None},
         'train':_yaml_safe(state.get('train_log')),
         'stim':_yaml_safe(state.get('stim_summary'))}
    return cfg


def _id(ins,*_):
    return ins

class _Static:
    # jit treats these config objects as static arguments: equal configurations share compiled traces,
    # and a changed configuration gets a new trace
    def __hash__(self):
        return hash((type(self),self._key()))

    def __eq__(self,other):
        return type(self) is type(other) and self._key()==other._key()

def _tri_inv(L):
    """
    L^-1 of lower triangular factors L [ ... x n x n ], at their own batch shape. The densities apply it by broadcast
    multiply-sums: a triangular solve per (stimulus, category) pair would repeat thousands of tiny solves against the
    same few factors, which XLA:CPU runs one by one (20-40x slower than this on the CPU; the same on the GPU)
    """
    eye=jnp.broadcast_to(jnp.eye(L.shape[-1],dtype=L.dtype),L.shape)
    return lax.linalg.triangular_solve(L,eye,left_side=True,lower=True)

def _tri_apply(Li,x0):
    # L^-1 x0 [ ... x n ] for Li = L^-1 [ ... x n x n ] (broadcast): fused, nothing [ ... x n x n ] is stored per stimulus
    return jnp.sum(Li*x0[...,None,:],axis=-1)

@jit
def lmvn0(x0,cov):
    x0, cov = promote_dtypes_inexact(x0,cov)
    L = lax.linalg.cholesky(cov)
    y = _tri_apply(_tri_inv(L),x0)
    return (-1/2 * jnp.einsum('...i,...i->...', y, y)
            - cov.shape[-1]/2 * jnp.log(2*np.pi)
            - jnp.log(L.diagonal(axis1=-1, axis2=-2)).sum(-1))

@jit
def lmvt0(x0,scale,df):
    """log density of a multivariate student t with location 0, scale matrix and df degrees of freedom"""
    x0, scale = promote_dtypes_inexact(x0,scale)
    L = lax.linalg.cholesky(scale)
    y = _tri_apply(_tri_inv(L),x0)
    d = scale.shape[-1]
    q = jnp.einsum('...i,...i->...', y, y)
    return (jax.scipy.special.gammaln((df+d)/2) - jax.scipy.special.gammaln(df/2)
            - d/2*jnp.log(df*np.pi)
            - jnp.log(L.diagonal(axis1=-1, axis2=-2)).sum(-1)
            - (df+d)/2*jnp.log1p(q/df))

@jit
def lcn0(z0,cov):
    """log density of a circular (proper) complex gaussian CN(0, cov), z0 complex [ ... x n ], cov hermitian"""
    L = jnp.linalg.cholesky(cov)
    y = _tri_apply(_tri_inv(L),z0)
    n = cov.shape[-1]
    return (-jnp.real(jnp.einsum('...i,...i->...', jnp.conj(y), y))
            - n*jnp.log(np.pi)
            - 2*jnp.log(jnp.real(L.diagonal(axis1=-1, axis2=-2))).sum(-1))

def _flatten_responses(X):
    """
    [ nF x nStim_Ctg x nCtg ] or [ nF x nSplit x nStim_Ctg x nCtg ], real or complex
    -> real [ nF' x nStim_Ctg x nCtg ]
    complex components (quadrature pairs) and split sub-filters each become their own response dimension
    """
    if jnp.iscomplexobj(X):
        X=jnp.concatenate((X.real,X.imag),axis=0)
    return jnp.reshape(X,(-1,)+X.shape[-2:])

def _centered_freqs(n):
    return np.fft.fftshift(np.fft.fftfreq(n))

_SQRT2=float(np.sqrt(2))                     # a python float keeps the precision of the array it scales

def _safe_divide(num,den):
    # zero denominators (e.g. the all-zero padding stimuli) give 0 rather than nan, with finite gradients
    return num/jnp.where(den!=0,den,1)

#- latent geometry: Y [ nCtg ] or [ nCtg x nDim ]; per (Stim.Yperiod) is None or one period (None = linear) per dimension
def _period_arrays(per,dtype=float):
    # periods and circular mask over the latent dimensions (period 1 on linear dimensions)
    P=np.array([0. if q is None else float(q) for q in per])
    return jnp.asarray(np.where(P>0,P,1.),dtype=dtype),jnp.asarray(P>0)

def _wrap(d,per):
    """differences of latent values d [ ... ] (1-D Y) or [ ... x nDim ], wrapped into [-P/2, P/2) on circular dimensions"""
    if per is None:
        return d
    P,bC=_period_arrays(per,d.dtype)
    if len(per)==1:
        P,bC=P[0],bC[0]
    return jnp.where(bC,d-P*jnp.floor(d/P+0.5),d)

class _ErrOpts(tuple):
    """static options of the error functions: the latent grid (Stim levels on a cartesian grid), whether the target is
    one-hot, and the entropic optimal transport settings"""
    def __new__(cls,grid=None,bOneHot=True,otEps=0.05,nOtIter=200):
        return super().__new__(cls,(grid,bOneHot,otEps,nOtIter))
    grid=property(lambda self: self[0])
    bOneHot=property(lambda self: self[1])
    otEps=property(lambda self: self[2])
    nOtIter=property(lambda self: self[3])

def _latent_grid(Y):
    """
    for Y [ nCtg x nDim ] whose levels are all combinations of per-dimension values: (grid shape, order) with order the
    categories in C order over the ascending values of each dimension; else None
    """
    Y=np.asarray(Y)
    if Y.ndim!=2:
        return None
    vals=[np.unique(Y[:,d]) for d in range(Y.shape[1])]
    shape=tuple(len(v) for v in vals)
    if int(np.prod(shape))!=len(Y):
        return None
    pos=np.stack([np.searchsorted(v,Y[:,d]) for d,v in enumerate(vals)],1)
    flat=np.ravel_multi_index(pos.T,shape)
    if len(np.unique(flat))!=len(Y):
        return None
    order=np.empty(len(Y),dtype=int)
    order[flat]=np.arange(len(Y))
    return shape,tuple(int(i) for i in order)

def _sinkhorn_value(la,lb,C,eps,nIter):
    """
    entropic optimal transport value OT_eps(a,b) between log-weights la [ ... x n ] and lb [ ... x m ] (broadcast) with cost
    C [ n x m ]: the dual value sum a f + sum b g at the Sinkhorn potentials. The potentials are not differentiated
    (envelope theorem: the gradient of the value in a is f), so no iteration is stored for the backward pass
    """
    la_,lb_=lax.stop_gradient(la),lax.stop_gradient(lb)
    shape=jnp.broadcast_shapes(la_.shape[:-1],lb_.shape[:-1])
    f=jnp.zeros(shape+(C.shape[0],),dtype=C.dtype)
    g=jnp.zeros(shape+(C.shape[1],),dtype=C.dtype)
    def body(_,fg):
        f,g=fg
        f=-eps*logsumexp((g[...,None,:]-C)/eps + lb_[...,None,:],axis=-1)
        g=-eps*logsumexp((f[...,:,None]-C)/eps + la_[...,:,None],axis=-2)
        return f,g
    f,g=lax.fori_loop(0,nIter,body,(f,g))
    f,g=lax.stop_gradient(f),lax.stop_gradient(g)
    return jnp.sum(jnp.exp(la)*f,axis=-1)+jnp.sum(jnp.exp(lb)*g,axis=-1)

def _take_stim(A,idx):
    # rows idx [ m x nCtg ] of A [ nStim_Ctg x nCtg (x nDim) ] within each category
    return jnp.take_along_axis(A,idx.reshape(idx.shape+(1,)*(A.ndim-2)),axis=0)

def _ysq(d,Y):
    # squared distance of (wrapped) differences, summed over latent dimensions
    return d**2 if Y.ndim==1 else jnp.sum(d**2,axis=-1)

def _ydist2(Y,per):
    """[ nCtg x nCtg ] squared (wrapped) distances between latent values"""
    return _ysq(_wrap(Y[None,:]-Y[:,None],per),Y)

def _inv_sqrtm(M,nIter=40):
    """
    M^-1/2 of a symmetric positive definite matrix by the coupled Newton-Schulz iteration. Unlike an
    eigendecomposition, its gradients stay finite when eigenvalues repeat (e.g. orthonormal filters, M = I).
    """
    I=jnp.eye(M.shape[0],dtype=M.dtype)
    s=jnp.trace(M)
    def step(c,_):
        Y,Z=c
        T=0.5*(3*I-Z@Y)
        return (Y@T,T@Z),None
    (_,Z),_=lax.scan(step,(M/s,I),None,length=nIter)
    return Z/jnp.sqrt(s)

def _centered_ft(v,axes,inverse=False):
    """orthonormal fourier transform on a centered grid (origin and DC in the middle) over the given axes"""
    fun=jnp.fft.ifftn if inverse else jnp.fft.fftn
    return jnp.fft.fftshift(fun(jnp.fft.ifftshift(v,axes=axes),axes=axes,norm='ortho'),axes=axes)

def _plot_signal(y,bFourier=False,clim=None):
    """1D: line plot (real, imaginary, and magnitude when complex). 2D: image (real part, with the imaginary part alongside when complex)"""
    y=np.asarray(y)
    if np.iscomplexobj(y) and not np.any(np.imag(y)!=0):
        y=np.real(y)
    if y.ndim==1:
        x=_centered_freqs(len(y)) if bFourier else np.arange(len(y))-len(y)//2
        filt.plotFT(x,y)
        if clim is not None:
            plt.ylim(*clim)
    elif y.ndim==2:
        if np.iscomplexobj(y) and np.any(np.imag(y)!=0):
            plt.imshow(np.concatenate((np.real(y),np.imag(y)),axis=1))
            plt.title('real | imaginary')
        else:
            plt.imshow(np.real(y))
            if clim is not None:
                plt.clim(*clim)
    else:
        raise Exception('plotting is only implemented for 1D and 2D stimuli')

class _BoundLoss:
    """
    a loss method of a settings-hashed object (_Static) for use as a static jit argument. Bound methods compare their
    objects by identity, so passing unit._loss_fun_lrn directly would compile training again for every new Unit.
    """
    def __init__(self,owner,name):
        self.owner=owner
        self.name=name

    def __call__(self,*args):
        return getattr(self.owner,self.name)(*args)

    def __hash__(self):
        return hash((self.owner,self.name))

    def __eq__(self,other):
        return isinstance(other,_BoundLoss) and self.name==other.name and self.owner==other.owner

def _freeze(v):
    """a hashable version of a (nested) configuration value: dicts, sequences, arrays, scalars"""
    if isinstance(v,dict):
        return ('dict',tuple(sorted((str(k),_freeze(x)) for k,x in v.items())))
    if isinstance(v,(list,tuple)):
        return ('seq',tuple(_freeze(x) for x in v))
    if isinstance(v,(np.ndarray,jnp.ndarray)):
        a=np.asarray(v)
        return ('arr',a.dtype.str,a.shape,a.tobytes())
    if isinstance(v,np.generic):
        return v.item()
    return v

class _GeneratedLoss:
    """
    the cost of filters generated from parameters (Unit.train_parametric, train_multiscale), for use as a static jit
    argument: hashed by the unit's settings, the generator (kind and its configuration), and the learned frequencies
    the generator reads from the unit, so repeated training with the same configuration reuses the compiled steps
    """
    def __init__(self,unit,kind,cfg):
        self.unit=unit
        self.kind=kind
        self.cfg=cfg
        flt=unit.filter
        self._k=(kind,_freeze(cfg),tuple(flt.pix_dims),np.asarray(flt.index.pix).tobytes())

    def full(self,prm):
        out={'f':self.unit._generated_filters(self.kind,prm,self.cfg)}
        if 'p' in prm:
            out['p']=prm['p']
        return out

    def __call__(self,prm,rng_key,stimval,stimweights,yCtg,Y):
        cost=self.unit._loss_fun(self.full(prm),rng_key,stimval,stimweights,yCtg,Y,None)
        pen=self.unit._generated_penalty(self.kind,prm,self.cfg)
        return cost if pen is None else cost+pen

    def heldout(self,prm,rng_key,stimval,stimweights,yCtg,Y,refval,refweights,refy):
        return self.unit._loss_fun_heldout(self.full(prm),rng_key,stimval,stimweights,yCtg,Y,refval,refweights,refy)

    def __hash__(self):
        return hash((self.unit,self._k))

    def __eq__(self,other):
        return isinstance(other,_GeneratedLoss) and self._k==other._k and self.unit==other.unit

@partial(jit, static_argnames=['loss'])
def _generated_heldout(loss,prm,rng_key,stimval,stimweights,yCtg,Y,refval,refweights,refy):
    return loss.heldout(prm,rng_key,stimval,stimweights,yCtg,Y,refval,refweights,refy)

class _ParentProp:
    def __init__(self, pname=None,default=None):
        self._pname=pname
        self.default = default

    @property
    def pattr(self):
        if self._pname:
            return self._pname
        else:
            return self.name

    def __set_name__(self, _, name):
        self.name = name

    def __get__(self, instance, _):
        if instance is None:
            return self
        elif not hasattr(instance,'_parent'):
            raise Exception(type(instance).__name__ + ' has no "_parent" attribute')
        elif getattr(instance,'_parent') is None:
            raise Exception('_parent not set')
        elif not hasattr(getattr(instance,'_parent'), self.pattr):
            raise Exception(type(instance._parent).__name__ + ' has does not have attribute set')
        else:
            return getattr( getattr(instance,'_parent'), self.pattr)


class _TypeFunc():
    # descriptor

    def __init__(self, bBinary=False,default=None):

        #- binary
        if not isinstance(bBinary,bool):
            raise Exception('First argument must be bool')
        self.bBinary=bBinary

        #- default
        if default is None:
            if self.bBinary:
                default=False
            else:
                default='none'
        else:
            if self.bBinary and not isinstance(default,bool):
                raise Exception('Default for binary type must be bool')
            elif not isinstance(default,str):
                raise Exception('Default must be a string')
        self.default=default

    def __set_name__(self, _, name):
        if self.bBinary and name[0]!='b':
            raise Exception('binary _TypeFuncs must have a name that begins with "b"')
        elif not self.bBinary and name[-4:]!='Type':
            raise Exception('_TypeFuncs must have a name that ends with "Type"')
        self.name = name

    def __get__(self, instance, _):
        if instance is None:
            return self
        return instance.__dict__.get(self.name,self.default)

    def __set__(self, instance, value):
        self.instance = instance
        if self.bBinary:
            if not isinstance(value,bool) and value is not None:
                raise Exception('Value must be bool')
        elif not isinstance(value,str) and value is not None:
            raise Exception('Value must be a string')

        instance.__dict__[self.name]=value
        if not hasattr(self.instance,self.func_name):
            raise Exception('Class ' + self.instance_class + ' has no method ' + self.func_name)

        setattr(instance, self.wrap_fun_name, self.func)

    @property
    def wrap_fun_name(self):
        return '_' + self._base_func_name + '_fun'


    @property
    def instance_class(self):
        return type(self.instance).__name__

    @property
    def func(self):
        return getattr(self.instance, self.func_name)

    @property
    def func_name(self):
        return '_' + self._base_func_name.split('_')[0] + '__' + self._inst_func_name


    @property
    def _base_func_name(self):
        if self.bBinary:
            return self.name[1:].lower()
        else:
            return self.name[:-4].lower()

    @property
    def _inst_func_name(self):
        val=self.instance.__dict__.get(self.name,self.default)
        if self.bBinary:
            if val:
                return 'true'
            else:
                return 'none'
        else:
            if not val or ( isinstance(val,str) and val.lower()=='none'):
                return 'none'
            else:
                return val


__all__=[k for k in list(globals()) if not k.startswith('__')]
