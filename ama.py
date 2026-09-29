"""
Accuracy Maximization Analysis (AMA) with JAX and optax (Burge & Jaini 2017; AMA-Gauss: Jaini & Burge 2017).

Pipeline (Unit._loss_fun*):
    Stim -> Nrn (responses, whitening, noise, normalization) -> Model (log-likelihoods)
         -> Objective (posterior, estimate, error, loss) -> Optimizer (optax, unit-norm filters)

Shapes (stimuli are grouped by category and padded with zeros; Stim.weights marks the valid stimuli):
    stim  [ nPix x nStim_Ctg x nCtg ]          split: [ nPix/nSplit x nSplit x nStim_Ctg x nCtg ]
    f     [ nPix x nF ]                        split: [ nPix/nSplit x nSplit x nF ]
    R     [ nF x nStim_Ctg x nCtg ]            split: [ nF x nSplit x nStim_Ctg x nCtg ]
          flattened to real dimensions [ nDim x nStim_Ctg x nCtg ] for the likelihood (_flatten_responses)
    lAll  [ nStim_Ctg x nCtg(true) x nCtg(candidate) ]   log-likelihoods; log-posteriors after Objective
"""


import copy
import functools
import importlib
import inspect
import pickle
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

def _warn_if_not_contrast_normalized(stimuli,tol=1e-3):
    # stimuli [ nPix x nStim ]; the mean component's norm is |mean|*sqrt(nPix)
    norm=np.linalg.norm(stimuli,axis=0)
    mean=np.abs(stimuli.mean(axis=0))*np.sqrt(stimuli.shape[0])
    if np.any(np.abs(norm-1)>tol) or np.any(mean>tol):
        warnings.warn('stimuli are not contrast normalized (zero mean, unit norm), which AMA assumes; '
                      'pass bContrastNormalize=True to Stim or use ama.contrast_normalize',stacklevel=3)

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
    """sha256 of this ama.py source: identifies the code a unit was trained or configured with (optimization paths can
    differ numerically between versions)"""
    import hashlib
    with open(__file__,'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


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

@jit
def lmvn0(x0,cov):
    x0, cov = promote_dtypes_inexact(x0,cov)
    L = lax.linalg.cholesky(cov)
    y = jnp.vectorize(
            partial(lax.linalg.triangular_solve, lower=True, transpose_a=True),
            signature="(n,n),(n)->(n)"
        )(L, x0)
    return (-1/2 * jnp.einsum('...i,...i->...', y, y)
            - cov.shape[-1]/2 * jnp.log(2*np.pi)
            - jnp.log(L.diagonal(axis1=-1, axis2=-2)).sum(-1))

@jit
def lmvt0(x0,scale,df):
    """log density of a multivariate student t with location 0, scale matrix and df degrees of freedom"""
    x0, scale = promote_dtypes_inexact(x0,scale)
    L = lax.linalg.cholesky(scale)
    y = jnp.vectorize(
            partial(lax.linalg.triangular_solve, lower=True, transpose_a=True),
            signature="(n,n),(n)->(n)"
        )(L, x0)
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
    y = jnp.vectorize(
            partial(jax.scipy.linalg.solve_triangular, lower=True),
            signature="(n,n),(n)->(n)"
        )(L, z0)
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

_FULL_CHUNK=2**24                            # full AMA: elements of the [ observed x reference x category ] terms per chunk
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

class Stim:
    def index(self,index=(0,0)):
        """
        stim [ nPix  x nStim ] -> [ nPix  x nStim_Ctg x nCtg ]        [(nPix / nSplit) x nSplit x nStim_Ctg x nCtg]
        """
        return jnp.reshape(self.val[...,index[0],index[1]],self.val.shape[:-2] if self.bIsSplit else self.dims)

    def __getitem__(self,index):
        return self._subset(index)

    def _subset(self,stimInd):
        # select stimuli along the within-category axis (applied to every category)
        stimInd=np.atleast_1d(np.asarray(stimInd,dtype=int))
        out=copy.copy(self)
        out.val=self.val[...,stimInd,:]
        out.weights=self.weights[stimInd,:]
        out.yCtg=self.yCtg[stimInd]
        out.yCtgInd=self.yCtgInd[stimInd,:]
        out.nStim_Ctg=len(stimInd)
        out.nStim=out.nStim_Ctg*out.nCtg
        return out

    def __init__(self,x,stimuli,yCtgInd,Y,bStimIsFourier=False,nSplit=0,bStimIsSplit=False,bContrastNormalize=False,
                 Yperiod=None,y=None):
        """
        stimuli [ *dims x nStim ], contrast normalized (zero mean, unit norm; a warning is given otherwise)
        yCtgInd [ nStim ] category label of each stimulus (any integer coding, e.g. 1-based from matlab)
        Y       [ nCtg ] or [ nCtg x nDim ] latent value(s) of each category, in sorted label order. Several latent
                dimensions (e.g. disparity and speed) are estimated jointly; the categories are their combinations
        Yperiod - None (linear), or the period of circular latent variables (e.g. 180 for orientation in degrees, 2*pi
                  for phase): one value for every dimension, or one per dimension with None for the linear ones.
                  Distances between latent values (errors, estimates, targets, category pooling) wrap around it
        bContrastNormalize - contrast normalize the stimuli (spatial domain only)
        y       - None, or [ nStim ] / [ nStim x nDim ] the latent value of each stimulus, for a continuous latent variable
                  whose categories are bins (see Stim.binned): Y is then each bin's representative level. Errors
                  ('l1', 'l2'), gaussian divergence targets (targetSigma) and performance measure against each stimulus's
                  own value; Model bWithin estimates within the bins
        """

        self.bIsFourier=bStimIsFourier
        self.x=x
        self.nSplit=nSplit
        self.bIsSplit=bStimIsSplit

        stimuli=np.asarray(stimuli)
        yCtgInd=np.asarray(yCtgInd).ravel()
        self.ctg,yCtgInd=np.unique(yCtgInd,return_inverse=True) # relabel to 0..nCtg-1
        self.nCtg=len(self.ctg)

        Y=np.asarray(Y,dtype=float)
        if Y.ndim==2 and 1 in Y.shape:
            Y=Y.ravel()                                                    # a row or column vector (e.g. from matlab)
        if Y.ndim not in (1,2):
            raise Exception('Y must be [ nCtg ] or [ nCtg x nDim ]')
        if len(Y)!=self.nCtg:
            raise Exception('Y has ' + str(len(Y)) + ' values but there are ' + str(self.nCtg) + ' categories')
        if len(np.unique(Y,axis=0))!=self.nCtg:
            raise Exception('categories must have distinct latent values Y')
        self.Y=jnp.asarray(Y)
        self.nDim=1 if Y.ndim==1 else Y.shape[1]
        self.Yperiod=Stim._parse_period(Yperiod,self.nDim)
        if stimuli.shape[-1]!=len(yCtgInd):
            raise Exception('last dimension of stimuli must equal the number of labels')
        self.bContinuous=y is not None
        if self.bContinuous:
            y=np.asarray(y,dtype=float)
            if self.nDim==1 and y.ndim==2 and 1 in y.shape:
                y=y.ravel()
            if y.shape!=(len(yCtgInd),)+Y.shape[1:]:
                raise Exception('y must be [ nStim ]' + ('' if self.nDim==1 else ' x ' + str(self.nDim)) + ', one latent value per stimulus')
            if not np.all(np.isfinite(y)):
                raise Exception('y must be finite')

        self.dims=tuple(stimuli.shape[:-1])
        self.ndim=len(self.dims)
        self.nPix=int(np.prod(self.dims))

        # regroup by category, padding to the largest category; weights mask the padding
        nStimCtg=np.bincount(yCtgInd,minlength=self.nCtg)
        self.nStim_Ctg=int(np.max(nStimCtg))

        stimuli=np.reshape(stimuli,(self.nPix,len(yCtgInd)))
        if bContrastNormalize:
            if bStimIsFourier:
                raise Exception('contrast normalize stimuli in the spatial domain')
            stimuli=contrast_normalize(stimuli)
        elif not bStimIsFourier:
            _warn_if_not_contrast_normalized(stimuli)
        val=np.zeros((self.nPix,self.nStim_Ctg,self.nCtg),dtype=stimuli.dtype)
        weights=np.zeros((self.nStim_Ctg,self.nCtg))
        yctg=np.zeros((self.nStim_Ctg,self.nCtg)+Y.shape[1:])
        for c in range(self.nCtg):
            val[:,:nStimCtg[c],c]=stimuli[:,yCtgInd==c]
            weights[:nStimCtg[c],c]=1
            yctg[:,c]=Y[c]                                                       # padding keeps the level
            if self.bContinuous:
                yctg[:nStimCtg[c],c]=y[yCtgInd==c]

        self.val=jnp.array(val)
        self.weights=jnp.array(weights)
        self.yCtg=jnp.array(yctg)
        self.yCtgInd=jnp.array(np.broadcast_to(np.arange(self.nCtg),(self.nStim_Ctg,self.nCtg)))
        self.nStim=self.nStim_Ctg*self.nCtg

        if self.bIsSplit:
            self.bIsSplit=False
            self.split()

    @staticmethod
    def _parse_period(Yperiod,nDim):
        # None, or a tuple of one period (float) or None per latent dimension
        if Yperiod is None:
            return None
        per=list(Yperiod) if isinstance(Yperiod,(list,tuple,np.ndarray)) else [Yperiod]*nDim
        if len(per)!=nDim:
            raise Exception('Yperiod needs one value per latent dimension (' + str(nDim) + ')')
        per=tuple(None if q is None else float(q) for q in per)
        if any(q is not None and not q>0 for q in per):
            raise Exception('periods in Yperiod must be positive (or None for linear dimensions)')
        return None if all(q is None for q in per) else per

    #- held-out data
    def _valid_indices(self):
        w=np.asarray(self.weights)>0
        return [np.flatnonzero(w[:,c]) for c in range(self.nCtg)]

    def _take(self,inds):
        # stimuli by within-category index (one index array per category), regrouped and zero padded as in __init__
        mMax=max(len(i) for i in inds)
        idx=np.zeros((mMax,self.nCtg),dtype=int)
        w=np.zeros((mMax,self.nCtg))
        for c,i in enumerate(inds):
            idx[:len(i),c]=i
            w[:len(i),c]=1
        out=copy.copy(self)
        idx=jnp.asarray(idx)
        wj=jnp.asarray(w,dtype=jnp.asarray(self.weights).dtype)
        out.val=jnp.take_along_axis(self.val,jnp.broadcast_to(idx,self.val.shape[:-2]+idx.shape),axis=-2)*wj.astype(self.val.dtype)
        out.weights=wj
        out.yCtg=_take_stim(jnp.asarray(self.yCtg),idx)
        out.yCtgInd=jnp.take_along_axis(jnp.asarray(self.yCtgInd),idx,axis=0)
        out.nStim_Ctg=mMax
        out.nStim=mMax*self.nCtg
        return out

    def train_test(self,testFraction=0.2,seed=0):
        """random (train, test) split, stratified by category; each category keeps at least one stimulus in each part"""
        rng=np.random.default_rng(seed)
        train,test=[],[]
        for i in self._valid_indices():
            i=rng.permutation(i)
            nTest=int(np.clip(np.round(testFraction*len(i)),1,len(i)-1))
            test.append(np.sort(i[:nTest]))
            train.append(np.sort(i[nTest:]))
        return self._take(train),self._take(test)

    def folds(self,k=5,seed=0):
        """k cross-validation folds [(train, test), ...], stratified by category; every stimulus is tested once"""
        rng=np.random.default_rng(seed)
        parts=[np.array_split(rng.permutation(i),k) for i in self._valid_indices()]
        out=[]
        for f in range(k):
            test=[np.sort(p[f]) for p in parts]
            train=[np.sort(np.concatenate([p[g] for g in range(k) if g!=f])) for p in parts]
            out.append((self._take(train),self._take(test)))
        return out

    @property
    def _pix_dims(self):
        # spatial dims of what is currently stored in the pixel axes
        if self.bIsSplit:
            return (self.dims[0]//self.nSplit,) + tuple(self.dims[1:])
        return self.dims

    def _ft_reshape(self):
        # the first axis holds the flattened pixels (of each sub-stimulus when split)
        return self._pix_dims + self.val.shape[1:], tuple(range(len(self._pix_dims)))

    def _fft(self):
        if self.bIsFourier:
            raise Exception('Stim is already in the fourier domain')
        shape,axes=self._ft_reshape()
        v=jnp.reshape(self.val,shape)
        v=jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(v,axes=axes),axes=axes,norm='ortho'),axes=axes)
        self.val=jnp.reshape(v,self.val.shape)
        self.bIsFourier=True

    def _ifft(self):
        if not self.bIsFourier:
            raise Exception('Stim is already out of the fourier domain')
        shape,axes=self._ft_reshape()
        v=jnp.reshape(self.val,shape)
        v=jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(v,axes=axes),axes=axes,norm='ortho'),axes=axes)
        self.val=jnp.reshape(v,self.val.shape)
        self.bIsFourier=False

    def _finalize(self,dtype,index,bFourier,bSplit):
        # canonical order: split (spatial) then fourier, so each sub-stimulus is transformed on its own
        if self.bIsFourier and (bSplit!=self.bIsSplit or not bFourier):
            self._ifft()

        if bSplit and not self.bIsSplit:
            self.split()
        elif not bSplit and self.bIsSplit:
            self.unsplit()

        if bFourier and not self.bIsFourier:
            self._fft()

        if not bFourier and jnp.iscomplexobj(self.val):
            self.val=self.val.real

        self.val=jnp.asarray(self.val,dtype=dtype)

        # weights and latent values in the matching real precision, so float32 learning stays float32 under jax x64
        rdtype=jnp.finfo(dtype).dtype
        self.weights=jnp.asarray(self.weights,dtype=rdtype)
        self.yCtg=jnp.asarray(self.yCtg,dtype=rdtype)
        self.Y=jnp.asarray(self.Y,dtype=rdtype)

        if index is not None:
            return self._subset(index)
        else:
            return self

    def split(self):
        #[ nPix  x nStim_Ctg x nCtg ]    (nPix / nSplit) x nSplit x nStim_Ctg x nCtg
        if self.nSplit is None or self.nSplit < 2:
            raise Exception('nSplit must be set (>=2) in Stim to split')
        if self.dims[0] % self.nSplit != 0:
            raise Exception('first stimulus dimension (' + str(self.dims[0]) + ') is not divisible by nSplit')
        # sub-stimuli are contiguous blocks of the first dimension (e.g. left then right eye)
        self.val=jnp.swapaxes(jnp.reshape(self.val,(self.nSplit, self.nPix//self.nSplit, self.nStim_Ctg, self.nCtg)),0,1)
        self.bIsSplit=True

    def unsplit(self):
        self.val=jnp.reshape(jnp.swapaxes(self.val,0,1),(self.nPix, self.nStim_Ctg, self.nCtg))
        self.bIsSplit=False

    def plot_all(self):
        plt.imshow(jnp.reshape(self.val,(self.nPix, self.nStim )))

    def plot(self,index=(0,0),bFourier=False):
        """plot stimulus index=(stimulus within category, category) in the spatial or fourier domain"""
        stim=np.asarray(self.val[...,index[0],index[1]])               # [ nPix x (nSplit) ]
        parts=[stim[:,i] for i in range(self.nSplit)] if self.bIsSplit else [stim]
        axes=tuple(range(self.ndim))
        for i,s in enumerate(parts):
            s=np.reshape(s,self._pix_dims)
            if bFourier and not self.bIsFourier:
                s=np.asarray(_centered_ft(s,axes))
            elif not bFourier and self.bIsFourier:
                s=np.real(np.asarray(_centered_ft(s,axes,inverse=True)))
            if len(parts)>1:
                plt.subplot(1,len(parts),i+1)
            _plot_signal(s,bFourier)

    @staticmethod
    def load(fname,dims=None,keys=None,**kw):
        """
        stimuli from a burgelab-style .mat file (github.com/burgelab/AMA) or a numpy .npz file with the same variables:
            s       [ nPix x nStim ] stimuli (the pixels of each stimulus flattened, e.g. column-major from matlab)
            ctgInd  [ nStim ] category labels,  X [ nCtg ] (or [ nCtg x nDim ]) the level of each category
        or, for a continuous latent variable, instead of ctgInd and X:
            y       [ nStim ] (or [ nStim x nDim ]) each stimulus's latent value, binned by Stim.binned (nBins, binType,
                    edges, like in kw)
        dims  - the shape of each stimulus (e.g. (32, 32) for images), None for 1-D; matlab (column-major) pixel order
                when loading a .mat file
        keys  - other variable names, e.g. {'s': 'stimuli', 'ctgInd': 'labels'}
        other keyword arguments go to Stim (Yperiod, nSplit, bContrastNormalize, ...), or Stim.binned
        """
        fname=str(fname)
        if fname.endswith('.mat'):
            D=loadmat(fname)
        elif fname.endswith('.npz'):
            D=dict(np.load(fname))
        else:
            raise Exception('Stim.load reads .mat and .npz files')
        k={'s':'s','ctgInd':'ctgInd','X':'X','y':'y'}
        k.update(keys or {})
        if k['s'] not in D:
            raise Exception("no stimuli ('" + k['s'] + "') in " + fname)
        s=np.asarray(D[k['s']])
        nStim=s.shape[-1]
        if dims is None:
            dims=(int(np.prod(s.shape[:-1])),)
        dims=tuple(int(d) for d in np.atleast_1d(dims))
        if int(np.prod(dims))!=int(np.prod(s.shape[:-1])):
            raise Exception('dims ' + str(dims) + ' do not match the ' + str(int(np.prod(s.shape[:-1]))) + ' pixels per stimulus')
        s=np.reshape(np.reshape(s,(-1,nStim)),dims+(nStim,),order='F' if fname.endswith('.mat') else 'C')
        x=filt.X(ndim=1,n=dims[0],totS=1) if len(dims)==1 else filt.X(n=dims,totS=1)
        if k['ctgInd'] in D and k['X'] in D:
            return Stim(x,s,D[k['ctgInd']],D[k['X']],**kw)
        if k['y'] in D:
            return Stim.binned(x,s,np.squeeze(np.asarray(D[k['y']],dtype=float)),**kw)
        raise Exception("no labels ('" + k['ctgInd'] + "' and '" + k['X'] + "', or '" + k['y'] + "') in " + fname)


    @classmethod
    def binned(cls,x,stimuli,y,nBins=10,binType='quantile',edges=None,like=None,Yperiod=None,**kw):
        """
        stimuli with a continuous latent variable y [ nStim ] or [ nStim x nDim ], grouped into bins that serve as the
        categories (see Stim y). Each bin's level Y is the mean of its stimuli's values (the circular mean on circular
        dimensions). Empty bins are dropped.
        nBins   - bins per latent dimension (a number, or one per dimension)
        binType - 'quantile' (equal counts) or 'uniform' (equal widths over the range of y; the only choice on circular
                  dimensions, where the bins tile the period centered at 0, P/nBins, ...)
        edges   - bin edges instead (an array, or one per dimension); values beyond them go to the outer bins
        like    - another binned Stim (e.g. the training set) whose edges and levels Y to use, so that the two can be
                  evaluated together (Unit.evaluate); it must have stimuli in the same bins
        other keyword arguments go to Stim (bContrastNormalize, nSplit, ...); the edges are kept in binEdges
        """
        y=np.asarray(y,dtype=float)
        yv=y.reshape(len(y),-1)
        nDim=yv.shape[1]
        per=Stim._parse_period(Yperiod,nDim)
        pd=per if per is not None else (None,)*nDim
        if like is not None:
            edges=like.binEdges
        elif edges is not None:
            edges=[np.asarray(edges,dtype=float)] if nDim==1 and np.ndim(edges[0])==0 else [np.asarray(e,dtype=float) for e in edges]
        else:
            nb=list(np.broadcast_to(np.asarray(nBins,dtype=int),(nDim,)))
            if binType not in ('quantile','uniform'):
                raise Exception("binType must be 'quantile' or 'uniform'")
            edges=[]
            for d in range(nDim):
                if pd[d] is not None:
                    edges.append((np.arange(nb[d]+1)-0.5)*pd[d]/nb[d])
                elif binType=='quantile':
                    edges.append(np.unique(np.quantile(yv[:,d],np.linspace(0,1,nb[d]+1))))
                else:
                    edges.append(np.linspace(yv[:,d].min(),yv[:,d].max(),nb[d]+1))
        if len(edges)!=nDim or any(len(e)<2 for e in edges):
            raise Exception('edges needs at least two edges for each of the ' + str(nDim) + ' latent dimensions')
        pos=[]
        for d,e in enumerate(edges):
            v=yv[:,d]
            if pd[d] is not None:
                v=e[0]+np.mod(v-e[0],pd[d])                                     # into the period the edges tile
            pos.append(np.clip(np.searchsorted(e,v,side='right')-1,0,len(e)-2))
        lab=np.ravel_multi_index(pos,tuple(len(e)-1 for e in edges))
        if like is not None:
            if not np.array_equal(np.unique(lab),np.asarray(like.ctg)):
                raise Exception('the stimuli must fall in the same bins as those of like')
            Ylev=np.asarray(like.Y)
        else:
            ctg=np.unique(lab)
            Ylev=np.zeros((len(ctg),nDim))
            for c,l in enumerate(ctg):
                for d in range(nDim):
                    v=yv[lab==l,d]
                    if pd[d] is None:
                        Ylev[c,d]=v.mean()
                    else:
                        Ylev[c,d]=np.angle(np.mean(np.exp(2j*np.pi*v/pd[d])))*pd[d]/(2*np.pi)
            if nDim==1:
                Ylev=Ylev[:,0]
        out=cls(x,stimuli,lab,Ylev,Yperiod=Yperiod,y=y,**kw)
        out.binEdges=[np.asarray(e) for e in edges]
        return out

    @classmethod
    def gen_test(cls,dims=(8,9),nStim=101):
        a=random.randn(*(dims+(int(np.ceil(nStim/2)),)))
        b=random.rand( *(dims+(int(np.floor(nStim/2)),)))
        stimuli=contrast_normalize(np.concatenate((a,b),len(dims)))
        yCtgInd=np.concatenate((np.zeros(a.shape[-1],dtype=int),np.ones(b.shape[-1],dtype=int)),0)
        X=np.array([1, 2])

        x=filt.X(n=dims,totS=1)
        return Stim(x,stimuli,yCtgInd,X,False)


class _Index():
    _parent=None
    n=_ParentProp()
    pix_dims=_ParentProp()
    nSplit=_ParentProp()
    bSplit=_ParentProp()

    def __init__(self,parent,ind_lrn,ind_fix,ind_rec,bAnalytic=True):
        self._parent=parent

        #- parse
        ind_lrn=_Index.__parse_ind(ind_lrn)
        ind_fix=_Index.__parse_ind(ind_fix)
        ind_rec=_Index.__parse_ind(ind_rec)

        if len(ind_rec)==0 and len(ind_lrn)==0:
            raise Exception('a lrn or rec index is required')

        #- No overlap
        comb=np.concatenate((ind_lrn,ind_fix,ind_rec))
        if len(np.unique(comb))!=len(comb):
            raise Exception('filter indices need to be mutually exclusive')

        if np.any((comb < 0) | (comb >= self.n)):
            raise Exception('indeces contain value outside of range (n=' + str(self.n) + ')')

        if len(comb)!=self.n:
            raise Exception('every filter index must be either learned, fixed, or recursed')

        self.ind_rec=ind_rec # old and learning
        self.ind_lrn=ind_lrn # new and learning
        self.ind_fix=ind_fix # old and helping

        self.bAnalytic=bAnalytic

    @staticmethod
    def __parse_ind(ind):
        return np.array(ind,dtype=int,ndmin=1).ravel()

    @property
    def shape_exp(self):
        return self._parent._shape_exp

    #- prepped
    @property
    def recover(self):
        return np.union1d(self.ind_fix,self.ind_rec).astype(int) # to get from f_out


    #- in
    @property
    def pix(self):
        """flat indices of the learned pixels, or frequencies for fourier-domain filters"""
        if not self._parent.bIsFourier:
            return np.arange(int(np.prod(self.pix_dims)))
        # half-space: the first nonzero frequency coordinate is positive. The other half is the complex conjugate,
        # so the implied spatial filters are real. Excludes DC (no hilbert pair) and frequencies whose first nonzero
        # coordinate is at nyquist (their own conjugate).
        grids=np.meshgrid(*[_centered_freqs(d) for d in self.pix_dims],indexing='ij')
        keep=np.zeros(self.pix_dims,dtype=bool)
        undecided=np.ones(self.pix_dims,dtype=bool)
        for g in grids:
            keep|=undecided & (g>0)
            undecided&=(g==0)
        return np.flatnonzero(keep)

    @property
    def _ix(self):
        # [ pix x (nSplit) x filters ] open mesh
        lead=[np.arange(self.nSplit)] if self.bSplit else []
        return np.ix_(self.pix,*lead,np.union1d(self.ind_lrn,self.ind_rec).astype(int))

    @property
    def insert(self):
    # jaxxed: index into [ nPix x (nSplit) x nF ]
        return self._ix

    @property
    def insert_exp(self):
    # jaxxed: index into [ *pix_dims x (nSplit) x nF ]
        ix=self._ix
        return tuple(np.unravel_index(ix[0],self.pix_dims)) + tuple(ix[1:])

class Filter():
    index=None
    _insert_index_jx=None
    def __init__(self):
        self.last=None
        self.out=None

    def _finalize(self,stim,dtype,n,ind_lrn,ind_fix,ind_rec=(),bAnalytic=True,last=None,bSplit=False):
        #[ nPix x nF ]                   (nPix / nSplit) x nSplit x nF

        self.x=stim.x
        self.bIsFourier=stim.bIsFourier
        self.nSplit=stim.nSplit
        self.bSplit=bSplit

        self.dtype=dtype
        self.dims=stim.dims
        self.ndim=len(self.dims)

        self.n=n

        if bSplit:
            self.pix_dims=(self.dims[0]//self.nSplit,) + tuple(self.dims[1:])
            self._shape=(int(np.prod(self.pix_dims)),self.nSplit,self.n)
            self._shape_exp=self.pix_dims + (self.nSplit,self.n,)
        else:
            self.pix_dims=tuple(self.dims)
            self._shape=(int(stim.nPix),self.n)
            self._shape_exp=self.pix_dims + (self.n,)

        if last is not None:
            self.last=last

        self.bNew=self.last is None or np.size(self.last)==0

        if self.bNew and len(ind_rec) !=0:
            raise Exception('ind_rec cannot be set if there are no previous filter')

        self.index=_Index(self,ind_lrn,ind_fix,ind_rec,bAnalytic)

        # out
        if not self.bNew:
            self.out=np.asarray(self.last)
            if self.out.shape[:-1]!=self._shape_exp[:-1]:
                self.out=np.reshape(self.out,self._shape_exp[:-1] + (-1,))

            # extend
            lshape=np.shape(self.out)[-1]
            if self.n > lshape:
                nshape=self._shape_exp[:-1] + (self.n-lshape,)
                self.out=np.concatenate((self.out, np.full(nshape,np.nan,dtype=self.out.dtype)), axis=-1)
            elif self.n < lshape:
                raise Exception('specified n is smaller than shape of last: this should not happen!')

        else:
            self.out=np.full(self._shape_exp,np.nan,dtype=dtype)

        # prepped
        prepped=np.zeros(self._shape_exp,dtype=dtype)
        prepped[...,self.index.recover]=self.out[...,self.index.recover]
        self.prepped_jx=jnp.array(np.reshape(prepped,self._shape),dtype=dtype)
        self.prepped_exp_jx=jnp.array(prepped,dtype=dtype)

        # save insert index as a constant
        self._insert_index_jx=self.index.insert
        self._insert_index_exp_jx=self.index.insert_exp

    def load_in(self,f,bIsFourier):
        self.bIsFourier=bIsFourier
        self.out=f

    def get_f0(self,rng_key,rand_fun):
        # jaxxed
        if jnp.issubdtype(self.dtype,jnp.complexfloating):
            f00=jnp.array(rand_fun[0](rng_key,*rand_fun[1:],shape=self._shape,dtype=jnp.float32),self.dtype)
        else:
            f00=rand_fun[0](rng_key,*rand_fun[1:],shape=self._shape,dtype=self.dtype)

        # samplers like jax.random.ball append a trailing dimension
        if f00.ndim > len(self._shape):
            f00=f00[...,0]

        return f00[self.index.insert]

    def insert(self,fIn):
    # jaxxed
        return self.prepped_jx.at[self._insert_index_jx].set(fIn)

    def extract(self,fOut):
    # jaxxed
        self.out=jnp.reshape(self.insert(fOut),self._shape_exp)

    @property
    def out_flat(self):
        return jnp.reshape(jnp.asarray(self.out),self._shape)

    def implied_spatial(self,f=None):
        """
        spatial filters [ *pix_dims x (nSplit) x nF ] implied by f (default: out). For fourier-domain filters,
        fourierType=1 gives real filters; fourierType=2 gives complex filters whose real part is the filter and whose
        imaginary part is its quadrature pair (the two response components are the stimulus dot products with each).
        """
        f=np.asarray(self.out if f is None else f)
        if f.shape==self._shape:
            f=np.reshape(f,self._shape_exp)
        if not self.bIsFourier:
            return f
        g=np.asarray(_centered_ft(np.sqrt(2)*f,tuple(range(len(self.pix_dims))),inverse=True))
        return np.conj(g) if self.index.bAnalytic else np.real(g)

#- PLOT

    def plot_out(self,bFourier=None,name='f_out'):
        self._plot(self.out,bFourier=bFourier,name=name)

    def plot_last(self,bFourier=None,name='f_last'):
        if self.last is None:
            raise Exception('there are no previous filters (last is set by train_recurse and train_append)')
        self._plot(self.last,bFourier=bFourier,name=name)

    def plot_fprepped(self,name='f_prepped'):
        # parameters as stored, in the learning domain
        self._plot(self.prepped_jx,bFourier=self.bIsFourier,name=name,bConvert=False)

    def plot_indices(self,name='f_indices'):
        # learned (2), fixed (0), and not learned (-1) parameters, in the learning domain
        A=np.ones(self._shape)*-1
        A[self._insert_index_jx]=2
        if len(self.index.ind_fix)>0:
            A[...,self.index.ind_fix]=0
        self._plot(A,bFourier=self.bIsFourier,name=name,bConvert=False)

    def _plot(self,y,bFourier=None,name='',bConvert=True):
        y=np.asarray(y)
        if y.shape==self._shape:
            y=np.reshape(y,self._shape_exp)
        if bFourier is None:
            bFourier=self.bIsFourier
        if bConvert:
            if self.bIsFourier and not bFourier:
                y=self.implied_spatial(y)
            elif bFourier and not self.bIsFourier:
                y=np.asarray(_centered_ft(y,tuple(range(len(self.pix_dims)))))
        if not self.bSplit:
            y=y[...,None,:]                                               # [ *pix_dims x 1 x nF ]

        for i in range(y.shape[-1]):
            plt.figure(name + '_' + str(i))
            for k in range(y.shape[-2]):
                if y.shape[-2]>1:
                    plt.subplot(1,y.shape[-2],k+1)
                _plot_signal(y[...,k,i],bFourier)


class Nrn(_Static):
    """
    Response model (Burge & Jaini 2017, Eq 1)
        r      = rmax * f^H s                    mean response
        sigma2 = fano * |r| + var0               scaled additive noise
        R      = r + eta,  eta ~ N(0, sigma2)
    optionally with an activation, normalization, and noise before (1) and/or after (2) normalization.

    The likelihood (Model) uses the noise variance of the final responses (Nrn._likelihood_variance):
      stage 2 (after normalization): fano*|R| + var0; also used when no noise is sampled (responseType='mean', the
                                     mean-response approximation of Burge & Jaini 2017)
      stage 1 (bNoise_1, before normalization): fano*|r| + var0 carried through the normalization, exactly for 'broad' and
                                     'narrow' and to first order for 'gen' (which underestimates, by ~5-10% when noise is
                                     ~10% of the pooled response); with both stages on, the variances add, with
                                     stage 2's scaled by the expected |response| given stage-1 noise
    rho correlates the noise of the response dimensions: a number correlates all of them (filters, sub-filters,
    real/imaginary components) equally; a correlation matrix [ nDim x nDim ] specifies each pair, over the flattened
    response dimensions in the order of _flatten_responses (real parts of all filters, sub-filters within each filter,
    then the imaginary parts), in noise sampling and in the likelihood alike. None: no noise.
    """
    _noise_1_fun=_id
    _noise_2_fun=_id
    _normalize_fun=_id
    _activation_fun=_id
    _corr_fun=_id
    _average_fun=_id
    _whiten_fun=_id
    _W=None                      # frozen whitening matrix (Unit.freeze_whitening)
    bNoise_1=_TypeFunc(True)
    bNoise_2=_TypeFunc(True)
    activationType=_TypeFunc()
    normalizeType=_TypeFunc()
    corrType=_TypeFunc()
    averageType=_TypeFunc()
    whitenType=_TypeFunc()
    readoutType=_TypeFunc()
    _readout_fun=_id
    def __init__(self,fano=1.36,var0=0.23,rmax=5.7,normalizeType='None',activationType='None',bNoise_1=False,bNoise_2=False,rho=0,eps=0.001,averageType='full',nSamples=1,
                 whitenType='None',whitenMethod='zca',whitenEps=1e-5,readoutType='None',bBias=False,bSplitNegatives=False,
                 rsat=None,c50=None,nNR=2.,softmaxT=1.,normPool=None,bLearnNormPool=False,nReadout=None,readoutActivation='None',
                 respBudget=None):
        """
        activationType - response nonlinearity, on real and imaginary components separately (responses are in units of
                       rmax): 'relu', 'softplus', 'abs', 'logistic', 'swish', 'swish2', 'tanh', 'gauss', 'igauss', and
                       'ramp'        - min(max(r,0), rsat): rectified, saturating at rsat (default rmax/2)
                       'nakarushton' - rmax r^nNR / (r^nNR + c50^nNR) for r > 0 (Naka-Rushton / hyperbolic ratio; c50
                                       default rmax/4)
                       'softmax'     - rmax softmax(r/softmaxT) over the filters (and sub-filters): a population
                                       nonlinearity whose responses sum to rmax (real responses only)
        bBias        - learn an offset per response dimension, added to the linear response before the activation (it
                       matters with an activation, normalization, or through the response-dependent noise variance)
        bSplitNegatives - ON/OFF channels: each response dimension r becomes two rectified ones, max(r,0) and max(-r,0)
                       (on real and imaginary components separately), after the activation: firing rates are not
                       negative, and each channel has its own noise (fano*|r| + var0)
        normPool     - normalizeType 'gen': [ nChan x nChan ] weights of the normalization pool, D_i = eps + sum_j
                       normPool_ij |r_j| (summed over sub-filters and components), with nChan the response dimensions
                       entering normalization (filters, x2 with bSplitNegatives; flattened dimensions when whitened).
                       None: every channel normalizes every other (the plain 'gen' normalization)
        bLearnNormPool - learn the normalization pool weights (nonnegative, softplus of Unit parameters, starting at
                       normPool or at all ones). Only their relative sizes are learned: each row keeps the total weight
                       of the initial pool (with fixed additive noise, a smaller denominator would otherwise always help)
        readoutType 'linear' - a second layer: nReadout output neurons (default: the number of filters), each a learned
                       unit-norm linear combination of all the (normalized) responses, on real and imaginary components
                       separately, followed by readoutActivation (any activationType) and stage-2 noise. With bBias, the
                       output neurons have learned offsets too.
                       nReadout as a tuple of widths, e.g. (8, 4), stacks that many layers, each a unit-norm linear
                       combination of the previous layer followed by readoutActivation (with bBias, offsets before it);
                       stage-2 noise is on the last. Without an activation, the stack is one linear map.
        whitenType   - population whitening of the mean responses, before noise (intended for full AMA, whose likelihood
                       assumes noise that is independent across response dimensions)
                       'None'
                       'gram'     - by the Gram matrix of the real filters behind the response dimensions: equivalent to
                                    orthonormalizing the filters (does not depend on the stimuli)
                       'response' - by the covariance of the mean responses over the (valid) stimuli: decorrelates the responses
        whitenMethod - 'zca' (symmetric) or 'chol' (Gram-Schmidt order: earlier filters are not changed by later ones).
                       Both keep each dimension's variance (or filter norm), so rmax, fano and var0 keep their meaning.
        whitenEps    - ridge added to the matrix being whitened, relative to its mean diagonal
        normalizeType 'phase' - unit phasors r/(|r|+eps): contrast-invariant responses (phase congruency inputs)
        respBudget   - a response budget: each final response dimension (after the readout, before stage-2 noise) is
                       scaled so that its root mean square over the training stimuli is respBudget. Its signal-to-noise
                       ratio against the fixed additive noise (var0) then can not grow with the size of the responses,
                       only with their shape: learned normalization pools (bLearnNormPool) and second layers can not
                       shrink denominators or grow weights to beat the noise. Stage-1 noise carried through is scaled
                       likewise. Held-out stimuli are scaled by the training stimuli's gains (as whitening; see
                       Unit.evaluate); with Optimizer batchSize, by each batch's.
        readoutType  - pooling of the (normalized) responses across filters, with learned nonnegative weights p
                       (softplus of the parameters Unit.pool_p, normalized to sum 1), before stage-2 noise:
                       'None'
                       'resultant'      - the filter responses and their weighted resultant sum_j p_j R_j
                       'resultant_only' - only the weighted resultant (weighted phase congruency with 'phase')
        """
        # rmax=5.7, var0=0.23 as in burgelab/AMA (paramRSP of AMAdataDisparity.mat)
        # python floats: numpy scalars (e.g. read from a .mat file) would promote float32 learning to float64 under jax x64
        self.fano=float(fano)
        self.var0=float(var0)
        self.rmax=float(rmax)
        self.eps=float(eps)

        self.bNoise_1=bNoise_1
        self.bNoise_2=bNoise_2
        self.normalizeType=normalizeType
        self.activationType=activationType
        self.averageType=averageType
        self.nSamples=nSamples
        self.whitenType=whitenType
        self.whitenMethod=whitenMethod
        self.whitenEps=whitenEps
        self.readoutType=readoutType
        self.bBias=bool(bBias)
        self.bSplitNegatives=bool(bSplitNegatives)
        self.rsat=None if rsat is None else float(rsat)
        self.c50=None if c50 is None else float(c50)
        self.nNR=float(nNR)
        self.softmaxT=float(softmaxT)
        self.normPool=None if normPool is None else np.array(normPool,dtype=float)
        self.bLearnNormPool=bool(bLearnNormPool)
        self.nReadout=None if nReadout is None else (int(nReadout) if np.ndim(nReadout)==0 else tuple(int(n) for n in nReadout))
        self.readoutActivation=readoutActivation
        self.respBudget=None if respBudget is None else float(respBudget)
        if self.respBudget is not None and not self.respBudget>0:
            raise Exception('respBudget must be positive (or None)')
        if not hasattr(Nrn,'_activation__'+str(readoutActivation).lower()):
            raise Exception('readoutActivation must be an activationType (e.g. None, relu, ramp, softmax)')
        if self.normPool is not None and (self.normPool.ndim!=2 or self.normPool.shape[0]!=self.normPool.shape[1]
                                          or np.any(self.normPool<0)):
            raise Exception('normPool must be a square matrix of nonnegative weights')
        if (self.rsat is not None and not self.rsat>0) or (self.c50 is not None and not self.c50>0) or not self.nNR>0 \
                or not self.softmaxT>0:
            raise Exception('rsat, c50, nNR and softmaxT must be positive')
        if self.nReadout is not None and (np.min(self.nReadout)<1 or np.size(self.nReadout)<1):
            raise Exception('nReadout must be at least 1 (or a tuple of layer widths of at least 1)')

        self.rho=rho

        self.filter=Filter()
        self.bFinalized=False

    @property
    def bAnalytic(self):
        if hasattr(self,'filter') and hasattr(self.filter,'index') and hasattr(self.filter.index,'bAnalytic'):
            return self.filter.index.bAnalytic
        else:
            return None

    def _key(self):
        return (self.fano,self.var0,self.rmax,self.eps,self.nSamples,_freeze(self.rho),
                self.bNoise_1,self.bNoise_2,self.activationType,self.normalizeType,self.corrType,self.averageType,
                self.whitenType,self.whitenMethod,self.whitenEps,self.readoutType,getattr(self.filter,'pix_dims',None),
                getattr(self,'bFourier',None),getattr(self,'bSplit',None),self.bAnalytic,str(getattr(self,'dtype',None)),
                self.bBias,self.bSplitNegatives,self.rsat,self.c50,self.nNR,self.softmaxT,_freeze(self.normPool),
                self.bLearnNormPool,self.nReadout,str(self.readoutActivation).lower(),getattr(self.filter,'n',None),self.respBudget)

    def copy(self):
       return Nrn(**_get_copy_dict(self,_NRN_EXCL))

    def _finalize(self,stim,dtype,n,ind_lrn,ind_fix,ind_rec,bFourier=False,bAnalytic=False,bSplit=False,last=None):
        if self.normalizeType=='narrow' and not bFourier:
            raise Exception("normalizeType='narrow' requires learning in the fourier domain (fourierType>=1)")

        self.dtype=dtype
        self.bFourier=bFourier
        self.bSplit=bSplit

        self.filter._finalize(stim,dtype,n,ind_lrn,ind_fix,ind_rec,bAnalytic=bAnalytic,last=last,bSplit=bSplit)

        self.bFinalized=True

    @staticmethod
    @jit
    def insert(fIn,f_prepped,index):
        return f_prepped.at[index].set(jnp.asarray(fIn,dtype=f_prepped.dtype))

    @partial(jit, static_argnames=['self'])
    def lrn_main(self,rng,stim,fIn,prepped,index,weights=None,p=None):
        # prepped and index are arguments rather than read from self.filter, so traces survive re-finalizing
        return self.main(rng,stim,self.insert(fIn,prepped,index),weights,None,p)

    @property
    def bReadout(self):
        return str(self.readoutType).lower()!='none'

    @property
    def bPooledReadout(self):
        return str(self.readoutType).lower() in ('resultant','resultant_only')

    @property
    def bParams(self):
        # whether the response model has learned parameters (Unit._p)
        return self.bReadout or self.bBias or self.bLearnNormPool

    #- response dimensions (after finalize)
    @property
    def bComplexOut(self):
        # complex responses (quadrature pairs, not whitened)
        return bool(self.bAnalytic) and not self._bWhiten

    @property
    def nIn(self):
        # rows (axis 0) of the responses entering the activation: filters, or flattened dimensions when whitened
        n=self.filter.n
        if self._bWhiten:
            return n*(2 if self.bAnalytic else 1)*(self.filter.nSplit if self.bSplit else 1)
        return n

    @property
    def nChan(self):
        # rows entering normalization and the readout
        return self.nIn*(2 if self.bSplitNegatives else 1)

    @property
    def nSub(self):
        # sub-filters per row (split parts, not whitened)
        return self.filter.nSplit if (self.bSplit and not self._bWhiten) else 1

    @property
    def _readoutWidths(self):
        # the widths of the layers of the readout 'linear'
        if self.nReadout is None:
            return (self.nChan,)
        return self.nReadout if isinstance(self.nReadout,tuple) else (self.nReadout,)

    @property
    def nOut(self):
        # the number of output neurons of the readout 'linear'
        return self._readoutWidths[-1]

    @property
    def nDimOut(self):
        # flattened real response dimensions of the final responses (see _flatten_responses)
        comp=2 if self.bComplexOut else 1
        rt=str(self.readoutType).lower()
        if rt=='linear':
            return self.nOut*comp
        rows={'resultant':self.nChan+1,'resultant_only':1}.get(rt,self.nChan)
        return rows*comp*self.nSub

    def params0(self,old=None,seed=0):
        """
        initial learned response-model parameters (a dict; see Unit._p), keeping the values in old whose shapes match:
            'pool'  [ nChan ]            pooling weights (readoutType 'resultant', 'resultant_only')
            'bias'  [ nComp x nIn (x nSplit) ] offsets before the activation (bBias)
            'norm'  [ nChan x nChan ]    normalization pool (bLearnNormPool)
            'A'     [ n1 x nChan*nSub ] second-layer weights (readoutType 'linear', n1 the first of nReadout's widths),
                    'bias2' [ nComp x n1 ] its offsets; 'A_l' [ n(l+1) x n(l) ] and 'bias2_l' further layers (l = 1, ...)
        """
        old=old or {}
        rd=np.dtype(jnp.finfo(self.dtype).dtype)
        comp=2 if self.bComplexOut else 1
        out={}
        if self.bPooledReadout:
            out['pool']=np.zeros(self.nChan)
        if self.bBias:
            out['bias']=np.zeros((comp,self.nIn)+((self.filter.nSplit,) if (self.bSplit and not self._bWhiten) else ()))
        if self.bLearnNormPool:
            M=np.ones((self.nChan,self.nChan)) if self.normPool is None else np.asarray(self.normPool)
            out['norm']=np.log(np.expm1(np.maximum(M,1e-6)*np.log(2.)))       # softplus^-1(M log 2)
        if str(self.readoutType).lower()=='linear':
            nI=self.nChan*self.nSub
            rng=np.random.default_rng(seed)
            for l,nO in enumerate(self._readoutWidths):
                sfx='' if l==0 else '_'+str(l)
                out['A'+sfx]=np.eye(nO,nI)+0.1*rng.standard_normal((nO,nI))/np.sqrt(nI)
                if self.bBias:
                    out['bias2'+sfx]=np.zeros((comp,nO))
                nI=nO
        for k,v in list(out.items()):
            o=old.get(k)
            if o is not None and np.shape(o)==v.shape:
                v=np.asarray(o)
            elif o is not None and k=='pool':
                # new filters: pooling weights extended with zeros (equal weights)
                v=v.copy(); o=np.asarray(o).ravel()[:len(v)]; v[:len(o)]=o
            elif o is not None and k=='bias' and np.ndim(o)==v.ndim and np.shape(o)[::2]==v.shape[::2]:
                # new filters (train_append): previous offsets for the previous filters, 0 for the new ones
                v=v.copy(); m=min(v.shape[1],np.shape(o)[1]); v[:,:m]=np.asarray(o)[:,:m]
            out[k]=jnp.asarray(v,dtype=rd)
        return out

    @partial(jit, static_argnames=['self','bGain'])
    def main(self,rng,stim,f,weights=None,W=None,p=None,G=None,bGain=False):
        """
        returns r, rNs, R, RNs, RVar
            r    mean response before normalization
            rNs  noisy response before normalization
            R    mean response (after normalization)
            RNs  noisy response (after normalization)
            RVar noise variance of the final response given its mean, fano*|R| + var0
        with whitening, responses are flattened real dimensions [ nDim x nStim_Ctg x nCtg ] (see _flatten_responses).
        weights [ nStim_Ctg x nCtg ] mark the valid stimuli (for whitenType='response' and respBudget); W is a frozen
        whitening matrix; p the learned response-model parameters (a dict, see Nrn.params0; an array is the pooling
        weights); G the respBudget gains to use (default: those of these stimuli). bGain: return the gains instead
        """
        rng_key1,rng_key2 = jxrandom.split(rng)
        if p is not None and not isinstance(p,dict):
            p={'pool':p}                                                         # pooling weights alone (older callers)

        # respond
        # fourier-domain filters hold half of the frequencies (see _Index.pix). With the orthonormal transform in Stim,
        # the real part of the response is that of the real spatial filter with this spectrum, and the imaginary part is
        # that of its quadrature (hilbert) pair. sqrt(2) gives each of those real filters unit norm when ||f||=1.
        #   fourierType=1: real part only (a real filter)      fourierType=2: both parts (a quadrature pair)
        f=self._scale(f)
        r=self._respond(f,stim)
        # population whitening of the mean responses, before noise
        r=self._whiten_fun(r,f,weights,W)
        r=self._add_bias(r,None if p is None else p.get('bias'))
        r=self._activation_fun(r,self.bFourier,self)
        r=self._split_negatives(r)

        # noisey output 1
        rNs = self._average_fun(self._noise_1_fun(r,self.fano,self.var0,self.nSamples,rng_key1,self.rho))

        # normalize
        M=self._norm_pool(p)
        R   = self._normalize_fun(r,  f,stim,self.eps,self.bSplit,M)
        RN  = self._normalize_fun(rNs,f,stim,self.eps,self.bSplit,M)

        # readout across filters (pooled resultant, or a second layer)
        R   = self._readout_fun(R,p,self)
        RN  = self._readout_fun(RN,p,self)

        # response budget
        if self.respBudget is not None:
            G=self.budget_gain(R,weights) if G is None else G
            if bGain:
                return G
            R,RN=R*G,RN*G

        # noisey output 2
        RNs = self._average_fun(self._noise_2_fun(RN,self.fano,self.var0,self.nSamples,rng_key2,self.rho))

        return r,rNs,R,RNs,self._likelihood_variance(r,R,f,stim,M,G if self.respBudget is not None else None)

    def budget_gain(self,R,weights=None):
        # respBudget / the root mean square of each response dimension over the valid stimuli [ ... x 1 x 1 ]
        if weights is None:
            weights=jnp.ones(R.shape[-2:],dtype=jnp.real(R).dtype)
        ms=jnp.sum(jnp.abs(R)**2*weights,axis=(-2,-1),keepdims=True)/jnp.sum(weights)
        return self.respBudget*_safe_divide(1.,jnp.sqrt(ms))

    def gain(self,stim,f,weights=None,W=None,p=None):
        """the respBudget gains of these (training) stimuli, to decode other stimuli with (None without respBudget)"""
        if self.respBudget is None:
            return None
        if p is not None and not isinstance(p,dict):
            p={'pool':p}
        return self.main(jxrandom.key(0),stim,f,weights,W,p,bGain=True)

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self,rho):
        # the noise correlation type follows rho, also when rho is changed after construction
        if rho is not None and not isinstance(rho,str) and np.ndim(rho)>0:
            rho=np.array(rho,dtype=float)                                        # a correlation matrix
            if rho.ndim!=2 or rho.shape[0]!=rho.shape[1]:
                raise Exception('rho must be a number or a square correlation matrix')
            self._rho=rho
            self.corrType='corr'
            return
        self._rho=rho
        if rho is None or (isinstance(rho,str) and rho=='None'):
            self.corrType='None'
        elif rho==0:
            self.corrType='uncorr'
        else:
            self.corrType='corr'

    def _likelihood_variance(self,r,R,f,stim,M=None,G=None):
        # see the class docstring. r: responses before normalization, R: after (and after the respBudget gains G);
        # M: the 'gen' normalization pool
        if not self.bNoise_1:
            return self.variance(R,self.fano,self.var0)
        v1=self.variance(r,self.fano,self.var0)
        if str(self.normalizeType).lower()=='gen':
            # R_i = r_i/D_i, D_i = eps + sum_j M_ij |r_j| (j over channels, summed over sub-filters):
            # var(R_i) ~ sum_j (dR_i/dr_j)^2 v_j = v_i/D_i^2 - 2 M_ii |r_i| v_i/D_i^3 + r_i^2 sum_j M_ij^2 v_j/D_i^4  (real)
            D=Nrn._gen_denominator(r,self.eps,M)
            D=jnp.where(D>0,D,1)
            if M is None:
                axes=tuple(range(r.ndim-2))
                V=jnp.sum(v1,axis=axes,keepdims=True)
                Mii=1.
            else:
                V=Nrn._pool_rows(v1,M**2)
                Mii=jnp.diagonal(M).reshape((-1,)+(1,)*(r.ndim-1))
            var=v1/D**2 - 2*Mii*jnp.abs(r)*v1/D**3 + r**2*V/D**4
        else:
            # linear normalizations ('broad', 'narrow') divide each response by a stimulus-dependent denominator: its gain
            # is the normalization of ones (also where an activation made the response exactly 0)
            g=self._normalize_fun(jnp.ones(r.shape,dtype=jnp.real(r).dtype),f,stim,self.eps,self.bSplit,M)
            var=v1*g**2
        if G is not None:
            var=var*G**2
        if self.bNoise_2:
            # stage-2 noise is scaled by the noisy stage-1 response: E[fano*|RN| + var0], RN ~ N(R, var) per component
            var=var+self._expected_variance(R,var)
        return var

    def _expected_variance(self,R,var):
        def comp(mu,v):
            sd=jnp.sqrt(v)
            sd_safe=jnp.where(sd>0,sd,1)
            folded=jnp.where(sd>0,sd*jnp.sqrt(2/jnp.pi)*jnp.exp(-mu**2/(2*sd_safe**2)) + mu*jax.scipy.special.erf(mu/(sd_safe*jnp.sqrt(2))),jnp.abs(mu))
            return self.fano*folded + self.var0
        if jnp.iscomplexobj(R):
            return comp(R.real,var.real) + 1j*comp(R.imag,var.imag)
        return comp(R,var)

    @staticmethod
    def variance(R,fano,var0):
        if jnp.iscomplexobj(R):
            return (fano*jnp.abs(R.real) + var0) + 1j*(fano*jnp.abs(R.imag) + var0)
        return fano*jnp.abs(R) + var0

    # respond
    @staticmethod
    @partial(jit, static_argnames=['rmax','bSplit'])
    def respond(f,stim,rmax,bSplit):
        """
        f                         [ nPix x nF ]                       [(nPix / nSplit) x nSplit x nF]
        stim [ nPix  x nStim ] -> [ nPix  x nStim_Ctg x nCtg ]        [(nPix / nSplit) x nSplit x nStim_Ctg x nCtg]
        R    [ nF    x nStim ] -> [ nF    x nStim_Ctg x nCtg]         [ nF x nSplit x nStim_Ctg x nCtg ]
        """
        if bSplit:
            return rmax*jnp.einsum('psf,psnc->fsnc',jnp.conjugate(f),stim)
        else:
            return rmax*jnp.einsum('pf,pnc->fnc',jnp.conjugate(f),stim)

    #- activation (applied to real and imaginary components separately)
    @staticmethod
    def _componentwise(fun,R,bComplex):
        if jnp.iscomplexobj(R):
            return fun(R.real) + 1j*fun(R.imag)
        return fun(R)

    @staticmethod
    def _activation__none(R,*_):
        return R

    @staticmethod
    def _activation__relu(R,bComplex,nrn=None):
        return Nrn._componentwise(lambda x: jnp.maximum(x,0),R,bComplex)

    @staticmethod
    def _activation__softplus(R,bComplex,nrn=None):
        return Nrn._componentwise(lambda x: jnp.logaddexp(x,0),R,bComplex)

    @staticmethod
    def _activation__abs(R,bComplex,nrn=None):
        return Nrn._componentwise(jnp.abs,R,bComplex)

    @staticmethod
    def _activation__logistic(R,bComplex,nrn=None):
        return Nrn._componentwise(lambda x: 1/(1+jnp.exp(-x)),R,bComplex)

    @staticmethod
    def _activation__swish(R,bComplex,nrn=None):
        return Nrn._componentwise(lambda x: x/(1+jnp.exp(-x)),R,bComplex)

    @staticmethod
    def _activation__swish2(R,bComplex,nrn=None):
        return Nrn._componentwise(lambda x: x*((1+jnp.tanh(x))/2),R,bComplex)

    @staticmethod
    def _activation__tanh(R,bComplex,nrn=None):
        return Nrn._componentwise(jnp.tanh,R,bComplex)

    @staticmethod
    def _activation__gauss(R,bComplex,nrn=None):
        return Nrn._componentwise(lambda x: jnp.exp(-x**2),R,bComplex)

    @staticmethod
    def _activation__igauss(R,bComplex,nrn=None):
        return Nrn._componentwise(lambda x: 1-jnp.exp(-x**2),R,bComplex)

    @staticmethod
    def _activation__ramp(R,bComplex,nrn=None):
        rsat=nrn.rmax/2 if nrn.rsat is None else nrn.rsat
        return Nrn._componentwise(lambda x: jnp.clip(x,0,rsat),R,bComplex)

    @staticmethod
    def _activation__nakarushton(R,bComplex,nrn=None):
        c50=nrn.rmax/4 if nrn.c50 is None else nrn.c50
        def nr(x):
            xp=jnp.maximum(x,0)**nrn.nNR
            return nrn.rmax*xp/(xp+c50**nrn.nNR)
        return Nrn._componentwise(nr,R,bComplex)

    @staticmethod
    def _activation__softmax(R,bComplex,nrn=None):
        # over all response dimensions of a stimulus (filters, sub-filters)
        if jnp.iscomplexobj(R):
            raise Exception("activation 'softmax' needs real responses")
        X=jnp.reshape(R,(-1,)+R.shape[-2:])
        return jnp.reshape(nrn.rmax*jax.nn.softmax(X/nrn.softmaxT,axis=0),R.shape)

    #- learned response-model parameters (see Unit._p): offsets, split, normalization pool
    @staticmethod
    def _add_bias(r,b):
        # b [ nComp x (response dims before the stimulus axes) ]: offsets of the real (and imaginary) components
        if b is None:
            return r
        b=b.astype(jnp.real(r).dtype).reshape(b.shape+(1,1))
        if jnp.iscomplexobj(r):
            return r + (b[0] + 1j*b[1])
        return r + b[0]

    def _split_negatives(self,r):
        if not self.bSplitNegatives:
            return r
        on=lambda x: jnp.maximum(x,0)
        if jnp.iscomplexobj(r):
            return jnp.concatenate((on(r.real)+1j*on(r.imag),on(-r.real)+1j*on(-r.imag)),axis=0)
        return jnp.concatenate((on(r),on(-r)),axis=0)

    def _norm_pool(self,p):
        # the 'gen' normalization pool [ nChan x nChan ], or None for all ones
        if str(self.normalizeType).lower()!='gen':
            return None
        if self.bLearnNormPool:
            # relative weights: each row keeps the total weight of the initial pool, or shrinking the denominator would
            # scale all responses up against the fixed additive noise (var0)
            W=jnp.logaddexp(p['norm'],0.)
            M0=np.ones((W.shape[0],W.shape[0])) if self.normPool is None else self.normPool
            return W*(jnp.asarray(M0.sum(1),dtype=W.dtype)/jnp.sum(W,axis=1))[:,None]
        return None if self.normPool is None else jnp.asarray(self.normPool)

    @staticmethod
    def _pool_rows(A,M):
        # sum_j M_ij sum_(other non-stimulus axes) A_j  ->  [ nChan x 1 ... x nStim_Ctg x nCtg ]
        a=jnp.sum(A.reshape(A.shape[:1]+(-1,)+A.shape[-2:]),axis=1)             # [ nChan x nStim_Ctg x nCtg ]
        out=jnp.einsum('ij,jsc->isc',M.astype(a.dtype),a)
        return out.reshape((A.shape[0],)+(1,)*(A.ndim-3)+A.shape[-2:])

    @staticmethod
    def _gen_denominator(R,eps,M):
        if M is None:
            axes=tuple(range(R.ndim-2))
            return eps + jnp.sum(jnp.abs(R),axis=axes,keepdims=True)
        return eps + Nrn._pool_rows(jnp.abs(R),M)

    #- normalize
    @staticmethod
    def _normalize__none(R,*_):
        return R

    @staticmethod
    def _normalize__gen(R,f,stim,eps,bSplit,M=None):
        # divisive normalization by the pooled population response (the pool M, or all channels)
        return _safe_divide(R,Nrn._gen_denominator(R,eps,M))

    @staticmethod
    def _normalize__broad(R,f,stim,eps,bSplit,M=None):
        # N_brd = ||s||_2 (stimulus contrast energy), one value per stimulus
        axes=tuple(range(stim.ndim-2))
        N=jnp.sqrt(jnp.sum(jnp.abs(stim)**2,axis=axes))       # [ nStim_Ctg x nCtg ]
        return _safe_divide(R,eps + N)

    @staticmethod
    def _normalize__narrow(R,f,stim,eps,bSplit,M=None):
        # N_nrw = A_s^T A_f (amplitude spectra), one value per filter and stimulus; stim and f are fourier-domain
        if bSplit:
            return _safe_divide(R,eps + jnp.einsum('psf,psnc->fsnc',jnp.abs(f),jnp.abs(stim)))
        else:
            return _safe_divide(R,eps + jnp.einsum('pf,pnc->fnc',jnp.abs(f),jnp.abs(stim)))

    @staticmethod
    def _normalize__phase(R,f,stim,eps,bSplit,M=None):
        # unit phasors (signs for real responses): contrast- and gain-invariant
        return R/(jnp.abs(R)+eps)

    #- pooled readout over filters (axis 0), weights p [ nF ] (unconstrained; softplus, normalized)
    @staticmethod
    def pool_weights(p):
        w=jnp.logaddexp(p,0.)
        return w/jnp.sum(w)

    @staticmethod
    def _readout__none(R,p,nrn=None):
        return R

    @staticmethod
    def _pool(R,p):
        # weighted sum over filters; real and imaginary parts separately, so the gradient of the real weights is real
        w=Nrn.pool_weights(p).astype(jnp.real(R).dtype)
        if jnp.iscomplexobj(R):
            return (jnp.tensordot(w,R.real,axes=(0,0)) + 1j*jnp.tensordot(w,R.imag,axes=(0,0)))[None]
        return jnp.tensordot(w,R,axes=(0,0))[None]

    @staticmethod
    def _readout__resultant(R,p,nrn=None):
        return jnp.concatenate((R,Nrn._pool(R,p['pool'])),axis=0)

    @staticmethod
    def _readout__resultant_only(R,p,nrn=None):
        return Nrn._pool(R,p['pool'])

    @staticmethod
    def readout_matrix(A):
        # unit-norm rows: the readout, like the filters, cannot raise its signal-to-noise ratio by its scale
        return A/jnp.sqrt(jnp.sum(A**2,axis=1,keepdims=True))

    @staticmethod
    def _readout__linear(R,p,nrn=None):
        # learned layers: [ nOut x nStim_Ctg x nCtg ] from all non-stimulus dimensions of R (see readoutType 'linear')
        out=jnp.reshape(R,(-1,)+R.shape[-2:])
        act=getattr(Nrn,'_activation__'+str(nrn.readoutActivation).lower())
        for l in range(len(nrn._readoutWidths)):
            sfx='' if l==0 else '_'+str(l)
            A=Nrn.readout_matrix(p['A'+sfx]).astype(jnp.real(out).dtype)
            mix=lambda x: jnp.einsum('oi,isc->osc',A,x)
            out=mix(out.real)+1j*mix(out.imag) if jnp.iscomplexobj(out) else mix(out)
            out=act(Nrn._add_bias(out,p.get('bias2'+sfx)),jnp.iscomplexobj(out),nrn)
        return out

    #- filters and responses
    def _scale(self,f):
        return f*_SQRT2 if self.bFourier else f

    def _respond(self,f,stim):
        # f scaled by _scale
        r=self.respond(f,stim,self.rmax,self.bSplit)
        if self.bFourier and not self.bAnalytic:
            r=r.real
        return r

    def _filter_matrix(self,f):
        """
        real spatial filters [ nDim x (nSplit) x nPix ], one per flattened response dimension (same order as
        _flatten_responses), whose dot products with the spatial stimulus give the responses. f is scaled by _scale.
        """
        h=f
        if self.bFourier:
            pix=tuple(self.filter.pix_dims)
            h=jnp.reshape(_centered_ft(jnp.reshape(f,pix+f.shape[1:]),tuple(range(len(pix))),inverse=True),f.shape)
        h=jnp.moveaxis(h,-1,0)                                                  # [ nF x nPix x (nSplit) ]
        if self.bSplit:
            h=jnp.swapaxes(h,1,2)                                               # [ nF x nSplit x nPix ]
        if self.bFourier:
            # response = sum conj(h) s: the real part responds with Re h, the imaginary part with -Im h
            h=jnp.concatenate((h.real,-h.imag),axis=0) if self.bAnalytic else h.real
        return h

    def _gram(self,f):
        E=self._filter_matrix(f)
        if self.bSplit:
            # sub-filters of different parts (e.g. eyes) do not overlap
            n,S=E.shape[0],E.shape[1]
            Gs=jnp.einsum('iep,jep->eij',E,E)
            return jnp.reshape(jnp.einsum('eij,ed->iejd',Gs,jnp.eye(S,dtype=Gs.dtype)),(n*S,n*S))
        return E@E.T

    def gram(self,f):
        """[ nDim x nDim ] inner products of the real spatial filters behind the flattened response dimensions"""
        return self._gram(self._scale(f))

    @staticmethod
    def _response_cov(Rf,weights):
        # covariance of flattened responses [ nDim x nStim_Ctg x nCtg ] over the valid stimuli
        n=jnp.sum(weights)
        mu=jnp.sum(Rf*weights,axis=(1,2),keepdims=True)/n
        D=(Rf-mu)*jnp.sqrt(weights)
        return jnp.einsum('isc,jsc->ij',D,D)/(n-1)

    @property
    def _bWhiten(self):
        return str(self.whitenType).lower()!='none'

    def _whitening_matrix(self,M):
        # D^1/2 M^-1/2 (zca) or D^1/2 L^-1 (chol, M = L L^T), D = diag(M): W M W^T = D
        d=jnp.diagonal(M)
        Mr=M + self.whitenEps*jnp.mean(d)*jnp.eye(M.shape[0],dtype=M.dtype)
        if self.whitenMethod=='chol':
            Minvh=jax.scipy.linalg.solve_triangular(jnp.linalg.cholesky(Mr),jnp.eye(M.shape[0],dtype=M.dtype),lower=True)
        else:
            Minvh=_inv_sqrtm(Mr)
        return jnp.sqrt(d)[:,None]*Minvh

    def whitening(self,stim,f,weights=None):
        """whitening matrix [ nDim x nDim ] for filters f (unscaled, e.g. Filter.out_flat) and stimuli"""
        f=self._scale(f)
        if str(self.whitenType).lower()=='gram':
            return self._whitening_matrix(self._gram(f))
        Rf=_flatten_responses(self._respond(f,stim))
        if weights is None:
            weights=jnp.ones(Rf.shape[-2:],dtype=Rf.dtype)
        return self._whitening_matrix(self._response_cov(Rf,weights))

    #- whitening: [ nF x (nSplit) x nStim_Ctg x nCtg ] (possibly complex) -> real [ nDim x nStim_Ctg x nCtg ]
    def _whiten__none(self,r,f,weights,W):
        return r

    def _whiten__gram(self,r,f,weights,W):
        if W is None:
            W=self._whitening_matrix(self._gram(f))
        return jnp.einsum('ij,jsc->isc',W,_flatten_responses(r))

    def _whiten__response(self,r,f,weights,W):
        Rf=_flatten_responses(r)
        if W is None:
            if weights is None:
                weights=jnp.ones(Rf.shape[-2:],dtype=Rf.dtype)
            W=self._whitening_matrix(self._response_cov(Rf,weights))
        return jnp.einsum('ij,jsc->isc',W,Rf)

    #- noise: returns samples along a trailing axis [ ... x nSamples ]
    @staticmethod
    def _noise__none(R,*_):
        return R[...,None]

    @staticmethod
    def _noise__true(R,fano,var0,nSamples,rng_key,rho):
        """
        samples [ ... x nSamples ] of R + eta, eta ~ N(0, fano*|R| + var0) in each response dimension (real and imaginary
        components of complex responses are separate dimensions), correlated by rho across all dimensions
        """
        bComplex=jnp.iscomplexobj(R)
        Rr=jnp.concatenate((R.real,R.imag),axis=0) if bComplex else R          # [ nDim x ... ]
        z=jxrandom.normal(rng_key,Rr.shape+(nSamples,),dtype=Rr.dtype)
        if rho is not None and (np.ndim(rho)>0 or rho!=0):
            zf=jnp.reshape(z,(-1,)+z.shape[-3:])                                 # [ nDim' x nStim_Ctg x nCtg x nSamples ]
            L=jnp.linalg.cholesky(Nrn._corr_mat(rho,zf.shape[0],zf.dtype))
            z=jnp.reshape(jnp.einsum('fg,g...->f...',L,zf),z.shape)
        eta=z*jnp.sqrt(fano*jnp.abs(Rr)+var0)[...,None]
        if bComplex:
            n=R.shape[0]
            eta=eta[:n] + 1j*eta[n:]
        return R[...,None] + eta

    #- noise averaging over samples
    @staticmethod
    def _average__full(RNs):
        # a single noisy response
        return RNs[...,0]

    @staticmethod
    def _average__mean(RNs):
        return jnp.mean(RNs,axis=-1)

    @staticmethod
    def _average__log_mean(RNs):
        gm=lambda x: jnp.sign(jnp.mean(x,axis=-1))*jnp.exp(jnp.mean(jnp.log(jnp.abs(x)),axis=-1))
        if jnp.iscomplexobj(RNs):
            return gm(RNs.real) + 1j*gm(RNs.imag)
        return gm(RNs)

    @staticmethod
    def _average__median(RNs):
        if jnp.iscomplexobj(RNs):
            return jnp.median(RNs.real,axis=-1) + 1j*jnp.median(RNs.imag,axis=-1)
        return jnp.median(RNs,axis=-1)

    def corr_matrix(self,n,dtype=float):
        # noise correlation between the n flattened response dimensions
        return Nrn._corr_mat(self.rho,n,dtype)

    @staticmethod
    def _corr_mat(rho,n,dtype=float):
        # rho: one correlation for all pairs, or the correlation matrix itself
        if np.ndim(rho)>0:
            return jnp.asarray(rho,dtype=dtype)
        return rho + (1-rho)*jnp.eye(n,dtype=dtype)

    #- noise covariance, per category, of the flattened (real) responses
    #  RVar [ nF x nStim_Ctg x nCtg ], weights [ nStim_Ctg x nCtg ] -> [ nCtg x nF x nF ]
    @staticmethod
    def _mean_var(RVar,weights):
        return jnp.sum(RVar*weights,axis=1)/jnp.sum(weights,axis=0)  # [ nF x nCtg ]

    @staticmethod
    def _corr__none(RVar,weights,rho):
        return jnp.zeros((RVar.shape[-1],RVar.shape[0],RVar.shape[0]),dtype=RVar.dtype)

    @staticmethod
    def _corr__uncorr(RVar,weights,rho):
        v=Nrn._mean_var(RVar,weights).T                              # [ nCtg x nF ]
        return v[:,:,None]*jnp.eye(RVar.shape[0],dtype=RVar.dtype)[None]

    @staticmethod
    def _corr__corr(RVar,weights,rho):
        sd=jnp.sqrt(Nrn._mean_var(RVar,weights).T)                   # [ nCtg x nF ]
        corrMat=Nrn._corr_mat(rho,RVar.shape[0],RVar.dtype)
        return sd[:,:,None]*sd[:,None,:]*corrMat[None]


class Model(_Static):
    """
    Likelihood model. Operates on flattened real responses [ nF x nStim_Ctg x nCtg ] and returns
    log-likelihoods lAll [ nStim_Ctg x nCtg(true) x nCtg(candidate) ],  lAll[l,k,i] = log p(R(k,l) | X_i)

    'gss'  AMA-Gauss (Jaini & Burge 2017):  p(R|X_i) = N(R; mu_i, Sigma_i + Lambda_i)
           mu_i, Sigma_i = mean and (unbiased) covariance of mean responses in category i
           Lambda_i      = noise covariance from Nrn (mean noise variance in category i)
    'full' original AMA (Burge & Jaini 2017, Eq 5):  p(R|X_i) = 1/N_i sum_j N(R; r_ij, diag(sigma2_ij))
    'student' as 'gss' with a multivariate student t of df degrees of freedom, scaled so its covariance is
           Sigma_i + Lambda_i (df > 2): heavy tails, e.g. for natural-image responses
    'circ' circular complex gaussian on quadrature-pair responses (fourierType=2):
           p(R|X_i) = CN(R; mu_i, C_i + N_i), C_i hermitian covariance of the complex mean responses.
           circMean='zero' fixes mu_i = 0 and uses the second moment E[R R^H]: the likelihood for responses
           whose complex gain (contrast and phase) is unknown, R = G m_i + noise with a circularly symmetric G.
    The prior p(X_i)=N_i/N is applied by Objective, so the posterior equals Eq 5 exactly.

    'mix'  a gaussian mixture per category: p(R|X_i) = sum_c pi_ic N(R; mu_ic, Sigma_ic + Lambda_i), fit to the category's
           mean responses by nEM EM iterations, initialized by quantiles of the responses along the category's first
           principal axis (a piecewise constant assignment). The EM iterations are differentiated, so gradients include
           how the fit moves with the filters. The fitted component covariances are Bessel corrected, so nMix=1 with
           mixReg=0 is 'gss'.
           nMix   - components per category
           nEM    - EM iterations
           bWarmEM, nEMWarm - while training, start EM from the previous iteration's fit and run nEMWarm iterations
                    (unit.loss and evaluate fit from scratch with nEM)
           bLeaveOneOut - the stimulus's responsibility-weighted contribution is removed from its own category's
                    components, with the responsibilities of the full fit held fixed
           mixReg - ridge added to each component covariance, relative to the category's mean response variance
                    (keeps components from collapsing onto few stimuli)
           covShrink - shrink each component covariance toward covTarget, before the ridge: its diagonal ('diag'), or
                    the count-weighted mean category covariance of all categories ('pooled')
           ctgPoolWidth - each component c of category i also borrows the kernel-weighted scatter of the other
                    categories, in proportion to its weight: Sigma_ic = (S_ic + pi_ic P_i)/(dof_ic + pi_ic Q_i), with
                    P_i = sum_{j!=i} K_ij S_j and Q_i = sum_{j!=i} K_ij (N_j-1) over the other categories' whole scatters
                    S_j (the category's own share relative to its neighbours is that of 'gss', so nMix=1 with mixReg=0 is
                    'gss' with ctgPoolWidth). bPoolMeans is not used.
                    With either kind of pooling, bLeaveOneOut also removes the stimulus from the other categories'
                    pooled statistics (with their responsibilities fixed).
           A component that EM empties keeps weight 0; a warm start from such a fit starts again from the initialization
           (nEM iterations) instead.

    nRef ('full' only) - while training, decode each stimulus against nRef reference stimuli per category, drawn at
                   random each iteration, instead of all of them: O(N nRef) instead of O(N^2) per iteration. The cost
                   is then a stochastic (slightly pessimistic: log of a sample mean) estimate of full AMA's; unit.loss,
                   evaluate and performance decode against all training stimuli.

    nNeighbors ('full' only) - while training, decode each stimulus exactly against the nNeighbors references of each
                   category with the largest likelihood terms, plus nTail references per category drawn at random each
                   iteration, which estimate the sum over the rest of that category without bias (each of the rest is
                   below the neighbours, which bounds the variance). The neighbours are searched again with the current
                   filters every Optimizer nStepsPerChunk iterations (an O(N^2) search, like one evaluation of full AMA);
                   each iteration in between is O(N nCtg (nNeighbors + nTail)). unit.loss, evaluate and performance
                   decode against all training stimuli. Not combined with nRef. With Optimizer batchSize, each
                   iteration decodes a batch of stimuli (stratified by category, as AMA-SGD) against all training
                   stimuli (their neighbours and the tail), with the prior of all training stimuli: O(batchSize nCtg
                   (nNeighbors + nTail)) per iteration, plus computing every stimulus's responses.

    bWithin      - continuous estimates within the categories, for stimuli with their own latent values (Stim y, e.g.
                   Stim.binned): each candidate category i stands for E[y | R, X_i] instead of its level Y_i, and the
                   'mean', 'median' and 'mode' estimates (errType 'l1', 'l2'; Unit.estimates, performance) combine
                   these by the posterior over categories.
                   'full': the likelihood-weighted mean of its reference stimuli's values (kernel regression; with
                   bLeaveOneOut without the stimulus itself).
                   others: the linear regression within the category, ybar_i + c_i^T (Sigma_i + Lambda_i)^-1 (R - mu_i),
                   with c_i the covariance of the mean responses and y, from the sample statistics (no shrinkage or
                   pooling; with bLeaveOneOut, the own category's statistics and noise covariance without the stimulus)

    covRank ('gss', 'student', 'mix', 'circ') - model each category (or mixture component) covariance as factor analysis,
                   L L^T + Psi (L L^H + Psi, complex, for 'circ') with covRank factors L and a diagonal Psi, fit to the sample covariance by nFA EM
                   iterations (differentiated; initialized from its leading subspace). Needs far fewer stimuli per
                   category than a full covariance when there are many response dimensions. Applied before covShrink,
                   and to the left-out covariances of bLeaveOneOut. covRank=0 is a diagonal covariance.

    Category statistics ('gss', 'student', 'circ'):
    covShrink     - shrink each category covariance toward covTarget by this fraction (0 = sample covariance):
                    Sigma = (1-covShrink) Sigma + covShrink T, before the noise covariance is added
    covTarget     - 'diag' (the covariance's own diagonal) or 'pooled' (the count-weighted mean covariance of all
                    categories)
    ctgPoolWidth  - pool category statistics over neighbouring latent values with a gaussian kernel of this width
                    (in units of Y): each category's covariance is the kernel- and count-weighted pooled scatter,
                    which stabilizes covariances of small categories whose statistics change smoothly with Y
    bPoolMeans    - also pool the category means with the same kernel (biases the means toward neighbours)

    bLeaveOneOut - leave the decoded stimulus out of its own category. Otherwise each stimulus is scored against statistics
                   that include it, which makes the cost optimistic when noise is low, filters are many, or categories
                   (or batches) are small.
                   'full': the posterior is Eq 5 with that stimulus removed from the training set.
                   'gss', 'student', 'circ': the category statistics are recomputed without it, exactly: a rank-one
                   downdate of its own category's mean and covariance, and with ctgPoolWidth (bPoolMeans) or the pooled
                   shrinkage target also of every category's pooled covariance (and mean), which it enters too.
                   The stimulus is also left out of its category's prior, (N_k-1)/(N-1), and noise covariance (the mean
                   noise variance); full AMA's form already has the left-out prior (see _model__full).
    nNoiseSamples - with responseType='basic', the cost is the mean over this many noisy observations of each stimulus
                   (independent noise draws, decoded against the same statistics or references): a Monte Carlo estimate of
                   the expected cost under the response noise. The mean-response approximation (responseType='mean')
                   decodes noise-free responses and underestimates that cost, more so at high noise. Training, unit.loss,
                   evaluate. Memory grows with nNoiseSamples, except for 'full' (the samples run one after another).
    bFixedNoise  - while training, draw the same noise (from noiseSeed) at every iteration, so the training cost is a
                   deterministic function of the filters (a sample average approximation; e.g. for Optimizer 'lbfgs' or
                   tolFun). Reference subsets (nRef, nNeighbors' tail) and batches are still drawn anew.
    bLooNoise    - with bLeaveOneOut ('gss', 'student', 'circ', 'mix'), also leave the stimulus out of its category's noise
                   covariance. False keeps the category's noise covariance, an O(1/N_k) change of its mean noise variance
                   (none when the noise variance does not depend on the response, fano=0). Without covRank, covShrink or
                   pooling over categories, the left-out covariance of 'gss', 'student' and 'circ' is then a rank-one
                   downdate of one matrix per category, and each left-out likelihood costs O(nF^2) (matrix determinant
                   lemma and Sherman-Morrison) instead of a Cholesky factorization per stimulus.
    """
    _model_fun=_id
    _response_fun=_id
    _Yperiod=None                                                               # from the Stim, set by Unit._finalize
    modelType=_TypeFunc()
    responseType=_TypeFunc()

    def __init__(self,modelType='gss',responseType='basic',bLeaveOneOut=False,covShrink=0.,covTarget='diag',df=5.,
                 ctgPoolWidth=None,bPoolMeans=False,circMean='estimate',nMix=2,nEM=20,mixReg=1e-3,nRef=None,covRank=None,nFA=50,
                 bWarmEM=False,nEMWarm=3,nNeighbors=None,nTail=8,bWithin=False,bLooNoise=True,nNoiseSamples=1,bFixedNoise=False,
                 noiseSeed=0):
        self.modelType=modelType
        self.responseType=responseType
        self.bLeaveOneOut=bLeaveOneOut
        self.covShrink=float(covShrink)
        self.covTarget=covTarget
        self.df=float(df)
        self.ctgPoolWidth=None if ctgPoolWidth is None else float(ctgPoolWidth)
        self.bPoolMeans=bool(bPoolMeans)
        self.circMean=circMean
        self.nMix=int(nMix)
        self.nEM=int(nEM)
        self.mixReg=float(mixReg)
        self.nRef=None if nRef is None else int(nRef)
        self.bWarmEM=bool(bWarmEM)
        self.nEMWarm=int(nEMWarm)
        self.covRank=None if covRank is None else int(covRank)
        self.nFA=int(nFA)
        self.nNeighbors=None if nNeighbors is None else int(nNeighbors)
        self.nTail=int(nTail)
        self.bWithin=bool(bWithin)
        self.bLooNoise=bool(bLooNoise)
        self.nNoiseSamples=int(nNoiseSamples)
        self.bFixedNoise=bool(bFixedNoise)
        self.noiseSeed=int(noiseSeed)
        if self.nNoiseSamples<1:
            raise Exception('nNoiseSamples must be at least 1')
        if (self.nNeighbors is not None and self.nNeighbors<1) or self.nTail<1:
            raise Exception('nNeighbors must be None or at least 1, and nTail at least 1')
        if (self.covRank is not None and self.covRank<0) or self.nFA<1:
            raise Exception('covRank must be None or not negative, and nFA at least 1')
        if self.nRef is not None and self.nRef<1:
            raise Exception('nRef must be None or at least 1')
        if self.nMix<1 or self.nEM<0 or self.mixReg<0:
            raise Exception('nMix must be at least 1, nEM and mixReg not negative')
        if not 0<=self.covShrink<=1:
            raise Exception('covShrink must be in [0, 1]')
        if covTarget not in ('diag','pooled'):
            raise Exception("covTarget must be 'diag' or 'pooled'")
        if circMean not in ('estimate','zero'):
            raise Exception("circMean must be 'estimate' or 'zero'")
        if self.ctgPoolWidth is not None and self.ctgPoolWidth<=0:
            raise Exception('ctgPoolWidth must be positive (or None)')

    def _key(self):
        return (self.modelType,self.responseType,self.bLeaveOneOut,self.covShrink,self.covTarget,self.df,
                self.ctgPoolWidth,self.bPoolMeans,self.circMean,self._Yperiod,self.nMix,self.nEM,self.mixReg,self.nRef,
                self.covRank,self.nFA,self.bWarmEM,self.nEMWarm,self.nNeighbors,self.nTail,self.bWithin,self.bLooNoise,
                self.nNoiseSamples,self.bFixedNoise,self.noiseSeed)

    def copy(self):
       return Model(**_get_copy_dict(self))

    #-main
    @partial(jit, static_argnames=['self'])
    def lrn_main(self,R,Rm,RVar,noiseCov,noiseCorr,weights,Y,refIdx=None,yRef=None):
        # yRef [ nStim_Ctg x nCtg (x nDim) ]: the reference stimuli's own latent values (bWithin); then (lAll, Yc)
        return self._model_fun(R,Rm,RVar,noiseCov,noiseCorr,weights,self.bLeaveOneOut,self,Y,refIdx,yRef)

    #- within-category estimates (bWithin)
    @staticmethod
    def _with_within(lAll,R,Rm,noiseCov,weights,yRef,Y,m,bLeaveOneOut=False,RVar=None):
        if yRef is None:
            return lAll
        noiseL=Model._noise_loo(noiseCov,RVar,weights,m) if bLeaveOneOut else None
        return lAll,Model._within_linear(R,Rm,noiseCov,weights,yRef,Y,m._Yperiod,noiseL)

    @staticmethod
    def _within_linear(R,Rm,noiseCov,weights,yRef,Y,per,noiseL=None):
        """
        Yc [ nStim_Ctg x nCtg x nCtg(i) (x nDim) ]: the linear least-squares estimate of the latent value from the observed
        responses R within each candidate category i, ybar_i + c_i^T (Sigma_i + Lambda_i)^-1 (R - mu_i), from the sample
        statistics of the reference responses Rm, their latent values yRef, and the noise covariance. On circular
        dimensions the values are offsets from the level Y_i, wrapped.
        noiseL [ nStim_Ctg x nCtg x nF x nF ] (bLeaveOneOut): each stimulus's own category is estimated from its statistics
        without the stimulus (rank-one downdates of the means and scatters, weights 0 or 1) and this noise covariance
        """
        wc=jnp.sum(weights,axis=0)
        mu=jnp.sum(Rm*weights,axis=1)/wc                                         # [ nF x nCtg ]
        Yv=Y if Y.ndim>1 else Y[:,None]                                          # [ nCtg x nDim ]
        d=_wrap((yRef if Y.ndim>1 else yRef[...,None])-Yv[None],per)             # [ nStim_Ctg x nCtg x nDim ]
        dbar=jnp.sum(d*weights[...,None],axis=0)/wc[:,None]
        Dr=Rm-mu[:,None,:]
        S=jnp.einsum('fsc,gsc->cfg',Dr*weights,Dr)/(wc-1)[:,None,None]
        C=jnp.einsum('fsc,scd->cfd',Dr*weights,d-dbar[None])/(wc-1)[:,None,None] # [ nCtg x nF x nDim ]
        B=jnp.linalg.solve(S+noiseCov,C)
        x=jnp.transpose(R,(1,2,0))[:,:,None,:]-mu.T[None,None]                  # [ nStim_Ctg x nCtg x nCtg(i) x nF ]
        Yc=(Yv+dbar)[None,None]+jnp.einsum('lkif,ifd->lkid',x,B)
        if noiseL is not None:
            dR=jnp.transpose(Dr,(1,2,0))                                         # [ nStim_Ctg x nCtg x nF ]
            e=d-dbar[None]                                                       # [ nStim_Ctg x nCtg x nDim ]
            r=(weights/(wc-weights))[...,None]
            cc=(weights*wc/(wc-weights))[...,None,None]
            dofL=(wc[None]-1-weights)[...,None,None]
            SL=(S[None]*(wc-1)[None,:,None,None]-cc*dR[...,:,None]*dR[...,None,:])/dofL
            CL=(C[None]*(wc-1)[None,:,None,None]-cc*dR[...,:,None]*e[...,None,:])/dofL
            xL=jnp.transpose(R,(1,2,0))-(mu.T[None]-r*dR)
            v=jnp.linalg.solve(SL+noiseL,xL[...,None])[...,0]
            own=Yv[None]+dbar[None]-r*e+jnp.einsum('lkf,lkfd->lkd',v,CL)
            Yc=jnp.where(jnp.eye(Yc.shape[2],dtype=bool)[None,:,:,None],own[:,:,None],Yc)
        return Yc if Y.ndim>1 else Yc[...,0]

    @staticmethod
    def _ref_subset(rng,weights,nRef):
        # nRef random valid reference stimuli per category [ nRef x nCtg ] (fewer valid ones: padding, weight 0, fills up)
        nRef=min(nRef,weights.shape[0])
        score=jxrandom.uniform(rng,weights.shape) + jnp.where(weights>0,0.,jnp.inf)
        return jnp.argsort(score,axis=0)[:nRef]

    #- category statistics
    @staticmethod
    def _ctg_kernel(Y,width,per=None):
        # [ nCtg x nCtg ] gaussian weights over (wrapped) distances between latent values
        return jnp.exp(-_ydist2(Y,per)/(2*width**2))

    @staticmethod
    def _ctg_stats(Rm,weights,m,Y,bCentered=True):
        """
        category means [ nF x nCtg ] and covariances [ nCtg x nF x nF ] of the mean responses Rm [ nF x nStim_Ctg x nCtg ]
        (real or complex; covariances are hermitian), with pooling over neighbouring categories and shrinkage (see Model)
        """
        wc=jnp.sum(weights,axis=0)                                               # [ nCtg ]
        mu=jnp.sum(Rm*weights,axis=1)/wc                                         # [ nF x nCtg ]
        if m.ctgPoolWidth is not None and m.bPoolMeans:
            K=Model._ctg_kernel(Y,m.ctgPoolWidth,m._Yperiod)*wc[None,:]
            mu=jnp.einsum('ik,fk->fi',K,mu)/jnp.sum(K,axis=1)[None,:]
        Dv=(Rm-mu[:,None,:])*jnp.sqrt(weights) if bCentered else Rm*jnp.sqrt(weights)
        S=jnp.einsum('isc,jsc->cij',Dv,jnp.conj(Dv))                             # [ nCtg x nF x nF ] scatter
        dof=(wc-1) if bCentered else wc
        if m.ctgPoolWidth is not None:
            K=Model._ctg_kernel(Y,m.ctgPoolWidth,m._Yperiod)
            cov=jnp.einsum('ik,kfg->ifg',K,S)/(K@dof)[:,None,None]
        else:
            cov=S/dof[:,None,None]
        cov=Model._low_rank(cov,m)
        if m.covShrink>0:
            if m.covTarget=='pooled':
                T=jnp.broadcast_to(jnp.sum(S,axis=0)/jnp.sum(dof),cov.shape)
            else:
                T=cov*jnp.eye(cov.shape[-1],dtype=cov.dtype)
            cov=(1-m.covShrink)*cov + m.covShrink*T
        return mu,cov

    @staticmethod
    def _low_rank(S,m):
        """
        factor analysis covariance L L^H + Psi (covRank factors) of sample covariances S [ ... x p x p ], real symmetric or
        complex hermitian ('circ'), by the EM of Ghahramani & Hinton (1996) on S: beta = L^H Sigma^-1,
        Ezz = I - beta L + beta S beta^H, L = S beta^H Ezz^-1, Psi = diag(S - L beta S). Initialized from the leading
        subspace of S (subspace iteration).
        """
        r=m.covRank
        if r is None:
            return S
        p=S.shape[-1]
        dg=jnp.real(jnp.diagonal(S,axis1=-2,axis2=-1))
        floor=1e-6*jnp.mean(dg,axis=-1,keepdims=True)                          # keeps Psi positive
        if r==0:
            return jnp.maximum(dg,floor)[...,None]*jnp.eye(p,dtype=S.dtype)
        mH=lambda A: jnp.conj(jnp.swapaxes(A,-1,-2))
        # initial factors from the leading subspace by subspace iteration: unlike eigh (whose gradient is not finite at
        # repeated eigenvalues, e.g. the zeros of a covariance of few stimuli) it is smooth, so the whole fit is
        # differentiated exactly
        Q=jnp.broadcast_to(jnp.asarray(np.linalg.qr(np.random.default_rng(0).standard_normal((p,r)))[0],dtype=S.dtype),
                           S.shape[:-2]+(p,r))
        for _ in range(8):
            Q=jnp.linalg.qr(S@Q)[0]
        lead=jnp.real(jnp.diagonal(mH(Q)@S@Q,axis1=-2,axis2=-1))              # [ ... x r ]
        sig2=(jnp.sum(dg,axis=-1,keepdims=True)-jnp.sum(lead,axis=-1,keepdims=True))/max(p-r,1)
        L=Q*jnp.sqrt(jnp.maximum(lead-sig2,floor))[...,None,:]
        psi=jnp.maximum(dg-jnp.sum(jnp.abs(L)**2,axis=-1),floor)
        I=jnp.eye(r,dtype=S.dtype)
        eye=jnp.eye(p,dtype=S.dtype)
        def step(_,Lpsi):
            L,psi=Lpsi
            Sig=L@mH(L)+psi[...,None]*eye
            beta=mH(jnp.linalg.solve(Sig,L))                                     # [ ... x r x p ]
            Ezz=I-beta@L+beta@S@mH(beta)
            L=S@mH(beta)@jnp.linalg.inv(Ezz)
            psi=jnp.maximum(jnp.real(jnp.diagonal(S-L@beta@S,axis1=-2,axis2=-1)),floor)
            return L,psi
        L,psi=lax.fori_loop(0,m.nFA,step,(L,psi))
        return L@mH(L)+psi[...,None]*eye

    @staticmethod
    def _ctg_stats_loo(Rm,weights,m,bCentered=True):
        """
        own-category mean [ nStim_Ctg x nCtg x nF ] and covariance [ nStim_Ctg x nCtg x nF x nF ] of each stimulus's
        category with that stimulus (weight w) left out: a rank-one downdate of the scatter, with shrinkage recomputed
        """
        wc=jnp.sum(weights,axis=0)                                               # [ nCtg ]
        mu=jnp.sum(Rm*weights,axis=1)/wc if bCentered else jnp.zeros(Rm.shape[::2],dtype=Rm.dtype)
        Dv=(Rm-mu[:,None,:])*jnp.sqrt(weights)
        S=jnp.einsum('isc,jsc->cij',Dv,jnp.conj(Dv))                             # [ nCtg x nF x nF ]
        dof=(wc-1) if bCentered else wc

        d=jnp.transpose(Rm-mu[:,None,:],(1,2,0))                                 # [ nStim_Ctg x nCtg x nF ]
        w=weights[...,None]
        c=weights*wc/(wc-weights) if bCentered else weights                     # [ nStim_Ctg x nCtg ]
        dS=c[...,None,None]*d[...,:,None]*jnp.conj(d)[...,None,:]
        muL=(mu.T[None] - w*d/(wc[None,:,None]-w)) if bCentered else jnp.zeros_like(d)
        dofL=dof[None]-weights
        cov=(S[None]-dS)/dofL[...,None,None]
        cov=Model._low_rank(cov,m)
        if m.covShrink>0:
            if m.covTarget=='pooled':
                T=(jnp.sum(S,axis=0)[None,None]-dS)/(jnp.sum(dof)-weights)[...,None,None]
            else:
                T=cov*jnp.eye(cov.shape[-1],dtype=cov.dtype)
            cov=(1-m.covShrink)*cov + m.covShrink*T
        return muL,cov

    @staticmethod
    def _loo_needs_all(m):
        # whether leaving a stimulus out changes the statistics of other categories too
        return m.ctgPoolWidth is not None or (m.covShrink>0 and m.covTarget=='pooled')

    @staticmethod
    def _ctg_stats_loo_all(Rm,weights,m,Y,bCentered=True):
        """
        every category's mean [ nStim_Ctg x nCtg(k) x nCtg(i) x nF ] and covariance [ nStim_Ctg x nCtg(k) x nCtg(i) x nF x nF ]
        with stimulus (l,k) left out, exactly, including category pooling (ctgPoolWidth, bPoolMeans) and the pooled
        shrinkage target. Built from each category's own mean o_j and scatter C_j about it: a category's scatter about
        any point a is C_j + n_j (o_j-a)(o_j-a)^H, and leaving a stimulus out downdates only its own n, o and C.
        """
        nF,_,nC=Rm.shape
        wc=jnp.sum(weights,axis=0)                                               # [ nCtg ]
        eye=jnp.eye(nC,dtype=weights.dtype)
        if bCentered:
            o=(jnp.sum(Rm*weights,axis=1)/wc).T                                  # [ nCtg x nF ] own means
        else:
            o=jnp.zeros((nC,nF),dtype=Rm.dtype)
        Dv=(Rm-o.T[:,None,:])*jnp.sqrt(weights)
        C=jnp.einsum('isc,jsc->cij',Dv,jnp.conj(Dv))                             # [ nCtg x nF x nF ] own scatter
        K=eye if m.ctgPoolWidth is None else Model._ctg_kernel(Y,m.ctgPoolWidth,m._Yperiod).astype(weights.dtype)

        # the left-out stimulus's own category k: counts, means, and scatter downdate
        w=weights                                                                # [ l x k ]
        d=jnp.transpose(Rm,(1,2,0))-o[None]                                      # [ l x k x nF ]
        c=w*wc/(wc-w) if bCentered else w
        dC=c[...,None,None]*d[...,:,None]*jnp.conj(d)[...,None,:]               # [ l x k x nF x nF ]
        nL=wc[None,None,:]-w[...,None]*eye[None]                                 # [ l x k x j ]
        oL=o[None,None]+eye[None,:,:,None]*((-w/(wc-w))[...,None]*d if bCentered else jnp.zeros_like(d))[:,:,None,:]
        dofL=nL-1 if bCentered else nL                                           # [ l x k x j ]

        # means: own, or pooled over categories
        if bCentered and m.ctgPoolWidth is not None and m.bPoolMeans:
            Kn=jnp.einsum('ij,lkj->lkij',K,nL)                                   # [ l x k x i x j ]
            mu=jnp.einsum('lkij,lkjf->lkif',Kn,oL)/jnp.sum(Kn,axis=-1)[...,None]
            # scatter of each category j about its (pooled) mean: + n_j e_j e_j^H, e_j = o_j - m_j
            e=oL-mu
            extra=nL[...,None,None]*e[...,:,None]*jnp.conj(e)[...,None,:]        # [ l x k x j x nF x nF ]
        else:
            mu=oL
            extra=None

        def pooled(Kw):
            # sum_j Kw_ij S_j with the left-out stimulus removed: [ l x k x i x nF x nF ]
            P=jnp.einsum('ij,jfg->ifg',Kw,C)[None,None] - Kw.T[None,:,:,None,None]*dC[:,:,None]
            if extra is not None:
                P=P+jnp.einsum('ij,lkjfg->lkifg',Kw,extra)
            return P

        cov=pooled(K)/jnp.einsum('ij,lkj->lki',K,dofL)[...,None,None]
        cov=Model._low_rank(cov,m)
        if m.covShrink>0:
            if m.covTarget=='pooled':
                ones=jnp.ones_like(K)
                T=(pooled(ones)/jnp.einsum('ij,lkj->lki',ones,dofL)[...,None,None])
            else:
                T=cov*jnp.eye(nF,dtype=cov.dtype)
            cov=(1-m.covShrink)*cov + m.covShrink*T
        return mu,cov

    @staticmethod
    def _noise_loo(noiseCov,RVar,weights,m=None):
        """
        the noise covariance of each stimulus's own category without it [ nStim_Ctg x nCtg x nF x nF ]: the mean noise
        variances are recomputed without the stimulus, keeping the correlations (the category's own with m.bLooNoise False)
        """
        if m is not None and not m.bLooNoise:
            return jnp.broadcast_to(noiseCov[None],weights.shape+noiseCov.shape[-2:])
        wc=jnp.sum(weights,axis=0)
        v=(jnp.sum(RVar*weights,axis=1)/wc).T                                     # [ nCtg x nF ]
        w=weights[...,None]
        vL=(wc[None,:,None]*v[None]-w*jnp.transpose(RVar,(1,2,0)))/(wc[None,:,None]-w)
        sc=jnp.sqrt(_safe_divide(vL,v[None]))
        return noiseCov[None]*sc[...,:,None]*sc[...,None,:]

    @staticmethod
    def _loo_rank_one(m):
        # whether the left-out own-category covariance is a rank-one downdate of one matrix per category (see bLooNoise)
        return not m.bLooNoise and m.covRank is None and m.covShrink==0 and not Model._loo_needs_all(m)

    @staticmethod
    def _loo_quad(R,Rm,noiseCov,weights,bCentered=True):
        """
        the quadratic form q = x^H M^-1 x and log det M [ nStim_Ctg x nCtg ] of each stimulus's own-category likelihood,
        left out (bLooNoise False): M = B_k - a d d^H, with B_k = S_k/(dof_k-1) + noise_k the covariance a stimulus of
        weight 1 leaves, and x = R - mu_k(left out). From a Cholesky factor B_k = L L^H, u = L^-1 d and z = L^-1 x:
        log det M = log det B_k + log(1 - a u^H u), q = z^H z + a |u^H z|^2 / (1 - a u^H u).
        Stim weights are 0 (padding, whose value is ignored) or 1.
        """
        wc=jnp.sum(weights,axis=0)
        mu=jnp.sum(Rm*weights,axis=1)/wc if bCentered else jnp.zeros(Rm.shape[::2],dtype=Rm.dtype)
        Dv=(Rm-mu[:,None,:])*jnp.sqrt(weights)
        S=jnp.einsum('isc,jsc->cij',Dv,jnp.conj(Dv))                             # [ nCtg x nF x nF ]
        dofL=((wc-1) if bCentered else wc)-1
        B=S/dofL[:,None,None]+noiseCov.astype(S.dtype)
        d=jnp.transpose(Rm-mu[:,None,:],(2,0,1))                                 # [ nCtg x nF x nStim_Ctg ]
        c=weights*wc/(wc-weights) if bCentered else weights                     # [ nStim_Ctg x nCtg ]
        a=c/dofL
        x=jnp.transpose(R,(2,0,1))-mu.T[:,:,None]
        if bCentered:
            x=x+d*(weights/(wc-weights)).T[:,None,:]                             # R - (mu - w d/(N-w))
        L=jnp.linalg.cholesky(B)
        solve=lambda v: lax.linalg.triangular_solve(L,v,left_side=True,lower=True)   # L^-1 v, v [ nCtg x nF x nStim_Ctg ]
        u,z=solve(d),solve(x)
        uu=jnp.sum(jnp.abs(u)**2,axis=1).T                                       # [ nStim_Ctg x nCtg ]
        zz=jnp.sum(jnp.abs(z)**2,axis=1).T
        uz=jnp.abs(jnp.sum(jnp.conj(u)*z,axis=1)).T**2
        den=1-a*uu
        q=zz+a*uz/den
        logdet=2*jnp.sum(jnp.log(jnp.real(jnp.diagonal(L,axis1=-2,axis2=-1))),axis=-1)[None]+jnp.log(den)
        return q,logdet

    @staticmethod
    def _noise_all(noiseCov,noiseL):
        # [ nStim_Ctg x nCtg(k) x nCtg(i) x nF x nF ]: the left-out noise for the own category (i = k), else the category's
        eye=jnp.eye(noiseCov.shape[0],dtype=bool)[None,:,:,None,None]
        return jnp.where(eye,noiseL[:,:,None],noiseCov[None,None])

    @staticmethod
    def _loo_prior(weights):
        # log (N_k - w)/N_k: the own category's prior without the stimulus (the common 1/(N-1) cancels in the posterior)
        wc=jnp.sum(weights,axis=0)
        return jnp.log(jnp.where(weights>0,(wc-weights)/wc,1.))

    @staticmethod
    def _add_own(lAll,v):
        # add v [ nStim_Ctg x nCtg ] to lAll[l,k,k]
        return lAll + jnp.eye(lAll.shape[-1],dtype=lAll.dtype)[None]*v[:,:,None]

    @staticmethod
    def _set_own(lAll,own):
        # replace lAll[l,k,k] with own [ nStim_Ctg x nCtg ]
        return jnp.where(jnp.eye(lAll.shape[-1],dtype=bool)[None],own[:,:,None],lAll)

    #- response: which responses are decoded. returns observed, mean, variance
    @staticmethod
    def _response__mean(r,rNs,R,RNs,RVar):
        # decode mean responses; noise enters through the likelihood only (paper's approximation, Eq 9)
        return R,R,RVar

    @staticmethod
    def _response__basic(r,rNs,R,RNs,RVar):
        # decode noisy responses
        return RNs,R,RVar

    #- models
    @staticmethod
    def _model__gss(R,Rm,RVar,noiseCov,noiseCorr,weights,bLeaveOneOut=False,m=None,Y=None,refIdx=None,yRef=None):
        #: R, Rm [ nF x nStim_Ctg x nCtg ]
        #: lAll  [ nStim_Ctg x nCtg x nCtg ]
        m=Model() if m is None else m
        mu,cov=Model._ctg_stats(Rm,weights,m,Y)
        cov=cov + noiseCov

        x=jnp.transpose(R,(1,2,0))[:,:,None,:] - mu.T[None,None,:,:]         # [ nStim_Ctg x nCtg x nCtg x nF ]
        lAll=lmvn0(x,cov[None,None])
        if bLeaveOneOut and Model._loo_needs_all(m):
            muL,covL=Model._ctg_stats_loo_all(Rm,weights,m,Y)
            noiseL=Model._noise_all(noiseCov,Model._noise_loo(noiseCov,RVar,weights,m))
            lAll=Model._add_own(lmvn0(jnp.transpose(R,(1,2,0))[:,:,None,:]-muL,covL+noiseL),Model._loo_prior(weights))
        elif bLeaveOneOut and Model._loo_rank_one(m):
            q,logdet=Model._loo_quad(R,Rm,noiseCov,weights)
            lAll=Model._set_own(lAll,-q/2-R.shape[0]/2*np.log(2*np.pi)-logdet/2 + Model._loo_prior(weights))
        elif bLeaveOneOut:
            muL,covL=Model._ctg_stats_loo(Rm,weights,m)
            noiseL=Model._noise_loo(noiseCov,RVar,weights,m)
            lAll=Model._set_own(lAll,lmvn0(jnp.transpose(R,(1,2,0))-muL,covL+noiseL) + Model._loo_prior(weights))
        return Model._with_within(lAll,R,Rm,noiseCov,weights,yRef,Y,m,bLeaveOneOut,RVar)

    @staticmethod
    def _model__student(R,Rm,RVar,noiseCov,noiseCorr,weights,bLeaveOneOut=False,m=None,Y=None,refIdx=None,yRef=None):
        # multivariate t with covariance cov + noiseCov: scale = cov * (df-2)/df
        mu,cov=Model._ctg_stats(Rm,weights,m,Y)
        scale=(cov + noiseCov)*(m.df-2)/m.df
        x=jnp.transpose(R,(1,2,0))[:,:,None,:] - mu.T[None,None,:,:]
        lAll=lmvt0(x,scale[None,None],m.df)
        if bLeaveOneOut and Model._loo_needs_all(m):
            muL,covL=Model._ctg_stats_loo_all(Rm,weights,m,Y)
            noiseL=Model._noise_all(noiseCov,Model._noise_loo(noiseCov,RVar,weights,m))
            lAll=Model._add_own(lmvt0(jnp.transpose(R,(1,2,0))[:,:,None,:]-muL,(covL+noiseL)*(m.df-2)/m.df,m.df),
                                Model._loo_prior(weights))
        elif bLeaveOneOut and Model._loo_rank_one(m):
            # scale (df-2)/df M: q scales by df/(df-2), log det by nF log((df-2)/df)
            q,logdet=Model._loo_quad(R,Rm,noiseCov,weights)
            n,df=R.shape[0],m.df
            lt=(jax.scipy.special.gammaln((df+n)/2)-jax.scipy.special.gammaln(df/2)-n/2*np.log(df*np.pi)
                -(logdet+n*np.log((df-2)/df))/2-(df+n)/2*jnp.log1p(q/(df-2)))
            lAll=Model._set_own(lAll,lt + Model._loo_prior(weights))
        elif bLeaveOneOut:
            muL,covL=Model._ctg_stats_loo(Rm,weights,m)
            noiseL=Model._noise_loo(noiseCov,RVar,weights,m)
            lAll=Model._set_own(lAll,lmvt0(jnp.transpose(R,(1,2,0))-muL,(covL+noiseL)*(m.df-2)/m.df,m.df)
                                + Model._loo_prior(weights))
        return Model._with_within(lAll,R,Rm,noiseCov,weights,yRef,Y,m,bLeaveOneOut,RVar)

    @staticmethod
    def _model__circ(R,Rm,RVar,noiseCov,noiseCorr,weights,bLeaveOneOut=False,m=None,Y=None,refIdx=None,yRef=None):
        # flattened responses hold the real parts of all complex dimensions, then the imaginary parts
        h=R.shape[0]//2
        Rc=R[:h]+1j*R[h:]
        Rmc=Rm[:h]+1j*Rm[h:]
        bZero=m.circMean=='zero'
        mu,cov=Model._ctg_stats(Rmc,weights,m,Y,bCentered=not bZero)
        if bZero:
            mu=jnp.zeros_like(mu)
        # circular noise: the complex variance is the sum of the real and imaginary component (co)variances
        blocks=lambda M: M[...,:h,:h] + M[...,h:,h:]
        Nc=blocks(noiseCov)
        cov=cov + Nc.astype(cov.dtype)
        x=jnp.transpose(Rc,(1,2,0))[:,:,None,:] - mu.T[None,None,:,:]
        lAll=lcn0(x,cov[None,None])
        if bLeaveOneOut and Model._loo_needs_all(m):
            muL,covL=Model._ctg_stats_loo_all(Rmc,weights,m,Y,bCentered=not bZero)
            NcL=blocks(Model._noise_all(noiseCov,Model._noise_loo(noiseCov,RVar,weights,m)))
            lAll=Model._add_own(lcn0(jnp.transpose(Rc,(1,2,0))[:,:,None,:]-muL,covL+NcL.astype(covL.dtype)),Model._loo_prior(weights))
        elif bLeaveOneOut and Model._loo_rank_one(m):
            q,logdet=Model._loo_quad(Rc,Rmc,Nc,weights,bCentered=not bZero)
            lAll=Model._set_own(lAll,-q-h*np.log(np.pi)-logdet + Model._loo_prior(weights))
        elif bLeaveOneOut:
            muL,covL=Model._ctg_stats_loo(Rmc,weights,m,bCentered=not bZero)
            NcL=blocks(Model._noise_loo(noiseCov,RVar,weights,m))
            lAll=Model._set_own(lAll,lcn0(jnp.transpose(Rc,(1,2,0))-muL,covL+NcL.astype(covL.dtype)) + Model._loo_prior(weights))
        return Model._with_within(lAll,R,Rm,noiseCov,weights,yRef,Y,m,bLeaveOneOut,RVar)

    @staticmethod
    def _mix_fit(Rm,weights,m,state=None,bDetail=False,Y=None):
        """
        gaussian mixture of each category's mean responses Rm [ nF x nStim_Ctg x nCtg ]: weights pi [ nCtg x nMix ],
        means [ nCtg x nMix x nF ], covariances [ nCtg x nMix x nF x nF ]. state: a previous fit (pi, mu, cov) to start
        EM from (nEMWarm iterations) instead of the principal-axis initialization (nEM iterations). bDetail: also
        return a dict of the final responsibilities, component counts and scatters (for leave-one-out). Y: the category
        levels (ctgPoolWidth)
        """
        K=m.nMix
        X=jnp.transpose(Rm,(2,1,0))                                              # [ nCtg x nStim_Ctg x nF ]
        w=weights.T                                                              # [ nCtg x nStim_Ctg ]
        n=jnp.sum(w,axis=1)
        nF=X.shape[-1]
        mu0=jnp.sum(w[...,None]*X,axis=1)/n[:,None]
        D0=X-mu0[:,None]
        cov0=jnp.einsum('cs,csf,csg->cfg',w,D0,D0)/(n-1)[:,None,None]
        ridge=m.mixReg*jnp.trace(cov0,axis1=-2,axis2=-1)/nF                     # [ nCtg ]
        eye=jnp.eye(nF,dtype=X.dtype)
        pool=Model._mix_pool(cov0,n,m,Y)
        if pool is not None:
            cov0=(cov0*(n-1)[:,None,None]+pool['P'])/(n-1+pool['Q'])[:,None,None]

        def mstep(r,bBessel=False):
            # maximum likelihood during EM (so it converges to the ML mixture); the returned fit is Bessel corrected
            Nc=jnp.sum(r,axis=1)                                                 # [ nCtg x nMix ]
            Ns=jnp.where(Nc>1e-8,Nc,1.)
            mu=jnp.einsum('csk,csf->ckf',r,X)/Ns[...,None]
            D=X[:,None]-mu[:,:,None]                                             # [ nCtg x nMix x nStim_Ctg x nF ]
            S=jnp.einsum('csk,cksf,cksg->ckfg',r,D,D)
            # an empty component keeps the category covariance (its weight is 0); its denominator must stay nonzero, or
            # the masked 0/0 still makes the gradient nan (a warm start carries an empty component along)
            live=(Nc>1e-8)[...,None,None]
            den=jnp.where(Nc>1e-8,jnp.maximum(Nc-1,Nc/2),1.) if bBessel else Ns
            pi=Nc/n[:,None]
            cov=Model._mix_cov(S,den,pi,pool,m) + ridge[:,None,None,None]*eye
            cov=jnp.where(live,cov,cov0[:,None]+ridge[:,None,None,None]*eye)
            return pi,mu,cov,Nc,S

        def estep(pi,mu,cov):
            l=lmvn0(X[:,:,None,:]-mu[:,None],cov[:,None]) + jnp.log(jnp.where(pi>0,pi,1.))[:,None] \
              + jnp.where(pi>0,0.,-jnp.inf)[:,None]                               # [ nCtg x nStim_Ctg x nMix ]
            return jnp.exp(l-logsumexp(l,axis=-1,keepdims=True))*w[...,None]

        # initial responsibilities: quantile groups along the first principal axis of each category
        def init():
            v=jnp.linalg.eigh(lax.stop_gradient(cov0))[1][...,-1]                # [ nCtg x nF ]
            sc=jnp.where(w>0,jnp.einsum('csf,cf->cs',lax.stop_gradient(D0),v),jnp.inf)
            rank=jnp.argsort(jnp.argsort(sc,axis=1),axis=1)
            grp=jnp.clip(jnp.floor(rank*K/n[:,None]).astype(int),0,K-1)
            return jax.nn.one_hot(grp,K,dtype=X.dtype)*w[...,None]

        # the EM iterations are differentiated (unrolled; only the responsibilities are kept per iteration), so the
        # gradient includes how the fit moves with the filters
        em=lambda r,nIter: lax.fori_loop(0,nIter,lambda _,r: estep(*mstep(r)[:3]),r) if K>1 else r
        if state is not None:
            # a warm start keeps an emptied component empty (its weight is 0): then start again from the initialization
            warm=lambda: em(estep(*[lax.stop_gradient(a) for a in state]),m.nEMWarm)
            cold=lambda: em(init(),m.nEM)
            r=lax.cond(jnp.any(lax.stop_gradient(state[0])<=1e-8),cold,warm) if K>1 else warm()
        else:
            r=em(init(),m.nEM)
        pi,mu,cov,Nc,S=mstep(r,bBessel=True)
        if bDetail:
            # state: the maximum likelihood components (EM's fixed point), for a warm start
            return (pi,mu,cov),dict(r=r,Nc=Nc,S=S,n=n,cov0=cov0,ridge=ridge,state=mstep(r)[:3],pool=pool)
        return pi,mu,cov

    @staticmethod
    def _mix_pool(cov0,n,m,Y):
        """
        pooled statistics for 'mix' (see Model): P [ nCtg x nF x nF ] and Q [ nCtg ], the kernel-weighted scatters and
        degrees of freedom of the other categories; Scat, dof the categories' own; K the kernel with a zero diagonal. None
        without pooling
        """
        if m.ctgPoolWidth is None and not (m.covShrink>0 and m.covTarget=='pooled'):
            return None
        dof=n-1
        Scat=cov0*dof[:,None,None]
        nC=cov0.shape[0]
        if m.ctgPoolWidth is not None:
            K=Model._ctg_kernel(Y,m.ctgPoolWidth,m._Yperiod).astype(cov0.dtype)*(1-jnp.eye(nC,dtype=cov0.dtype))
        else:
            K=jnp.zeros((nC,nC),dtype=cov0.dtype)
        return dict(P=jnp.einsum('ij,jfg->ifg',K,Scat),Q=K@dof,K=K,Scat=Scat,dof=dof)

    @staticmethod
    def _mix_cov(S,den,pi,pool,m,P=None,Q=None,T=None):
        """
        component covariances [ ... x nCtg x nMix x nF x nF ] from their scatters S, denominators den and weights pi, with
        pooling over categories (P, Q [ ... x nCtg (x nF x nF) ], default pool's) and shrinkage (pooled target T,
        broadcastable to the covariances, default pool's), before the ridge
        """
        if pool is not None:
            P=pool['P'] if P is None else P
            Q=pool['Q'] if Q is None else Q
            cov=(S+pi[...,None,None]*P[...,None,:,:])/(den+pi*Q[...,None])[...,None,None]
        else:
            cov=S/den[...,None,None]
        cov=Model._low_rank(cov,m)
        if not m.covShrink>0:
            return cov
        if m.covTarget=='pooled':
            T=jnp.sum(pool['Scat'],axis=0)/jnp.sum(pool['dof']) if T is None else T
            return (1-m.covShrink)*cov + m.covShrink*T
        return (1-m.covShrink)*cov + m.covShrink*cov*jnp.eye(cov.shape[-1],dtype=cov.dtype)

    @staticmethod
    def _model__mix(R,Rm,RVar,noiseCov,noiseCorr,weights,bLeaveOneOut=False,m=None,Y=None,refIdx=None,yRef=None):
        return Model._with_within(Model._mix_likelihoods(R,Rm,noiseCov,weights,bLeaveOneOut,m,RVar=RVar,Y=Y)[0],R,Rm,noiseCov,weights,
                                  yRef,Y,m,bLeaveOneOut,RVar)

    @staticmethod
    def _logpi(pi):
        return jnp.log(jnp.where(pi>0,pi,1.)) + jnp.where(pi>0,0.,-jnp.inf)

    @staticmethod
    def _mix_likelihoods(R,Rm,noiseCov,weights,bLeaveOneOut,m,state=None,RVar=None,Y=None):
        """
        lAll[l,k,i] = log sum_c pi_ic N(R[:,l,k]; mu_ic, Sigma_ic + noiseCov_i), and the fit (the state to start the
        next EM from)
        """
        fit,det=Model._mix_fit(Rm,weights,m,state,bDetail=True,Y=Y)
        pi,mu,cov=fit
        x=jnp.transpose(R,(1,2,0))                                               # [ nStim_Ctg x nCtg x nF ]
        l=lmvn0(x[:,:,None,None,:]-mu[None,None],(cov+noiseCov[:,None])[None,None]) + Model._logpi(pi)[None,None]
        lAll=logsumexp(l,axis=-1)                                                # [ nStim_Ctg x nCtg x nCtg ]
        if bLeaveOneOut and det['pool'] is not None:
            lAll=Model._mix_loo_pooled(x,Rm,noiseCov,RVar,weights,m,mu,det)
        elif bLeaveOneOut:
            # own category without stimulus (l,k): remove its responsibility-weighted share a_c from each component
            a=jnp.transpose(det['r'],(1,0,2))                                    # [ l x k x nMix ]
            Nc,S,n=det['Nc'][None],det['S'][None],det['n'][None,:,None]
            d=jnp.transpose(Rm,(1,2,0))[:,:,None,:]-mu[None]                     # [ l x k x nMix x nF ], its mean response
            NL=Nc-a
            live=NL>1e-8
            NLs=jnp.where(live,NL,1.)
            muL=mu[None]-(a/NLs)[...,None]*d
            SL=S-(a*Nc/NLs)[...,None,None]*d[...,:,None]*d[...,None,:]
            eye=jnp.eye(x.shape[-1],dtype=x.dtype)
            ridge=det['ridge'][None,:,None,None,None]
            piL=jnp.where(live,NL,0.)/(n-weights[...,None])
            covL=Model._mix_cov(SL,jnp.where(live,jnp.maximum(NL-1,NL/2),1.),piL,None,m) + ridge*eye
            covL=jnp.where(live[...,None,None],covL,det['cov0'][None,:,None]+ridge*eye)
            noiseL=Model._noise_loo(noiseCov,RVar,weights,m)
            own=logsumexp(lmvn0(x[:,:,None,:]-muL,covL+noiseL[:,:,None]) + Model._logpi(piL),axis=-1)
            lAll=Model._set_own(lAll,own + Model._loo_prior(weights))
        return lAll,tuple(lax.stop_gradient(v) for v in det['state'])

    @staticmethod
    def _mix_loo_pooled(x,Rm,noiseCov,RVar,weights,m,mu,det):
        """
        leave-one-out 'mix' likelihoods with pooling over categories [ nStim_Ctg x nCtg(k) x nCtg(i) ]: stimulus (l,k)
        leaves its responsibility-weighted share of its own category's components, and its category's scatter, which
        enters the other categories' pooled statistics (P, Q) and the pooled shrinkage target; responsibilities are fixed
        """
        pool=det['pool']
        nC=weights.shape[1]
        E=jnp.eye(nC,dtype=x.dtype)[None,:,:,None]                              # [ 1 x k x i x 1 ] own category
        a=jnp.transpose(det['r'],(1,0,2))[:,:,None,:]*E                         # [ l x k x i x nMix ]
        Nc,S,n=det['Nc'][None,None],det['S'][None,None],det['n']
        Rt=jnp.transpose(Rm,(1,2,0))                                             # [ l x k x nF ]
        d=Rt[:,:,None,None,:]-mu[None,None]                                      # [ l x k x i x nMix x nF ]
        NL=Nc-a
        live=NL>1e-8
        NLs=jnp.where(live,NL,1.)
        muL=mu[None,None]-(a/NLs)[...,None]*d
        SL=S-(a*Nc/NLs)[...,None,None]*d[...,:,None]*d[...,None,:]
        # the stimulus's category scatter downdate, removed from the other categories' pooled statistics
        w=weights
        o=(jnp.sum(Rm*w,axis=1)/n).T                                             # [ nCtg x nF ] category means
        dc=Rt-o[None]
        dS=(w*n/(n-w))[...,None,None]*dc[...,:,None]*dc[...,None,:]             # [ l x k x nF x nF ]
        Kt=pool['K'].T[None]                                                     # [ 1 x k x i ]: K_ik
        PL=pool['P'][None,None]-Kt[...,None,None]*dS[:,:,None]                   # [ l x k x i x nF x nF ]
        QL=pool['Q'][None,None]-Kt*w[...,None]
        nL=n[None,None]-E[...,0]*w[...,None]                                     # [ l x k x i ]
        piL=jnp.where(live,NL,0.)/nL[...,None]
        T=None
        if m.covShrink>0 and m.covTarget=='pooled':
            T=((jnp.sum(pool['Scat'],axis=0)[None,None]-dS)/(jnp.sum(pool['dof'])-w)[...,None,None])[:,:,None,None]                  # [ l x k x 1 x 1 x nF x nF ]
        eye=jnp.eye(x.shape[-1],dtype=x.dtype)
        ridge=det['ridge'][None,None,:,None,None,None]
        covL=Model._mix_cov(SL,jnp.where(live,jnp.maximum(NL-1,NL/2),1.),piL,pool,m,PL,QL,T) + ridge*eye
        covL=jnp.where(live[...,None,None],covL,det['cov0'][None,None,:,None]+ridge*eye)
        noise=Model._noise_all(noiseCov,Model._noise_loo(noiseCov,RVar,weights,m))  # [ l x k x i x nF x nF ]
        lAll=logsumexp(lmvn0(x[:,:,None,None,:]-muL,covL+noise[:,:,:,None]) + Model._logpi(piL),axis=-1)
        return Model._add_own(lAll,Model._loo_prior(weights))

    @staticmethod
    def _full_terms(Rm,RVar,noiseCorr,weights):
        """
        q(Rc) [ lc x nStim_Ctg(j) x nCtg(i) ]: log N(Rc[:,l]; Rm[:,j,i], S_ji P S_ji) of observed responses Rc [ nF x lc ],
        S_ji = diag(sqrt(RVar[:,j,i])), P the noise correlation matrix (identity when noiseCorr is None); -inf at padding.
        Quadratic forms are expanded in Rc, so no [ nF x l x j x i ] tensor is formed.
        """
        nF=Rm.shape[0]
        a=1/jnp.sqrt(RVar)                                                         # [ nF x nStim_Ctg(j) x nCtg(i) ]
        m=Rm*a
        if noiseCorr is None:
            Pm=m
            logdetP=0.
        else:
            Pinv=jnp.linalg.inv(noiseCorr)
            Pm=jnp.einsum('fg,gji->fji',Pinv,m)
            logdetP=jnp.linalg.slogdet(noiseCorr)[1]
        c=(-0.5*jnp.sum(m*Pm,axis=0)
           +jnp.sum(jnp.log(a),axis=0) - 0.5*logdetP - 0.5*nF*jnp.log(2*jnp.pi)
           +jnp.where(weights>0,0.,-jnp.inf))                                      # [ nStim_Ctg(j) x nCtg(i) ]
        aPm=a*Pm
        if noiseCorr is None:
            a2=a**2
            quad=lambda Rk: jnp.einsum('fl,fji->lji',Rk**2,a2)
        else:
            B=a[:,None]*a[None,:]*Pinv[:,:,None,None]                              # [ nF x nF x j x i ]
            quad=lambda Rk: jnp.einsum('fl,gl,fgji->lji',Rk,Rk,B)
        return lambda Rc: -0.5*quad(Rc) + jnp.einsum('fl,fji->lji',Rc,aPm) + c[None]

    @staticmethod
    def _lpdf_pairs(R,Rn,Vn,noiseCorr):
        """log N(R; Rn, S P S) with S = diag(sqrt(Vn)) for paired observations R and references Rn, Vn [ nF x ... ]"""
        nF=R.shape[0]
        z=(R-Rn)/jnp.sqrt(Vn)
        if noiseCorr is None:
            quad=jnp.sum(z**2,axis=0)
            logdetP=0.
        else:
            quad=jnp.einsum('f...,fg,g...->...',z,jnp.linalg.inv(noiseCorr),z)
            logdetP=jnp.linalg.slogdet(noiseCorr)[1]
        return -0.5*quad - 0.5*jnp.sum(jnp.log(Vn),axis=0) - 0.5*logdetP - 0.5*nF*jnp.log(2*jnp.pi)

    @staticmethod
    def _full_map(R,nPer,fun,*extra):
        """
        fun(Rc,li,k,*extra_c) over the observed stimuli of each true category k, in chunks of at most _FULL_CHUNK//nPer
        observed stimuli: R [ nF x nStim_Ctg(l) x nCtg(k) ], extra arrays [ nStim_Ctg(l) x nCtg(k) x ... ]. fun returns
        arrays (or a tuple) with a leading axis of the chunk's observed stimuli; the result is [ nStim_Ctg(l) x nCtg(k) x ... ]
        """
        nF,nObs,nCtg=R.shape
        lc=min(nObs,max(1,_FULL_CHUNK//nPer))
        nCh=-(-nObs//lc)
        if nCh>1 or nCtg*nObs*nPer>_FULL_CHUNK:
            # the gradient recomputes each chunk's terms instead of storing them: memory O(chunk) instead of O(N^2)
            fun=jax.checkpoint(fun)

        def per_true_ctg(args):
            Rk,k,ex=args
            if nCh==1:
                return fun(Rk,jnp.arange(nObs),k,*ex)
            pad=nCh*lc-nObs                                                      # padded observations are dropped
            Rch=jnp.moveaxis(jnp.pad(Rk,((0,0),(0,pad))).reshape(nF,nCh,lc),1,0)
            exch=[jnp.pad(e,((0,pad),)+((0,0),)*(e.ndim-1)).reshape((nCh,lc)+e.shape[1:]) for e in ex]
            out=lax.map(lambda a: fun(a[0],a[1],k,*a[2]),(Rch,jnp.arange(nCh*lc).reshape(nCh,lc),exch))
            return tree_util.tree_map(lambda o: o.reshape((nCh*lc,)+o.shape[2:])[:nObs],out)

        out=lax.map(per_true_ctg,(jnp.moveaxis(R,-1,0),jnp.arange(nCtg),[jnp.moveaxis(e,1,0) for e in extra]))
        return tree_util.tree_map(lambda o: jnp.moveaxis(o,0,1),out)

    @staticmethod
    def _full_map_flat(R,nPer,fun,*extra):
        """
        as _full_map, with the observed stimuli of all categories in one batch (for passes whose work per observed stimulus
        is small): fun(Rc,li,kk,*extra_c) gets each observed stimulus's index li and true category kk [ lc ]
        """
        nF,nObs,nCtg=R.shape
        M=nObs*nCtg
        lc=min(M,max(1,_FULL_CHUNK//nPer))
        nCh=-(-M//lc)
        if nCh>1:
            fun=jax.checkpoint(fun)
        pad=nCh*lc-M
        Rf=jnp.pad(R.reshape(nF,M),((0,0),(0,pad)))
        li=jnp.pad(jnp.repeat(jnp.arange(nObs),nCtg),(0,pad))
        kk=jnp.pad(jnp.tile(jnp.arange(nCtg),nObs),(0,pad))
        ex=[jnp.pad(e.reshape((M,)+e.shape[2:]),((0,pad),)+((0,0),)*(e.ndim-2)) for e in extra]
        if nCh==1:
            out=fun(Rf,li,kk,*ex)
        else:
            out=lax.map(lambda a: fun(a[0],a[1],a[2],*a[3]),(jnp.moveaxis(Rf.reshape(nF,nCh,lc),1,0),li.reshape(nCh,lc),
                                                             kk.reshape(nCh,lc),[e.reshape((nCh,lc)+e.shape[1:]) for e in ex]))
            out=tree_util.tree_map(lambda o: o.reshape((nCh*lc,)+o.shape[2:]),out)
        return tree_util.tree_map(lambda o: o[:M].reshape((nObs,nCtg)+o.shape[1:]),out)

    @staticmethod
    def _latent_offsets(yRef,Y,per):
        # offsets of the reference stimuli's latent values from their category's level [ nStim_Ctg x nCtg (x nDim) ]
        return _wrap(yRef-(Y[None] if Y.ndim>1 else Y[None,:]),per)

    @staticmethod
    def _model__full(R,Rm,RVar,noiseCov,noiseCorr,weights,bLeaveOneOut=False,m=None,Y=None,refIdx=None,yRef=None):
        #: lAll[l,k,i] = log mean_j N(R[:,l,k]; Rm[:,j,i], S_ji P S_ji)   (see _full_terms)
        #: with yRef (bWithin), also Yc[l,k,i] = Y_i + the likelihood-weighted mean offset of category i's references
        wc=jnp.sum(weights,axis=0)
        terms=Model._full_terms(Rm,RVar,noiseCorr,weights)
        nCtg=weights.shape[1]
        # reference j of category i is observed stimulus refIdx[j,i] of that category (all of them, in order, by default)
        if refIdx is None:
            refIdx=jnp.broadcast_to(jnp.arange(weights.shape[0])[:,None],weights.shape)
        dY=None if yRef is None else Model._latent_offsets(yRef,Y,None if m is None else m._Yperiod)

        def chunk(Rc,li,k):
            # Rc [ nF x lc ] observed stimuli li of true category k -> [ lc x nCtg(i) ]
            q=terms(Rc)
            if bLeaveOneOut:
                self_match=(li[:,None,None]==refIdx[None]) & (jnp.arange(nCtg)==k)[None,None,:]
                q=jnp.where(self_match,-jnp.inf,q)
            lse=logsumexp(q,axis=1)
            if dY is None:
                return lse
            wq=jnp.exp(q-jnp.where(jnp.isfinite(lse),lse,0.)[:,None,:])
            return lse,jnp.einsum('lji,ji...->li...',wq,dY)

        # dividing by N_i also for the left-out category keeps the posterior (with the prior N_i/N) in Eq 5's sum form
        out=Model._full_map(R,weights.shape[0]*nCtg,chunk)
        if dY is None:
            return out-jnp.log(wc)[None,None,:]
        lse,yc=out
        return lse-jnp.log(wc)[None,None,:],yc+(Y[None,None] if Y.ndim>1 else Y[None,None,:])

    @staticmethod
    def _full_search(R,Rm,RVar,noiseCorr,weights,K,bLeaveOneOut):
        """
        for each observed stimulus, the K references of every category with the largest likelihood terms
        [ nStim_Ctg(l) x nCtg(k) x nCtg(i) x K ], as indices j into category i's stimuli; with bLeaveOneOut not the stimulus
        itself. Categories with fewer than K valid references fill up with padding (weight 0)
        """
        terms=Model._full_terms(Rm,RVar,noiseCorr,weights)
        nJ,nCtg=weights.shape
        K=min(K,nJ)

        def chunk(Rc,li,k):
            q=terms(Rc)
            if bLeaveOneOut:
                self_match=(li[:,None,None]==jnp.arange(nJ)[None,:,None]) & (jnp.arange(nCtg)==k)[None,None,:]
                q=jnp.where(self_match,-jnp.inf,q)
            return lax.top_k(jnp.swapaxes(q,1,2),K)[1]                         # [ lc x nCtg(i) x K ]

        return Model._full_map(R,weights.size,chunk)

    @staticmethod
    def _full_neighbors(R,Rm,RVar,noiseCorr,weights,nbr,tailIdx,bLeaveOneOut,yRef=None,Y=None,per=None,obsIdx=None):
        """
        full AMA from the nearest references of each category, nbr [ nStim_Ctg x nCtg(k) x nCtg(i) x K ] (_full_search), and
        a random tail tailIdx [ nTail x nCtg ] of each category's references (or None): for observed stimulus (l,k) and
        category i, sum_j N_lji = the sum over its neighbours in i + (N_i - their number) * the mean over the tail references
        of i that are not among them (nor, with bLeaveOneOut, the stimulus itself). Every tail term is at most the K-th
        neighbour's (when the neighbours are current), which bounds the variance of the estimate. lAll = log of that / N_i.
        With yRef (bWithin), also Yc as in _model__full.
        obsIdx [ nObs x nCtg ]: the observed stimuli R [ nF x nObs x nCtg ] are these of the references (a batch), for
        bLeaveOneOut; by default all of them, in order
        """
        nF,nJ,nC=Rm.shape
        wc=jnp.sum(weights,axis=0)
        K=nbr.shape[-1]
        ci=jnp.arange(nC)
        dY=None if yRef is None else Model._latent_offsets(yRef,Y,per)
        nT=0 if tailIdx is None else tailIdx.shape[0]
        if nT:
            take=lambda A: jnp.take_along_axis(A,jnp.broadcast_to(tailIdx,A.shape[:-2]+tailIdx.shape),axis=-2)
            tterms=Model._full_terms(take(Rm),take(RVar),noiseCorr,take(weights))
            dYt=None if dY is None else _take_stim(dY,tailIdx)                   # [ nTail x nCtg (x nDim) ]

        def chunk(Rc,li,kk,nb):
            # Rc [ nF x lc ] observed stimuli li of true categories kk, nb [ lc x nCtg(i) x K ] -> [ lc x nCtg(i) ]
            if obsIdx is not None:
                li=obsIdx[li,kk]
            own=ci[None,:]==kk[:,None]                                           # [ lc x i ]
            q=Model._lpdf_pairs(Rc[:,:,None,None],Rm[:,nb,ci[None,:,None]],RVar[:,nb,ci[None,:,None]],noiseCorr)   # [ lc x i x K ]
            valid=weights[nb,ci[None,:,None]]>0
            if bLeaveOneOut:
                valid=valid & ~((nb==li[:,None,None]) & own[:,:,None])
            q=jnp.where(valid,q,-jnp.inf)
            mx=jnp.max(q,axis=-1)
            if nT:
                qt=jnp.swapaxes(tterms(Rc),1,2)                                  # [ lc x i x nTail ]
                inNb=jnp.any(nb[:,:,None,:]==tailIdx.T[None,:,:,None],axis=-1)
                tv=jnp.isfinite(qt) & ~inNb
                if bLeaveOneOut:
                    tv=tv & ~((tailIdx.T[None]==li[:,None,None]) & own[:,:,None])
                qt=jnp.where(tv,qt,-jnp.inf)
                mx=jnp.maximum(mx,jnp.max(qt,axis=-1))
            m0=lax.stop_gradient(jnp.where(jnp.isfinite(mx),mx,0.))
            e=jnp.where(valid,jnp.exp(q-m0[...,None]),0.)
            L=jnp.sum(e,axis=-1)
            if dY is not None:
                dn=dY[nb,ci[None,:,None]]                                        # [ lc x i x K (x nDim) ]
                num=jnp.einsum('lik,lik...->li...',e,dn)
            if nT:
                nt=jnp.sum(tv,axis=-1)
                rest=wc[None,:]-jnp.sum(valid,axis=-1)-bLeaveOneOut*own
                sc=jnp.where(nt>0,rest/jnp.where(nt>0,nt,1),0.)
                et=jnp.where(tv,jnp.exp(qt-m0[...,None]),0.)
                L=L+sc*jnp.sum(et,axis=-1)
                if dY is not None:
                    num=num+(sc[...,None] if dY.ndim==3 else sc)*jnp.einsum('lit,ti...->li...',et,dYt)
            # no terms in a category (no valid references): -inf, with a finite gradient
            Ls=jnp.where(L>0,L,1.)
            lse=jnp.where(L>0,m0+jnp.log(Ls),-jnp.inf)
            if dY is None:
                return lse
            return lse,num/(Ls[...,None] if dY.ndim==3 else Ls)

        out=Model._full_map_flat(R,nC*(K*(nF+1)+2*nT*(K+1)),chunk,nbr)
        if dY is None:
            return out-jnp.log(wc)[None,None,:]
        lse,yc=out
        return lse-jnp.log(wc)[None,None,:],yc+(Y[None,None] if Y.ndim>1 else Y[None,None,:])


class Objective(_Static):
    """
    loss(error(estimate(posterior(log-likelihood))))

    errType
        'map' -log p(X_k|R) at the correct level  (0,1 cost / KL, Burge & Jaini 2017 Eq 9)
        'mle' -log p(R|X_k) at the correct level
        'l2'  (Xhat - X_k)^2, estType defaults to posterior 'mean' (MMSE, Eq 13-15)
        'l1'  |Xhat - X_k|,   estType defaults to posterior 'median'
    Divergences between the posterior over levels p and a target q around X_k (targetSigma):
        'xent'        cross-entropy -sum_i q_i log p_i; equals 'map' with the one-hot target
        'js'          Jensen-Shannon divergence, bounded by log 2
        'wasserstein' 1-Wasserstein (earth mover's) distance over Y, sum |F_p - F_q| dY;
                      with the one-hot target the posterior mean of |Y - X_k|
        'fisher'      Fisher divergence sum q (s_p - s_q)^2 of the scores s = d log / dY, as finite differences
                      between neighbouring levels (independent of the posterior's normalization); needs targetSigma
        With several latent dimensions (euclidean distance, wrapped on circular dimensions): 'wasserstein' is exact for the
        one-hot target (the posterior mean distance to X_k) and the debiased entropic (Sinkhorn) divergence for a gaussian
        target (otEps, nOtIter); 'fisher' needs the levels on a cartesian grid and sums the per-axis divergences.

    Several latent dimensions: 'l1' and 'l2' sum over them (L1 and squared euclidean distance), 'mean' and 'median'
    estimate each dimension (the marginal median minimizes the expected L1 distance), and the gaussian target is
    isotropic. Circular dimensions (Stim Yperiod) wrap all differences, and 'mean' is their circular mean.
    """
    _posterior_fun=_id
    _est_fun=_id
    _err_fun=_id
    _loss_fun=_id
    bPosterior=_TypeFunc(True)
    estType=_TypeFunc()
    errType=_TypeFunc()
    lossType=_TypeFunc()
    _DIVERGENCES=('xent','js','wasserstein','fisher')
    _Yperiod=None                                                               # from the Stim, set by Unit._set_geometry
    _Ygrid=None                                                                 # _latent_grid of the Stim's Y
    _bContinuous=False                                                          # Stim y: stimuli have their own latent values
    def __init__(self,errType='map',bPosterior=None,estType=None,lossType='mean',regType='None',regWeight=0.,targetSigma=None,
                 otEps=0.05,nOtIter=200,_bCopy=False):
        """
        targetSigma - target distribution over the levels for the divergence errTypes: None for one-hot at X_k, or the
                      standard deviation (in units of Y) of a gaussian target q_i ~ exp(-(Y_i - X_k)^2 / 2 targetSigma^2)
        otEps, nOtIter - 'wasserstein' with several latent dimensions and a gaussian target: the debiased entropic
                      (Sinkhorn) divergence with regularization otEps times the mean distance between levels, from nOtIter
                      Sinkhorn iterations (smaller otEps approaches the exact distance but needs more iterations)
        regType   - penalty on the filters while training (not included in Unit.loss; see Unit.penalty), in the
                    learning domain (spatial, or fourier for fourierType >= 1), over the learned coefficients:
                    'None'
                    'l1'     - sum |f| per filter: sparse filters (sparse spectra when learning in the fourier domain)
                    'smooth' - sum of squared differences between neighbouring coefficients per filter: smooth
                               spatial filters, or smooth spectra (compact spatial support) in the fourier domain
        regWeight - weight of the penalty, added to the cost as regWeight * mean penalty over filters
        """
        self.regType=regType
        self.regWeight=float(regWeight)
        if str(regType).lower() not in ('none','l1','smooth'):
            raise Exception("regType must be 'None', 'l1', or 'smooth'")

        #- errType
        if isinstance(errType,(int, float)) and not isinstance(errType,bool):
            if errType not in (1,2):
                raise Exception('numeric errType must be 1 or 2')
            errType='l' + str(int(errType))
        self.errType=errType
        self.lossType=lossType
        self.targetSigma=None if targetSigma is None else float(targetSigma)
        if self.targetSigma is not None and not self.targetSigma>0:
            raise Exception('targetSigma must be None or positive')
        self.otEps=float(otEps)
        self.nOtIter=int(nOtIter)
        if not self.otEps>0 or self.nOtIter<1:
            raise Exception('otEps must be positive and nOtIter at least 1')

        if _bCopy:
            self.bPosterior=bPosterior
            self.estType=estType
            return

        #- posterior
        if self.errType=='fisher' and self.targetSigma is None:
            raise Exception("errType='fisher' needs a gaussian target (targetSigma): the one-hot target has no score")
        if self.targetSigma is not None and self.errType not in self._DIVERGENCES:
            raise Exception('targetSigma is only used by errType ' + ', '.join(self._DIVERGENCES))
        if   self.errType == 'mle':
            if bPosterior:
                raise Exception('bPosterior must not be set for errType=mle')
            bPosterior=False
        elif self.errType == 'map' or self.errType in self._DIVERGENCES:
            if bPosterior is False:
                raise Exception('bPosterior must not be False for errType=' + self.errType)
            bPosterior=True
        elif bPosterior is None:
            bPosterior=True
        self.bPosterior=bPosterior

        #- estType
        if self.errType in ('mle','map') + self._DIVERGENCES:
            if estType is not None:
                raise Exception('estType must not be set for errType=' + self.errType)
        elif estType is None:
            estType='median' if self.errType=='l1' else 'mean'
        self.estType=estType


    def _key(self):
        return (self.errType,self.bPosterior,self.estType,self.lossType,str(self.regType).lower(),self.regWeight,self.targetSigma,
                self.otEps,self.nOtIter,self._Yperiod,self._Ygrid,self._bContinuous)

    def _err_opts(self):
        return _ErrOpts(self._Ygrid,self.targetSigma is None,self.otEps,self.nOtIter)

    def copy(self):
       return Objective(**_get_copy_dict(self),_bCopy=True)

    @partial(jit, static_argnames=['self'])
    def lrn_main(self,lAll,stimweights,yCtg,Y,priorweights=None,Yc=None):
        # the prior comes from priorweights (the training stimuli) when decoding other stimuli. Yc: the latent values that
        # stand for each candidate category, per stimulus (Model bWithin), instead of the levels Y
        prior=stimweights if priorweights is None else priorweights
        per=self._Yperiod
        return self._loss_fun(self._err_fun(self._est_fun(self._posterior_fun(lAll,prior),Y,per,Yc),yCtg,self.log_target(Y,yCtg),
                                            Y,per,self._err_opts()),stimweights)

    def log_target(self,Y,yCtg=None):
        """
        log target distribution over the levels for the divergence errTypes: [ nCtg (correct) x nCtg ], or with stimuli
        that have their own latent values (Stim y) and a gaussian target [ nStim_Ctg x nCtg x nCtg ], centered at each
        stimulus's value yCtg
        """
        if self.targetSigma is None:
            return jnp.where(jnp.eye(len(Y),dtype=bool),0.,-jnp.inf).astype(Y.dtype)
        if self._bContinuous and yCtg is not None:
            d=_wrap(Y-(yCtg[...,None] if Y.ndim==1 else yCtg[...,None,:]),self._Yperiod)
            lq=-_ysq(d,Y)/(2*self.targetSigma**2)
        else:
            lq=-_ydist2(Y,self._Yperiod)/(2*self.targetSigma**2)
        return lq - logsumexp(lq,axis=-1,keepdims=True)

    #- posterior (log domain)
    @staticmethod
    def _posterior__true(lAll,stimweights):
        # prior p(X_i) = N_i / N from the training set
        wc=jnp.sum(stimweights,axis=0)
        lp=lAll + jnp.log(wc/jnp.sum(wc))
        return lp - logsumexp(lp,axis=-1,keepdims=True)

    @staticmethod
    def _posterior__none(lAll,stimweights):
        return lAll

    #- estimation
    @staticmethod
    def _prob(lpost):
        return jnp.exp(lpost - logsumexp(lpost,axis=-1,keepdims=True))

    # estimates [ ... ] for 1-D Y, [ ... x nDim ] for several latent dimensions. Yc [ ... x nCtg (x nDim) ] (Model bWithin):
    # the latent value each candidate category stands for, per stimulus, in place of the levels Y
    @staticmethod
    def _est__none(lpost,Y,per=None,Yc=None):
        return lpost

    @staticmethod
    def _wmedian(p,y):
        # median of point masses p at y (both [ ... x n ]), per row
        order=jnp.argsort(y,axis=-1)
        cdf=jnp.cumsum(jnp.take_along_axis(p,order,axis=-1),axis=-1)
        return jnp.vectorize(lambda c,v: jnp.interp(0.5,c,v),signature='(n),(n)->()')(cdf,jnp.take_along_axis(y,order,axis=-1))

    @staticmethod
    def _est__median(lpost,Y,per=None,Yc=None):
        # (marginal) median: cdf over latent values in ascending order (Y need not be sorted)
        p=Objective._prob(lpost)
        if Yc is not None:
            return Objective._median_within(p,Y,per,Yc)
        def median(y):
            order=jnp.argsort(y)
            cdf=jnp.cumsum(p[...,order],axis=-1)
            return jnp.vectorize(lambda c: jnp.interp(0.5,c,y[order]),signature='(n)->()')(cdf)
        def circ_median(y,P):
            # the circular median m minimizes the expected wrapped distance, and the diameter through m halves the
            # probability: find it among the levels and their antipodes (no gradient), cut the circle at its antipode,
            # and take the median of the unwrapped distribution
            cand=jnp.concatenate((y,y+P/2))
            dist=jnp.abs(_wrap(y[:,None]-cand[None,:],(P,)))                     # [ nCtg x 2 nCtg ]
            m0=lax.stop_gradient(cand[jnp.argmin(p@dist,axis=-1)])                 # [ ... ]
            yu=m0[...,None]+_wrap(y-m0[...,None],(P,))                             # [ ... x nCtg ]
            order=jnp.argsort(yu,axis=-1)
            cdf=jnp.cumsum(jnp.take_along_axis(p,order,axis=-1),axis=-1)
            ys=jnp.take_along_axis(yu,order,axis=-1)
            return _wrap(jnp.vectorize(lambda c,v: jnp.interp(0.5,c,v),signature='(n),(n)->()')(cdf,ys),(P,))
        if Y.ndim==1:
            return median(Y) if per is None else circ_median(Y,per[0])
        pd=per if per is not None else (None,)*Y.shape[1]
        return jnp.stack([median(Y[:,d]) if pd[d] is None else circ_median(Y[:,d],pd[d]) for d in range(Y.shape[1])],axis=-1)

    @staticmethod
    def _median_within(p,Y,per,Yc):
        # median of point masses at the within-category values Yc; on circular dimensions cut the circle opposite the
        # circular mean and take the median of the unwrapped values
        def one(y,P):
            if P is None:
                return Objective._wmedian(p,y)
            m0=lax.stop_gradient(jnp.angle(jnp.sum(p*jnp.exp(2j*jnp.pi*y/P),axis=-1))*P/(2*jnp.pi))
            return _wrap(Objective._wmedian(p,m0[...,None]+_wrap(y-m0[...,None],(P,))),(P,))
        if Y.ndim==1:
            return one(Yc,None if per is None else per[0])
        pd=per if per is not None else (None,)*Y.shape[1]
        return jnp.stack([one(Yc[...,d],pd[d]) for d in range(Y.shape[1])],axis=-1)

    @staticmethod
    def _est__mean(lpost,Y,per=None,Yc=None):
        # posterior mean; on circular dimensions the circular mean
        p=Objective._prob(lpost)
        if Yc is None:
            avg=lambda V: p @ V
            V=Y
        else:
            avg=lambda V: jnp.sum(p*V,axis=-1) if Y.ndim==1 else jnp.sum(p[...,None]*V,axis=-2)
            V=Yc
        m=avg(V)
        if per is None:
            return m
        P,bC=_period_arrays(per,Y.dtype)
        if Y.ndim==1:
            P,bC=P[0],bC[0]
        # zero phase on linear dimensions keeps angle() away from 0, where its gradient is not finite
        mc=jnp.angle(avg(jnp.exp(1j*jnp.where(bC,2*jnp.pi*V/P,0.))))*P/(2*jnp.pi)
        return jnp.where(bC,mc,m)

    @staticmethod
    def _est__mode(lpost,Y,per=None,Yc=None):
        # MAP estimate; piecewise constant, so provides no gradient
        i=jnp.argmax(lpost,axis=-1)
        if Yc is None:
            return Y[i]
        if Y.ndim==1:
            return jnp.take_along_axis(Yc,i[...,None],axis=-1)[...,0]
        return jnp.take_along_axis(Yc,i[...,None,None],axis=-2)[...,0,:]

    @staticmethod
    def _est__cmean(lpost,Y,per=None,Yc=None):
        # circular mean, Y in radians (see also Stim Yperiod, with which 'mean' is circular)
        p=Objective._prob(lpost)
        return jnp.angle(jnp.sum(p*jnp.exp(1j*Yc),axis=-1) if Yc is not None else p @ jnp.exp(1j*Y))

    #- error
    @staticmethod
    def _err__mle(lAll,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        return -Objective._at_correct(lAll)

    @staticmethod
    def _err__map(lpost,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        return -Objective._at_correct(lpost)

    @staticmethod
    def _err__l1(yHat,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        d=jnp.abs(_wrap(yHat-yCtg,per))
        return d if Y is None or Y.ndim==1 else jnp.sum(d,axis=-1)

    @staticmethod
    def _err__l2(yHat,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        d=jnp.abs(_wrap(yHat-yCtg,per))
        return d**2 if Y is None or Y.ndim==1 else jnp.sum(d**2,axis=-1)

    # divergences: lpost [ nStim_Ctg x nCtg x nCtg ] against the target row of each correct level, lQ [ nCtg x nCtg ]
    @staticmethod
    def _err__xent(lpost,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        q=jnp.exp(lQ)
        return -jnp.sum(jnp.where(q>0,q*lpost,0),axis=-1)

    @staticmethod
    def _err__js(lpost,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        # 0 log 0 = 0; lQ is -inf off the one-hot target, so both logs go through where
        p,q=jnp.exp(lpost),jnp.exp(lQ)
        lm=jnp.logaddexp(lpost,lQ)-jnp.log(2.)
        lms=jnp.where((p>0)|(q>0),lm,0)
        tp=jnp.where(p>0,p*(jnp.where(p>0,lpost,0)-lms),0)
        tq=jnp.where(q>0,q*(jnp.where(q>0,lQ,0)-lms),0)
        return 0.5*jnp.sum(tp+tq,axis=-1)

    @staticmethod
    def _err__wasserstein(lpost,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        if Y.ndim>1:
            return Objective._wasserstein_nd(lpost,lQ,Y,per,opts or _ErrOpts())
        # both distributions sit on the levels, so their cdfs are constant between neighbouring levels
        y,order,dY=Objective._levels(Y,per)
        dF=jnp.cumsum(jnp.exp(lpost)[...,order] - jnp.exp(lQ)[...,order],axis=-1)
        if per is None:
            return jnp.sum(jnp.abs(dF[...,:-1])*dY,axis=-1)
        # on a circle the cdf difference is defined up to a constant, W1 = min_c sum |dF - c| dY: c is the dY-weighted median
        o=jnp.argsort(dF,axis=-1)
        cw=jnp.cumsum(dY[o],axis=-1)
        c=jnp.take_along_axis(jnp.take_along_axis(dF,o,axis=-1),jnp.argmax(cw>=cw[...,-1:]/2,axis=-1)[...,None],axis=-1)
        return jnp.sum(jnp.abs(dF-c)*dY,axis=-1)

    @staticmethod
    def _wasserstein_nd(lpost,lQ,Y,per,opts):
        # euclidean ground distance between levels (wrapped on circular dimensions)
        D=jnp.sqrt(_ydist2(Y,per))
        if opts.bOneHot:
            # all the target's mass sits at X_k: the posterior mean distance to it, exactly
            return jnp.sum(jnp.exp(lpost)*D[None],axis=-1)
        n=D.shape[0]
        eps=opts.otEps*jnp.sum(D)/(n*(n-1))
        ot=lambda la,lb: _sinkhorn_value(la,lb,D,eps,opts.nOtIter)
        # Sinkhorn divergence: 0 at the target (one target per correct level, or per stimulus with Stim y)
        if lQ.ndim<lpost.ndim:
            lQ=lQ[None]
        return ot(lpost,lQ) - 0.5*ot(lpost,lpost) - 0.5*ot(lQ,lQ)

    @staticmethod
    def _err__fisher(lpost,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        # scores at the midpoints between neighbouring levels (on a circle also the last and first), weighted by the
        # target there
        if Y.ndim>1:
            return Objective._fisher_grid(lpost,lQ,Y,per,opts)
        y,order,dY=Objective._levels(Y,per)
        lp,lq=lpost[...,order],lQ[...,order]
        nxt=lambda a: jnp.roll(a,-1,axis=-1) if per is not None else a[...,1:]
        cur=lambda a: a if per is not None else a[...,:-1]
        sp=(nxt(lp)-cur(lp))/dY
        sq=(nxt(lq)-cur(lq))/dY
        q=jnp.exp(lq)
        w=(nxt(q)+cur(q))/2
        return jnp.sum(w*(sp-sq)**2,axis=-1)/jnp.sum(w,axis=-1)

    @staticmethod
    def _fisher_grid(lpost,lQ,Y,per,opts):
        # levels on a cartesian grid: finite-difference scores along each axis (wrapping on circular axes); the sum over
        # axes of the target-weighted mean squared score difference along that axis (the 1-D formula per axis)
        if opts is None or opts.grid is None:
            raise Exception("errType='fisher' with several latent dimensions needs the levels on a cartesian grid")
        shape,order=opts.grid
        order=jnp.asarray(order)
        lp=lpost[...,order].reshape(lpost.shape[:-1]+shape)
        lq=lQ[...,order].reshape(lQ.shape[:-1]+shape)
        Yg=Y[order].reshape(shape+(Y.shape[1],))
        nd=len(shape)
        tot=0.
        for d in range(nd):
            if shape[d]<2:                                                   # no neighbours along this axis
                continue
            ax=-nd+d
            bCirc=per is not None and per[d] is not None
            if bCirc:
                nxt=lambda a: jnp.roll(a,-1,axis=ax)
                cur=lambda a: a
            else:
                nxt=lambda a: lax.slice_in_dim(a,1,None,axis=a.ndim+ax)
                cur=lambda a: lax.slice_in_dim(a,0,a.shape[a.ndim+ax]-1,axis=a.ndim+ax)
            # spacing along axis d (the same at every position of the other axes); circular axes span less than a period
            # (Unit._check), so ascending values are in circular order
            yd=Yg[(0,)*d+(slice(None),)+(0,)*(nd-d-1)+(d,)]
            dY=jnp.diff(jnp.concatenate((yd,yd[:1]+per[d]))) if bCirc else jnp.diff(yd)
            dY=dY.reshape((-1,)+(1,)*(nd-d-1))
            sp=(nxt(lp)-cur(lp))/dY
            sq=(nxt(lq)-cur(lq))/dY
            q=jnp.exp(lq)
            w=(nxt(q)+cur(q))/2
            ax_sum=tuple(range(-nd,0))
            tot=tot+jnp.sum(w*(sp-sq)**2,axis=ax_sum)/jnp.sum(w,axis=ax_sum)
        return tot

    @staticmethod
    def _levels(Y,per):
        """
        a single latent dimension in ascending order: positions, order, and the spacings to the next level [ nCtg-1 ],
        or on a circle positions in [0, P) and spacings [ nCtg ] including the one from the last level around to the first
        """
        if Y.ndim>1:
            raise Exception("errType 'wasserstein' and 'fisher' need a single latent dimension")
        if per is None:
            order=jnp.argsort(Y)
            return Y[order],order,jnp.diff(Y[order])
        P=per[0]
        y=jnp.mod(Y,P)
        order=jnp.argsort(y)
        y=y[order]
        return y,order,jnp.diff(jnp.concatenate((y,y[:1]+P)))

    #- loss (padding in the category-grouped stimuli has weight 0)
    @staticmethod
    def _loss__mean(err,stimweights):
        return jnp.sum(jnp.where(stimweights>0,err,0)*stimweights)/jnp.sum(stimweights)

    @staticmethod
    def _loss__median(err,stimweights):
        return jnp.nanmedian(jnp.where(stimweights>0,err,jnp.nan))

    #- helpers
    @staticmethod
    def _at_correct(inAll):
        # [ nStim_Ctg x nCtg x nCtg ] -> [ nStim_Ctg x nCtg ]
        return jnp.diagonal(inAll, axis1=-2, axis2=-1)


def _ama_sgd(step0,stepMin,stepDecay):
    """
    the AMA-SGD step of Burge & Jaini (2017) and burgelab/AMA (gradSGD.m, updateSGD.m, amaR01sgdObjFunc.m) as an optax
    transformation: each filter moves a fixed distance eps along its unit-normalized (tangent-plane) gradient, with eps
    = max(stepMin, step0*(1-stepDecay)^t) at iteration t. Filters are the last axis of params['f']; other parameters
    (pooling weights) take the same step along their normalized gradient. Accepting a step only when it does not raise
    the batch cost happens in Optimizer._run_chunk.
    """
    def unit(g,bPerFilter):
        axes=tuple(range(g.ndim-1)) if bPerFilter and g.ndim>1 else None
        n=jnp.sqrt(jnp.sum(jnp.abs(g)**2,axis=axes,keepdims=bPerFilter and g.ndim>1))
        return g/jnp.where(n>0,n,1)

    def init(params):
        return {'count':jnp.zeros((),dtype=jnp.int32)}

    def update(grads,state,params=None):
        eps=jnp.maximum(stepMin,step0*(1-stepDecay)**state['count'])
        updates={k: -(eps*unit(g,k=='f')).astype(g.dtype) for k,g in grads.items()}
        return updates,{'count':state['count']+1}

    return optax.GradientTransformation(init,update)


class Optimizer():
    def __init__(self,optimizerType='adam',projectionType=['l2_sphere',1],lRate0=1e-1,nIterMax=1000,f0_jxrand_fun=['ball',1],
                 batchSize=None,nStepsPerChunk=100,bVerbose=True,nBatchMinCtg=2,patience=None,stepMin=0.,stepDecay=0.01,
                 tolFun=None,tolX=None,lbfgsMemory=10):
        """
        optimizerType  - any optax optimizer (e.g. 'adam', 'sgd'), with learning rate lRate0 (a number or an optax
                         schedule), or 'ama_sgd': the step of Burge & Jaini (2017) and burgelab/AMA. Each filter moves
                         lRate0*(1-stepDecay)^t (at least stepMin) along its unit-normalized tangent-plane gradient, and a
                         step is kept only if it does not raise the cost of the batch it was computed on (evaluated again,
                         with the same noise). Needs projectionType 'l2_sphere'. With batchSize, this is AMA-SGD.
                         'lbfgs': limited-memory BFGS with a zoom line search (optax.lbfgs, lbfgsMemory pairs), the
                         quasi-Newton method of matlab's fminunc/fmincon: full batch (not with batchSize), and lRate0 is
                         not used. The cost is evaluated at the normalized filters, so the unit-norm constraint is exact
                         and the line search moves along the sphere's tangent space.
        stepMin, stepDecay - 'ama_sgd' only: the smallest step, and the fraction the step shrinks each iteration (0.01:
                         1% per iteration, as in the paper)
        tolFun, tolX   - fmincon-like stopping tolerances, checked after every chunk (nStepsPerChunk iterations): stop when
                         the cost changed by less than tolFun*(1+|cost|) over the chunk (first vs last iteration), or when
                         no parameter moved by more than tolX. stop_reason records why training stopped ('nIterMax',
                         'patience', 'tolFun', 'tolX'). With batches or sampled noise the cost is noisy, so tolFun needs a
                         matching size
        batchSize      - None for full-batch learning, or the approximate number of stimuli per iteration (AMA-SGD,
                         Burge & Jaini 2017). Each iteration draws a new random batch, stratified so every category keeps
                         its share of the training set (the prior) with at least nBatchMinCtg stimuli. Posteriors are computed
                         against the batch, so full AMA costs O(batchSize^2) per iteration. loss_hist holds batch costs.
        nStepsPerChunk - iterations compiled into one lax.scan; the loss is reported once per chunk
        nBatchMinCtg   - minimum stimuli per category in a batch (AMA-Gauss needs more than the number of response dimensions)
        patience       - early stopping, when training is given validation stimuli (stimVal in Unit.train_*): the cost on
                         them is computed after every chunk (nStepsPerChunk iterations), training stops after this many
                         chunks without improvement (None = never stop early), and the filters with the lowest validation
                         cost are kept. val_hist holds the validation costs, best_step the kept iteration.
        """
        # filters are constrained to unit length, ||f||=1 (Burge & Jaini 2017)
        self.optimizerType=optimizerType
        self.projectionType=list(projectionType)
        self.lRate0=lRate0
        self.nIterMax=nIterMax
        self.f0_jxrand_fun=list(f0_jxrand_fun)
        self._f0_jxrand_fun=list(f0_jxrand_fun)
        if isinstance(f0_jxrand_fun[0],str):
            self._f0_jxrand_fun[0]=getattr(jxrandom,f0_jxrand_fun[0])
        self.batchSize=batchSize
        self.nBatchMinCtg=nBatchMinCtg
        self.nStepsPerChunk=nStepsPerChunk
        self.bVerbose=bVerbose
        self.patience=patience
        self.stepMin=float(stepMin)
        self.stepDecay=float(stepDecay)
        self.tolFun=None if tolFun is None else float(tolFun)
        self.tolX=None if tolX is None else float(tolX)
        self.lbfgsMemory=int(lbfgsMemory)
        if self.optimizerType=='lbfgs' and batchSize is not None:
            raise Exception("optimizerType='lbfgs' is a full-batch method; batchSize must be None")
        if self.optimizerType=='ama_sgd':
            if self.projectionType[0]!='l2_sphere':
                raise Exception("optimizerType='ama_sgd' steps on the unit sphere: projectionType must be 'l2_sphere'")
            if not isinstance(lRate0,(int,float)) or not lRate0>0 or not 0<=self.stepDecay<1 or self.stepMin<0:
                raise Exception("optimizerType='ama_sgd' needs a positive number lRate0, 0 <= stepDecay < 1 and stepMin >= 0")

    @property
    def _bAccept(self):
        # keep a step only if it does not raise the batch cost (AMA-SGD)
        return self.optimizerType=='ama_sgd'

    @property
    def _bLBFGS(self):
        return self.optimizerType=='lbfgs'

    def copy(self):
        return Optimizer(**_get_copy_dict(self,_OPT_EXCL))

    @property
    def _projection(self):
        #l2_sphere, l2_ball, l1_all, l1_sphere
        return getattr(optax.projections,'projection_' + self.projectionType[0])

    @property
    def _projection_params(self):
        return tuple(self.projectionType[1:])

    @property
    def optimizer(self):
        # the optax optimizer (or the AMA-SGD step) as a function of the learning rate
        if self.optimizerType=='ama_sgd':
            return lambda lRate: _ama_sgd(float(lRate),self.stepMin,self.stepDecay)
        if self._bLBFGS:
            return lambda lRate: optax.lbfgs(memory_size=self.lbfgsMemory)
        return getattr(optax,self.optimizerType)

    @property
    def tx(self):
        # one optax transformation per setting, shared by all Optimizers: a new transformation object would compile the
        # training step again (e.g. for every new Unit or cross-validation fold)
        if self.optimizerType=='ama_sgd':
            key=(self.optimizerType,self.lRate0,self.stepMin,self.stepDecay)
            if key not in _TX_CACHE:
                _TX_CACHE[key]=self.optimizer(self.lRate0)
            return _TX_CACHE[key]
        key=(self.optimizerType,self.lRate0) if not self._bLBFGS else (self.optimizerType,self.lbfgsMemory)
        if key not in _TX_CACHE:
            _TX_CACHE[key]=self.optimizer(self.lRate0)
        return _TX_CACHE[key]

    @staticmethod
    @partial(jit, static_argnames=['proj_fun','proj_params'])
    def insert_project_extract(params,prepped,index,proj_fun,proj_params):
        out=dict(params)
        fNew=prepped.at[index].set(params['f'])

        # constraint applies to each filter (last axis) separately
        vectorized_proj_fun = vmap(lambda f: proj_fun({'f': f}, *proj_params)['f'], in_axes=-1, out_axes=-1)
        fNew=vectorized_proj_fun(fNew)

        out['f']=fNew[index]
        return out

    #- batches
    def _batch_plan(self,weights):
        # number of stimuli drawn from each category per iteration, proportional to the category's size
        counts=np.asarray(weights).sum(0).astype(int)
        m=np.maximum(self.nBatchMinCtg,np.round(self.batchSize*counts/counts.sum())).astype(int)
        m=np.minimum(m,counts)
        mMax=int(m.max())
        mask=(np.arange(mMax)[:,None] < m[None,:]).astype(np.asarray(weights).dtype)  # [ mMax x nCtg ]
        return mMax,jnp.asarray(mask)

    @staticmethod
    def _batch_index(rng,mMax,mask,stimweights):
        # a random subset of the valid (non-padding) stimuli within each category: indices [ mMax x nCtg ] and weights
        score=jxrandom.uniform(rng,stimweights.shape) + jnp.where(stimweights>0,0.,jnp.inf)
        idx=jnp.argsort(score,axis=0)[:mMax]
        return idx,jnp.take_along_axis(stimweights,idx,axis=0)*mask

    @staticmethod
    def _sample_batch(rng,mMax,mask,stimval,stimweights,yCtg):
        idx,w=Optimizer._batch_index(rng,mMax,mask,stimweights)
        val=jnp.take_along_axis(stimval,jnp.broadcast_to(idx,stimval.shape[:-2]+idx.shape),axis=-2)
        y=_take_stim(yCtg,idx)
        return val,w,y

    #- learning
    @staticmethod
    def _tangent(g,f):
        # remove the radial component of each filter's gradient (last axis = filters; Burge & Jaini 2017, Eq 19).
        # Without this, optimizers that rescale coordinates (e.g. adam) turn the large radial component, which the
        # unit-sphere projection then discards, into tangent steps that can increase the cost.
        axes=tuple(range(f.ndim-1))
        radial=jnp.sum(jnp.real(jnp.conj(f)*g),axis=axes,keepdims=True)/jnp.sum(jnp.abs(f)**2,axis=axes,keepdims=True)
        return g-radial*f

    @staticmethod
    @partial(jit, static_argnames=['tx','loss_fun','proj_fun','proj_params','nSteps','mMax','bTangent','bAccept','bState','state_fun',
                                   'bLBFGS'])
    def _run_chunk(tx,loss_fun,proj_fun,proj_params,nSteps,mMax,bTangent,bAccept,bState,nActive,
                   params,opt_state,rng,prepped,prepped_exp,index,index_exp,batch_mask,stimval,stimweights,yCtg,Y,state_fun=None,
                   bLBFGS=False):
        if state_fun is not None:
            # a model state recomputed from the parameters at the start of every chunk (e.g. the nearest references)
            opt_state=(opt_state[0],state_fun(params,prepped,index,stimval,stimweights))
        # batches with a per-chunk state (nNeighbors): the loss gets all stimuli, which are the references, and decodes
        # the batch (indices and weights passed with the state)
        bRefBatch=state_fun is not None and mMax is not None

        def body(carry,i):
            params,opt_state,rng=carry
            rng,rng_batch,rng_key=jxrandom.split(rng,3)
            if mMax is None or bRefBatch:
                val,w,y=stimval,stimweights,yCtg
            else:
                val,w,y=Optimizer._sample_batch(rng_batch,mMax,batch_mask,stimval,stimweights,yCtg)

            # L-BFGS: the cost of the normalized filters (scale invariant), so its gradient and line search stay tangent
            norm=(lambda prm: Optimizer.insert_project_extract(prm,prepped_exp,index_exp,proj_fun,proj_params)) if bLBFGS else (lambda prm: prm)
            if bState:
                # a model state (e.g. the previous mixture fit) goes in with the parameters and comes out updated
                ost,mst=opt_state
                ext=(mst,)+Optimizer._batch_index(rng_batch,mMax,batch_mask,stimweights) if bRefBatch else mst
                cost=lambda prm: loss_fun(norm(prm),rng_key,prepped,index,val,w,y,Y,ext)
                (loss_value,mnew),grads=value_and_grad(cost,has_aux=True)(params)
                if bRefBatch:
                    mnew=mst
                value_fn=lambda prm: cost(prm)[0]
            else:
                ost=opt_state
                cost=lambda prm: loss_fun(norm(prm),rng_key,prepped,index,val,w,y,Y)
                loss_value,grads=value_and_grad(cost)(params)
                value_fn=cost
            # complex params: jax returns the conjugate of the ascent direction
            grads=tree_util.tree_map(jnp.conjugate,grads)
            if bTangent and not bLBFGS:
                grads=dict(grads,f=Optimizer._tangent(grads['f'],params['f']))
            if bLBFGS:
                # the line search's step sizes are float64 under jax x64: its trial points keep the parameters' precision
                cast=lambda prm: tree_util.tree_map(lambda a,q: a.astype(q.dtype),prm,params)
                updates,new_ost=tx.update(grads,ost,params,value=loss_value,grad=grads,value_fn=lambda prm: value_fn(cast(prm)))
                updates=cast(updates)
            else:
                updates,new_ost=tx.update(grads,ost,params)
            new_state=(new_ost,mnew) if bState else new_ost
            new_params=Optimizer.insert_project_extract(optax.apply_updates(params,updates),prepped_exp,index_exp,proj_fun,proj_params)
            if bAccept:
                # AMA-SGD: keep the step only if the cost of this batch (same noise) does not increase
                new_value=cost(new_params)
                ok=(new_value[0] if bState else new_value)<=loss_value
                new_params=tree_util.tree_map(lambda a,b: jnp.where(ok,a,b),new_params,params)

            # iterations past nIterMax in the last chunk leave the state unchanged
            keep=lambda new,old: tree_util.tree_map(lambda a,b: jnp.where(i<nActive,a,b),new,old)
            return (keep(new_params,params),keep(new_state,opt_state),rng),loss_value

        (params,opt_state,rng),losses=lax.scan(body,(params,opt_state,rng),jnp.arange(nSteps))
        return params,opt_state,rng,losses

    @staticmethod
    @partial(jit, static_argnames=['tx','loss_fun','nSteps','mMax','bLBFGS'])
    def _run_generated_chunk(tx,loss_fun,nSteps,mMax,nActive,params,opt_state,rng,batch_mask,stimval,stimweights,yCtg,Y,bLBFGS=False):
        # as _run_chunk, for parameters of generated filters (no filter insertion, projection or tangent step)
        def body(carry,i):
            params,opt_state,rng=carry
            rng,rng_batch,rng_key=jxrandom.split(rng,3)
            if mMax is None:
                val,w,y=stimval,stimweights,yCtg
            else:
                val,w,y=Optimizer._sample_batch(rng_batch,mMax,batch_mask,stimval,stimweights,yCtg)
            loss_value,grads=value_and_grad(loss_fun)(params,rng_key,val,w,y,Y)
            grads=tree_util.tree_map(jnp.conjugate,grads)
            if bLBFGS:
                cast=lambda prm: tree_util.tree_map(lambda a,q: a.astype(q.dtype),prm,params)
                updates,new_state=tx.update(grads,opt_state,params,value=loss_value,grad=grads,
                                            value_fn=lambda prm: loss_fun(cast(prm),rng_key,val,w,y,Y))
                updates=cast(updates)
            else:
                updates,new_state=tx.update(grads,opt_state,params)
            new_params=optax.apply_updates(params,updates)
            keep=lambda new,old: tree_util.tree_map(lambda a,b: jnp.where(i<nActive,a,b),new,old)
            return (keep(new_params,params),keep(new_state,opt_state),rng),loss_value

        (params,opt_state,rng),losses=lax.scan(body,(params,opt_state,rng),jnp.arange(nSteps))
        return params,opt_state,rng,losses

    def minimize_generated(self,params,rng,stim,loss_fun,val_fun=None):
        """minimize loss_fun (a _GeneratedLoss) over params, in the same chunks, batches and early stopping as minimize"""
        if self.optimizerType=='ama_sgd':
            raise Exception("optimizerType='ama_sgd' learns unit-norm filters directly; use another optimizer for generated filters")
        tx=self.tx
        mMax,batch_mask=(None,None) if self.batchSize is None else self._batch_plan(stim.weights)
        nSteps=max(1,min(self.nStepsPerChunk,self.nIterMax))
        run=lambda params,opt_state,rng,nActive: self._run_generated_chunk(tx,loss_fun,nSteps,mMax,nActive,params,opt_state,rng,
                                                                          batch_mask,stim.val,stim.weights,stim.yCtg,stim.Y,
                                                                          self._bLBFGS)
        params,_,rng=self._loop(run,nSteps,params,tx.init(params),rng,val_fun)
        return params,rng

    def _loop(self,run,nSteps,params,opt_state,rng,val_fun):
        # chunks of nSteps iterations (a fixed length keeps one compiled trace; the last chunk masks its extra
        # iterations), with early stopping on val_fun: best_step is the number of iterations of the kept parameters
        self.loss_hist=[]
        self.val_hist=[]
        self.best_step=None
        self.stop_reason='nIterMax'
        best=None
        nBad=0
        step=0
        tolFun,tolX=getattr(self,'tolFun',None),getattr(self,'tolX',None)
        while step < self.nIterMax:
            nActive=min(nSteps,self.nIterMax-step)
            prev=params
            params,opt_state,rng,losses=run(params,opt_state,rng,nActive)
            chunk=np.asarray(losses)[:nActive]
            self.loss_hist.extend(chunk.tolist())
            step+=nActive

            if self.bVerbose:
                print(f'step {step-1}, loss: {self.loss_hist[-1]}')

            if val_fun is not None:
                v=float(val_fun(params))
                self.val_hist.append(v)
                if best is None or v < best[0]:
                    best=(v,params,opt_state,step)
                    nBad=0
                else:
                    nBad+=1
                if self.bVerbose:
                    print(f'step {step-1}, validation loss: {v}')
                if self.patience is not None and nBad >= self.patience:
                    self.stop_reason='patience'
                    break

            # fmincon-like tolerances on the cost change and the parameter step over the chunk
            if tolFun is not None and len(chunk)>1 and abs(chunk[-1]-chunk[0])<=tolFun*(1+abs(chunk[-1])):
                self.stop_reason='tolFun'
                break
            if tolX is not None:
                dx=max(float(jnp.max(jnp.abs(a-b))) for a,b in zip(tree_util.tree_leaves(params),tree_util.tree_leaves(prev)))
                if dx<=tolX:
                    self.stop_reason='tolX'
                    break

        if best is not None:
            _,params,opt_state,self.best_step=best
        return params,opt_state,rng

    def minimize(self,f0,rng,stim,filter,loss_fun,opt_state=None,extra_params=None,val_fun=None,mstate0=None,state_fun=None):
        tx=self.tx
        proj_fun=self._projection
        proj_params=self._projection_params
        bTangent=self.projectionType[0]=='l2_sphere'
        index=tuple(jnp.asarray(i) for i in filter._insert_index_jx)
        index_exp=tuple(jnp.asarray(i) for i in filter._insert_index_exp_jx)

        # f0 (and any other learned parameters, e.g. pooling weights)
        params=self.insert_project_extract({'f':f0,**(extra_params or {})},filter.prepped_exp_jx,index_exp,proj_fun,proj_params)
        if opt_state is None:
            opt_state=tx.init(params)

        if self.batchSize is None:
            mMax,batch_mask=None,None
        else:
            mMax,batch_mask=self._batch_plan(stim.weights)

        nSteps=max(1,min(self.nStepsPerChunk,self.nIterMax))
        # mstate0: an initial model state to carry through training (returned states are the optax part only); state_fun:
        # a model state recomputed at the start of every chunk instead
        if state_fun is not None:
            mstate0=state_fun(params,filter.prepped_jx,index,stim.val,stim.weights)
        bState=mstate0 is not None
        run=lambda params,opt_state,rng,nActive: self._run_chunk(tx,loss_fun,proj_fun,proj_params,nSteps,mMax,bTangent,self._bAccept,
                                                                bState,nActive,params,opt_state,rng,
                                                                filter.prepped_jx,filter.prepped_exp_jx,index,index_exp,batch_mask,
                                                                stim.val,stim.weights,stim.yCtg,stim.Y,state_fun,self._bLBFGS)
        params,opt_state,rng=self._loop(run,nSteps,params,(opt_state,mstate0) if bState else opt_state,rng,val_fun)
        return params,(opt_state[0] if bState else opt_state),rng

class Unit(_Static):
    def __init__(self,stim,nrn,model,objective,optimizer=None,seed=None,rng=None,rng_last=None,name=None):
        """name: a label for this unit, used as the title of its figures (e.g. 'mixed amplitude / natural / phase pooled')"""
        self.name=name
        self.nrn=nrn
        self.stim=stim
        self.stim_full=stim
        self.model=model
        self.objective=objective
        self.optimizer=optimizer
        self.opt_state=None
        self.restart_costs=None
        self.pool_p=None          # pooling-weight parameters for Nrn readoutType (see Nrn.pool_weights)
        self.nrn_p=None           # other learned response-model parameters (Nrn bBias, bLearnNormPool, readoutType 'linear')
        self.train_log=[]         # training calls, for config() / from_config()
        self._set_geometry()

        if seed is None:
            seed=666
        self.seed=seed

        if rng is None:
            self.rng=jxrandom.key(seed)
        else:
            self.rng=rng

        self.rng_last=rng_last

    def _key(self):
        return (self.nrn._key(),self.model._key(),self.objective._key())

    def _set_geometry(self):
        # the latent geometry (circular dimensions) of the stimuli, used by the likelihood pooling and the objective. A Model
        # or Objective shared with a unit of another geometry is copied, so that unit keeps its own
        per=getattr(self.stim_full,'Yperiod',None)
        grid=_latent_grid(self.stim_full.Y) if hasattr(self.stim_full,'Y') else None
        bCont=bool(getattr(self.stim_full,'bContinuous',False))
        for name in ('model','objective'):
            obj=getattr(self,name)
            if getattr(obj,'_bGeometrySet',False) and (obj._Yperiod!=per or (name=='objective' and (obj._Ygrid!=grid or
                                                                                                   obj._bContinuous!=bCont))):
                obj=obj.copy()
                setattr(self,name,obj)
            obj._Yperiod=per
            obj._bGeometrySet=True
        self.objective._Ygrid=grid
        self.objective._bContinuous=bCont

    def split(self,stimInd=None):

        nrn=self.nrn.copy()
        model=self.model.copy()
        objective=self.objective.copy()
        optimizer=None if self.optimizer is None else self.optimizer.copy()

        stim=self.stim if stimInd is None else self.stim._subset(stimInd)

        # the same key as this unit (not a new split of it), so splitting does not change what this unit does next
        unit=Unit(stim,nrn,model,objective,optimizer=optimizer,seed=self.seed,rng=self.rng,rng_last=self.rng_last)
        unit.pool_p=self.pool_p
        unit.nrn_p=self.nrn_p
        unit.multiscale_out=getattr(self,'multiscale_out',None)
        unit.param_out=getattr(self,'param_out',None)

        if self.nrn.bFinalized:
            ind=self.nrn.filter.index
            nrn._finalize(stim,
                          self.nrn.dtype,
                          self.nrn.filter.n,
                          ind.ind_lrn,
                          ind.ind_fix,
                          ind.ind_rec,
                          bFourier=self.nrn.bFourier,
                          bAnalytic=self.nrn.bAnalytic,
                          bSplit=self.nrn.bSplit,
                          last=self.last
            )
            nrn.filter.out=self.out
            nrn._W=self.nrn._W

        unit._check()
        return unit

    def _finalize(self,n,ind_lrn,ind_fix=(),ind_rec=(),fourierType=None,bSplit=None,stimInd=None,dtype=None,last=None):
        if self.optimizer is None:
            raise Exception('Optimizer was not provided')
        # filters are about to change
        self.nrn._W=None

        if fourierType is None:
            if self.nrn.bFinalized:
                bFourier=self.nrn.bFourier
                bAnalytic=self.nrn.bAnalytic
            else:
                bFourier=False
                bAnalytic=False
        else:
            bFourier=fourierType>=1
            bAnalytic=fourierType==2

        if bSplit is None:
            bSplit=self.nrn.bSplit if self.nrn.bFinalized else False

        if dtype is None:
            if self.nrn.bFinalized and self.nrn.bFourier==bFourier:
                dtype=self.nrn.dtype
            elif bFourier:
                dtype=jnp.complex64
            else:
                dtype=jnp.float32
        if bFourier and not jnp.issubdtype(dtype,jnp.complexfloating):
            raise Exception('fourier-domain learning requires a complex dtype')

        # a copy: the caller's Stim (which other units may share) keeps its domain and precision
        stim=copy.copy(self.stim_full)._finalize(dtype,None,bFourier,bSplit)
        self.stim=stim if stimInd is None else stim._subset(stimInd)
        self._stimInd=None if stimInd is None else np.atleast_1d(np.asarray(stimInd,dtype=int))
        # results of generated filters describe only the training call that made them
        self.multiscale_out=self.param_out=None
        self._set_geometry()

        self.nrn._finalize(self.stim,
                           dtype,
                           n,
                           ind_lrn,
                           ind_fix,
                           ind_rec,
                           bFourier=bFourier,
                           bAnalytic=bAnalytic,
                           bSplit=bSplit,
                           last=last
        )
        self._check()


    @property
    def _nDim(self):
        # flattened real response dimensions
        return self.nrn.nDimOut

    def _loo_min_count(self):
        # stimuli per category for bLeaveOneOut: the left-out covariance needs a degree of freedom (a centered one two),
        # unless it is pooled over categories
        m=self.model
        return 2 if (m.modelType=='full' or m.ctgPoolWidth is not None or (m.modelType=='circ' and m.circMean=='zero')) else 3

    def _check_batches(self):
        if self.optimizer.batchSize is None or self.model.modelType not in ('gss','student','circ','mix'):
            return
        _,mask=self.optimizer._batch_plan(self.stim.weights)
        mMin=int(np.asarray(mask).sum(0).min())
        if self.model.modelType=='mix' and mMin<2*self.model.nMix:
            raise Exception("modelType='mix' needs at least 2*nMix stimuli per category in every batch; increase Optimizer nBatchMinCtg")
        if self.model.bLeaveOneOut and mMin<self._loo_min_count():
            raise Exception('bLeaveOneOut needs at least ' + str(self._loo_min_count()) + ' stimuli per category in every batch; '
                            'increase Optimizer nBatchMinCtg')
        if mMin < self._nDim+1 and self.model.covRank is None:
            warnings.warn('batches have as few as ' + str(mMin) + ' stimuli in a category, fewer than the ' + str(self._nDim+1)
                          + ' needed for a full-rank AMA-Gauss covariance of ' + str(self._nDim) + ' response dimensions; '
                          'increase batchSize or Optimizer nBatchMinCtg',stacklevel=3)

    def _check(self):
        counts=np.asarray(self.stim.weights).sum(0)
        if self.model.modelType in ('gss','student','circ') and np.any(counts<2) and self.model.ctgPoolWidth is None:
            raise Exception("modelType='" + self.model.modelType + "' needs at least 2 stimuli in every category (fewest: " + str(int(counts.min())) + ')')
        if self.model.nNoiseSamples>1 or self.model.bFixedNoise:
            if str(self.model.responseType).lower()!='basic':
                raise Exception("nNoiseSamples and bFixedNoise average over noisy observations: they need responseType='basic'")
            if not (self.nrn.bNoise_1 or self.nrn.bNoise_2):
                raise Exception('nNoiseSamples and bFixedNoise need sampled response noise (Nrn bNoise_1 or bNoise_2)')
        if self.model.modelType=='student' and not self.model.df>2:
            raise Exception("modelType='student' needs df > 2 (its covariance is matched to the category covariance)")
        if self.model.modelType=='circ' and self.nrn.bFinalized:
            if not self.nrn.bAnalytic:
                raise Exception("modelType='circ' needs complex quadrature-pair responses (fourierType=2)")
            if self.nrn._bWhiten:
                raise Exception("modelType='circ' can not be combined with whitening, which mixes real and imaginary components")
        if self.model.bLeaveOneOut is False and self.model.modelType not in ('gss','full','student','circ','mix'):
            raise Exception("modelType must be 'gss', 'full', 'student', 'circ', or 'mix'")
        if self.model.covRank is not None:
            if self.model.modelType not in ('gss','student','mix','circ'):
                raise Exception("covRank is used by modelType 'gss', 'student', 'mix' and 'circ'")
            nCov=self._nDim//2 if self.model.modelType=='circ' else self._nDim
            if self.nrn.bFinalized and self.model.covRank>=nCov:
                raise Exception('covRank (' + str(self.model.covRank) + ') must be below the number of ' +
                                ('complex ' if self.model.modelType=='circ' else '') + 'response dimensions (' + str(nCov) + ')')
        if self.model.nRef is not None and self.model.modelType!='full':
            raise Exception("nRef is only used by modelType='full'")
        if self.model.nNeighbors is not None:
            if self.model.modelType!='full':
                raise Exception("nNeighbors is only used by modelType='full'")
            if self.model.nRef is not None:
                raise Exception('nNeighbors and nRef are alternatives; set one of them')
        if self.model.bWithin and not getattr(self.stim,'bContinuous',False):
            raise Exception('bWithin estimates within categories from the stimuli\'s own latent values: give Stim y (e.g. Stim.binned)')
        if self.model.modelType=='full' and self.model.bLeaveOneOut and self.model.nRef is not None and self.model.nRef<2:
            raise Exception('bLeaveOneOut with nRef needs nRef of at least 2')
        if self.model.modelType=='mix':
            if self.model.bPoolMeans:
                raise Exception("modelType='mix' pools covariances over categories (ctgPoolWidth), not means (bPoolMeans)")
            if np.any(counts<2*self.model.nMix):
                raise Exception("modelType='mix' needs at least 2*nMix stimuli in every category (fewest: " + str(int(counts.min())) + ')')
        if self.model.modelType=='full' and self.nrn.corrType=='None':
            raise Exception("modelType='full' needs response noise; rho=None (no noise) is only supported with modelType='gss'")
        if self.nrn.corrType=='corr' and self.nrn.bFinalized:
            nDim=self._nDim
            rho=self.nrn.rho
            if np.ndim(rho)>0:
                if rho.shape!=(nDim,nDim):
                    raise Exception('the rho matrix must be ' + str(nDim) + ' x ' + str(nDim) + ' (the flattened response dimensions)')
                if not (np.allclose(rho,rho.T) and np.allclose(np.diag(rho),1) and np.linalg.eigvalsh(rho).min()>0):
                    raise Exception('the rho matrix must be a correlation matrix: symmetric, unit diagonal, positive definite')
            else:
                lo=-1/(nDim-1) if nDim>1 else -np.inf
                if not (lo < rho < 1):
                    raise Exception('rho must be in (' + str(lo) + ', 1) for ' + str(nDim) + ' response dimensions')
        if self.nrn._bWhiten:
            if str(self.nrn.whitenType).lower() not in ('gram','response'):
                raise Exception("whitenType must be 'None', 'gram', or 'response'")
            if self.nrn.whitenMethod not in ('zca','chol'):
                raise Exception("whitenMethod must be 'zca' or 'chol'")
            if self.nrn.normalizeType=='narrow':
                raise Exception("whitening can not be combined with normalizeType='narrow'")
            if str(self.nrn.whitenType).lower()=='response' and self.nrn.bFinalized:
                nDim=self._nDim
                if nDim >= counts.sum():
                    raise Exception("whitenType='response' needs more stimuli (" + str(int(counts.sum())) + ') than response dimensions (' + str(nDim) + ')')
        if str(self.nrn.readoutType).lower() not in ('none','resultant','resultant_only','linear'):
            raise Exception("readoutType must be 'None', 'resultant', 'resultant_only', or 'linear'")
        nrn=self.nrn
        if nrn.bSplitNegatives and str(nrn.normalizeType).lower()=='narrow':
            raise Exception("bSplitNegatives can not be combined with normalizeType='narrow' (one denominator per filter)")
        if nrn.bSplitNegatives and nrn.bPooledReadout:
            raise Exception("bSplitNegatives can not be combined with the pooled readouts ('resultant', 'resultant_only')")
        if (nrn.normPool is not None or nrn.bLearnNormPool) and str(nrn.normalizeType).lower()!='gen':
            raise Exception("normPool and bLearnNormPool apply to normalizeType='gen'")
        if nrn.bFinalized:
            if nrn.normPool is not None and str(nrn.normalizeType).lower()=='gen' and nrn.normPool.shape!=(nrn.nChan,nrn.nChan):
                raise Exception('normPool must be ' + str(nrn.nChan) + ' x ' + str(nrn.nChan) + ' (the response channels entering normalization)')
            if str(nrn.activationType).lower()=='softmax' and nrn.bComplexOut:
                raise Exception("activationType='softmax' needs real responses (not fourierType=2 without whitening)")
            if str(nrn.readoutActivation).lower()=='softmax' and nrn.bComplexOut:
                raise Exception("readoutActivation='softmax' needs real responses (not fourierType=2 without whitening)")
        if str(nrn.readoutType).lower()!='linear' and (nrn.nReadout is not None or str(nrn.readoutActivation).lower()!='none'):
            raise Exception("nReadout and readoutActivation are used by readoutType='linear'")
        if self.nrn.bReadout and self.nrn.bNoise_1:
            raise Exception('stage-1 noise (bNoise_1) is not modeled through a pooled readout')
        if self.nrn.bReadout and self.nrn._bWhiten:
            raise Exception('a pooled readout can not be combined with whitening')
        nDim,per=getattr(self.stim,'nDim',1),getattr(self.stim,'Yperiod',None)
        if self.objective.errType=='fisher' and nDim>1 and self.objective._Ygrid is None:
            raise Exception("errType='fisher' with several latent dimensions needs the levels (Y) on a cartesian grid")
        if per is not None:
            Yv=np.asarray(self.stim.Y).reshape(len(self.stim.Y),-1)
            for d,P in enumerate(per):
                if P is not None and np.ptp(Yv[:,d])>=P:
                    raise Exception('latent values of circular dimension ' + str(d) + ' must span less than its period (' + str(P) + ')')
        if self.objective.estType=='cmean' and nDim>1:
            raise Exception("estType='cmean' needs a single latent dimension; set Stim Yperiod and use 'mean'")
        if self.model.bLeaveOneOut:
            if self.objective.errType=='mle':
                raise Exception("bLeaveOneOut defines a leave-one-out posterior and can not be used with errType='mle'")
            nMin=self._loo_min_count()
            if np.any(counts<nMin):
                raise Exception('bLeaveOneOut with modelType=' + repr(self.model.modelType) + ' needs at least ' + str(nMin) + ' stimuli in every category')
        if self.nrn.bNoise_1 and str(self.nrn.normalizeType).lower()=='phase':
            raise Exception("stage-1 noise (bNoise_1) is not modeled through normalizeType='phase', which is not linear")
        if self.nrn.corrType!='None' and not self.nrn.var0>0:
            raise Exception('var0 must be positive: the noise variance fano*|r| + var0 is otherwise 0 for zero responses')
        if (self.nrn.bNoise_1 and str(self.nrn.normalizeType).lower()=='gen' and self.nrn.bFinalized
                and self.nrn.bAnalytic and not self.nrn._bWhiten):
            raise Exception("stage-1 noise (bNoise_1) with normalizeType='gen' is only modeled for real responses (not fourierType=2)")

#- LEARN MODES
    def _p_old(self):
        old=dict(getattr(self,'nrn_p',None) or {})
        if self.pool_p is not None:
            old['pool']=self.pool_p
        return old

    def _p0(self):
        # initial learned response-model parameters (Nrn.params0): previous values where they fit; pooling weights extended
        # with zeros (equal weights) for new filters
        return self.nrn.params0(self._p_old(),seed=self.seed)

    def _set_p(self,p):
        # store learned response-model parameters (a dict) in pool_p and nrn_p
        if p is None:
            return
        p={k:np.asarray(v) for k,v in p.items()}
        self.pool_p=p.pop('pool',None)
        self.nrn_p=p or None

    def _val_fun(self,stimVal):
        test=self._prepare_stim(stimVal)
        def fun(params):
            full={'f':self.filter.insert(params['f'])}
            if 'p' in params:
                full['p']=params['p']
            return self._loss_fun_heldout(full,self.rng,test.val,test.weights,test.yCtg,test.Y,self.stim.val,self.stim.weights,
                                          self.stim.yCtg)
        return fun

    def _opt_key(self,shape):
        # what an optimizer state belongs to: the learned filter columns, their parameter shape, the optimizer, the precision
        ind=self.filter.index
        cols=tuple(int(c) for c in np.sort(np.concatenate((ind.ind_lrn,ind.ind_rec))))
        return (tuple(shape),cols,str(self.optimizer.optimizerType),np.dtype(self.nrn.dtype).name)

    def _run(self,f0,rng,opt_state=None,stimVal=None):
        self._check_batches()
        extra={'p':self._p0()} if self.nrn.bParams else None
        val_fun=None if stimVal is None else self._val_fun(stimVal)
        nbr=self._bNeighbors()
        self.out_params,self.opt_state,self.rng_last=self.optimizer.minimize(f0,rng,self.stim,self.filter,_BoundLoss(self,'_loss_fun_lrn'),
                                                                             opt_state=opt_state,extra_params=extra,val_fun=val_fun,
                                                                             mstate0=None if nbr else self._mix_state0(f0),
                                                                             state_fun=_BoundLoss(self,'_nbr_state') if nbr else None)
        self._opt_param_shape=self.out_params['f'].shape
        self._opt_state_key=self._opt_key(self._opt_param_shape)
        self.filter.extract(self.out_params['f'])
        if 'p' in self.out_params:
            self._set_p(self.out_params['p'])

    def _train_random(self,rng,nRestarts,stimVal=None):
        # learn from random initial filters; with restarts, keep the run with the lowest cost on the validation stimuli
        # (stimVal) if given, else on the training stimuli. restart_costs holds those costs
        if nRestarts<1:
            raise Exception('nRestarts must be at least 1')
        self.restart_costs=[]
        best=None
        for i in range(nRestarts):
            if i>0:
                rng=jxrandom.fold_in(rng,i)
            rng,rng_key = jxrandom.split(rng)
            f0=self.filter.get_f0(rng_key,self.optimizer._f0_jxrand_fun)
            self._run(f0,rng,stimVal=stimVal)
            if nRestarts==1:
                self.restart_costs=None
                return
            cost=float(self.loss) if stimVal is None else self.evaluate(stimVal)
            self.restart_costs.append(cost)
            if best is None or cost<best[0]:
                best=(cost,jnp.asarray(self.filter.out),self.out_params,self.opt_state,self.rng_last,
                      list(self.optimizer.loss_hist),list(self.optimizer.val_hist),self.optimizer.best_step,
                      self._opt_param_shape,self._opt_state_key,self.pool_p,self.nrn_p)
        (_,self.filter.out,self.out_params,self.opt_state,self.rng_last,self.optimizer.loss_hist,self.optimizer.val_hist,
         self.optimizer.best_step,self._opt_param_shape,self._opt_state_key,self.pool_p,self.nrn_p)=best

    @_logged_training
    def train_new(self,n,fourierType=None,bSplit=None,stimInd=None,dtype=None,optimizer=None,nRestarts=1,stimVal=None):
        """learn n new filters. stimVal: validation stimuli for early stopping (Optimizer patience)"""
        if optimizer is not None:
            self.optimizer=optimizer
        self.pool_p=self.nrn_p=None

        self._finalize(n,
                      np.arange(n),
                      (),
                      (),
                      fourierType=fourierType,
                      stimInd=stimInd,
                      dtype=dtype,
                      bSplit=bSplit
        )

        self._train_random(self.rng,nRestarts,stimVal)

    @_logged_training
    def train_recurse(self,ind_rec=None,fourierType=None,bSplit=None,stimInd=None,dtype=None,optimizer=None,stimVal=None):
        if optimizer is not None:
            self.optimizer=optimizer

        n=self.nrn.filter.n
        full=np.arange(n)
        if ind_rec is None:
            ind_rec=full
        ind_rec=np.array(ind_rec,dtype=int,ndmin=1)
        ind_fix=full[np.isin(full,ind_rec,invert=True)]

        self._finalize(n,
                      (),
                      ind_fix,
                      ind_rec,
                      fourierType=fourierType,
                      bSplit=bSplit,
                      stimInd=stimInd,
                      dtype=dtype,
                      last=self.out
        )

        f0=self.filter.out_flat[self.filter._insert_index_jx]
        opt_state=self.opt_state if getattr(self,'_opt_state_key',None)==self._opt_key(f0.shape) else None

        rng,_ = jxrandom.split(self.rng if self.rng_last is None else self.rng_last)
        self._run(f0,rng,opt_state,stimVal=stimVal)

    @_logged_training
    def train_append(self,n_append,fourierType=None,bSplit=None,stimInd=None,dtype=None,optimizer=None,nRestarts=1,stimVal=None):
        if optimizer is not None:
            self.optimizer=optimizer

        n0=self.nrn.filter.n
        n=n0+n_append

        self._finalize(n,
                      np.arange(n0,n),
                      np.arange(n0),
                      (),
                      fourierType=fourierType,
                      bSplit=bSplit,
                      stimInd=stimInd,
                      dtype=dtype,
                      last=self.out
        )

        self._train_random(self.rng if self.rng_last is None else self.rng_last,nRestarts,stimVal)

    #- parametric filter banks
    def _parametric_init(self,family,n,bTied,init,b2D):
        init=dict(init or {})
        kTop=float(init.get('kTop',0.25))                   # peak frequency of the first filter, cycles per sample
        ratio=float(init.get('ratio',2.0))                  # scale ratio between adjacent filters (tied banks)
        rd=jnp.finfo(self.nrn.dtype).dtype
        full=lambda v: jnp.full((n,),v,dtype=rd) if not bTied else jnp.asarray(v,dtype=rd)
        params={}
        if bTied:
            params['log_kTop']=jnp.asarray(np.log(kTop),dtype=rd)
            params['log_ratio_m1']=jnp.asarray(np.log(ratio-1),dtype=rd)
        else:
            params['log_kPeak']=jnp.asarray(np.log(kTop/ratio**np.arange(n)),dtype=rd)
        if family=='morse':
            params['log_gamma']=full(np.log(float(init.get('gamma',2.0))))
            params['log_b']=full(np.log(float(init.get('b',2.0))))
        elif family=='loggabor':
            params['log_sigma_u']=full(np.log(float(init.get('sigma_u',0.5))))
        else:
            raise Exception("family must be 'morse' or 'loggabor'")
        if b2D:
            params['log_sigma_theta']=full(np.log(float(init.get('sigma_theta',np.deg2rad(20)))))
        return params

    def _parametric_filters(self,params,family,n,bTied,orientations):
        """unit-norm fourier-domain filters [ nPix x nF ] on the learned half-space coefficients"""
        flt=self.filter
        pix_dims=flt.pix_dims
        grids=np.meshgrid(*[_centered_freqs(d) for d in pix_dims],indexing='ij')
        kr=np.sqrt(sum(g**2 for g in grids)).ravel()[flt.index.pix]             # [ nPix_learned ]
        e=jnp.exp
        if bTied:
            kPeak=e(params['log_kTop'])/(1+e(params['log_ratio_m1']))**jnp.arange(n)
        else:
            kPeak=e(params['log_kPeak'])
        x=kr[:,None]/kPeak[None,:]                                               # [ nPix_learned x nF ]
        lx=jnp.log(jnp.maximum(x,1e-30))
        if family=='morse':
            gam,b=e(params['log_gamma']),e(params['log_b'])
            lprof=b*lx + (b/gam)*(1-jnp.exp(gam*lx))                             # peak 1 at x = 1
        else:
            su=e(params['log_sigma_u'])
            lprof=-lx**2/(2*su**2)
        if len(pix_dims)==2:
            ang=np.arctan2(grids[1],grids[0]).ravel()[flt.index.pix]
            d=(ang[:,None]-np.asarray(orientations)[None,:]+np.pi)%(2*np.pi)-np.pi
            st=e(params['log_sigma_theta'])
            lprof=lprof - d**2/(2*st**2) + jnp.where(np.cos(d)>0,0.,-jnp.inf)
        prof=jnp.exp(jnp.maximum(lprof,-700.))
        prof=prof/jnp.linalg.norm(prof,axis=0,keepdims=True)
        f=jnp.zeros((int(np.prod(pix_dims)),n),dtype=self.nrn.dtype)
        return f.at[flt.index.pix].set(prof.astype(self.nrn.dtype))

    def _train_generated(self,params,kind,cfg,stimVal=None):
        """
        optimize params (a dict of arrays; 'p' holds readout pooling weights) of the filters generated by kind ('parametric'
        or 'multiscale', see _generated_filters) with configuration cfg, on this unit's cost plus the generator's penalty,
        with this unit's Optimizer (optimizerType, lRate0, nIterMax, nStepsPerChunk, batchSize, bVerbose, patience with
        stimVal). Returns the final (or best validation) params.
        """
        loss=_GeneratedLoss(self,kind,cfg)
        val_fun=None
        if stimVal is not None:
            test=self._prepare_stim(stimVal)
            val_fun=lambda prm: _generated_heldout(loss,prm,self.rng,test.val,test.weights,test.yCtg,test.Y,
                                                   self.stim.val,self.stim.weights,self.stim.yCtg)
        rng,_=jxrandom.split(self.rng)
        params,self.rng_last=self.optimizer.minimize_generated(params,rng,self.stim,loss,val_fun)
        return params

    def _generated_filters(self,kind,params,cfg):
        # filters [ nPix x nF ] generated from params; with a learned phase (bPhase) each filter's spectrum is rotated by it
        if kind=='parametric':
            f=self._parametric_filters(params,cfg['family'],cfg['n'],cfg['bTied'],cfg['orientations'])
        else:
            f=self._multiscale_filters(params,cfg['nTot'],cfg)
        if 'phase' in params:
            f=f*jnp.exp(1j*params['phase']).astype(f.dtype)[None,:]
        return f

    def _generated_penalty(self,kind,params,cfg):
        # multiscale knotSmooth: squared second differences of each mother's tapered knots, relative to its energy
        if kind!='multiscale' or not cfg['knotSmooth']>0:
            return None
        nMothers=cfg['nMothers']
        tm=self._tapered_mothers(params,cfg)
        energy=jnp.sum((jnp.abs(tm)**2).reshape(nMothers,-1),axis=1)+1e-12
        tot=0.
        for v in (jnp.real(tm),jnp.imag(tm)):
            d=jnp.sum((jnp.diff(v,n=2,axis=1)**2).reshape(nMothers,-1),axis=1)
            if tm.ndim==3:                                                   # 2D: also along angle
                d=d+jnp.sum((jnp.diff(v,n=2,axis=2)**2).reshape(nMothers,-1),axis=1)
            tot=tot+jnp.sum(d/energy)
        return cfg['knotSmooth']*tot

    def _set_generated(self,f,params):
        # generated filters [ nPix x nF ] become this unit's ordinary filters
        self.filter.out=jnp.reshape(f,self.filter._shape_exp)
        self.out_params={'f':f[self.filter._insert_index_jx]}
        self._opt_param_shape=self.out_params['f'].shape
        self._opt_state_key=None
        self.opt_state=None
        if 'p' in params:
            self._set_p(params['p'])

    #- non-parametric multiscale filter banks
    @staticmethod
    def _interp_uniform(x,x0,dx,v):
        """
        linear interpolation of knot values v [ nKnot x ... ] on the uniform grid x0 + dx*arange(nKnot) at points x
        [ nPts x ... ] (broadcast), zero outside the grid
        """
        n=v.shape[0]
        t=(x-x0)/dx
        i=jnp.floor(t)
        w=t-i
        i=i.astype(int)
        inside=(t>=0)&(t<=n-1)
        i0=jnp.clip(i,0,n-1)
        i1=jnp.clip(i+1,0,n-1)
        val=v[i0]*(1-w)+v[i1]*w
        return jnp.where(inside,val,0.)

    @staticmethod
    def _knot_taper(nKnot,edgeTaper):
        """[ nKnot ] window: cosine from 0 at the outermost knots to 1 over a fraction edgeTaper of the knot range at each end"""
        t=np.linspace(0.,1.,nKnot)
        d=np.minimum(t,1-t)
        if edgeTaper<=0:
            return np.ones(nKnot)
        return np.where(d<edgeTaper,0.5*(1-np.cos(np.pi*d/edgeTaper)),1.)

    @staticmethod
    def _tapered_mothers(params,cfg):
        """complex mother knots [ nMothers x nKnot (x nKnotTheta) ] after the log-frequency edge taper"""
        w=jnp.asarray(cfg.get('knotTaper',np.ones(params['mother_re'].shape[1])),dtype=params['mother_re'].dtype)
        w=w.reshape((1,-1)+(1,)*(params['mother_re'].ndim-2))
        return (params['mother_re'] + 1j*params['mother_im'])*w

    def _multiscale_filters(self,params,n,cfg):
        """
        dilated (and, in 2D, rotated) copies of non-parametric mother filters, as unit-norm fourier-domain filters
        [ nPix x nF ]. Each mother is complex knot values over log(k/k_peak) (1D: [ nKnot ]; 2D: [ nKnot x nKnotTheta ]
        over log radius and angle), linearly interpolated; params hold [ nMothers x ... ]. Filter order: mother first,
        then the scale (and orientation) filters of cfg (every mother shares the same scales and orientations).
        """
        flt=self.filter
        pix_dims=flt.pix_dims
        grids=np.meshgrid(*[_centered_freqs(d) for d in pix_dims],indexing='ij')
        kr=np.sqrt(sum(g**2 for g in grids)).ravel()[flt.index.pix]
        lk=np.log(np.maximum(kr,1e-12))                                           # [ nPts ]
        e=jnp.exp
        if cfg['scaleType']=='tied':
            # scaleExp: the scale step of each filter (0, 1, 2, ... ; +1/2 for interleaved filters)
            logPeak=params['log_kTop'] - jnp.log1p(e(params['log_ratio_m1']))*jnp.asarray(cfg['scaleExp'])
        elif cfg['scaleType']=='fixed':
            logPeak=jnp.asarray(np.log(cfg['kPeak']))
        else:
            logPeak=params['log_kPeak']
        mothers=self._tapered_mothers(params,cfg)                                 # [ nMothers x nKnot (x nKnotTheta) ]
        u=lk[:,None]-logPeak[None,:]                                              # [ nPts x nF_mother ] log(k/kPeak_j)
        uw=cfg['uWidth']
        du=2*uw/(cfg['nKnot']-1)
        allvals=[]
        for im in range(cfg.get('nMothers',1)):
          mother=mothers[im]
          if len(pix_dims)==1:
            vals=self._interp_uniform(u,-uw,du,mother)                            # mother [ nKnot ]
          else:
            ang=np.arctan2(grids[1],grids[0]).ravel()[flt.index.pix]
            d=(ang[:,None]-np.asarray(cfg['orientations'])[None,:]+np.pi)%(2*np.pi)-np.pi   # [ nPts x nF ]
            nT=cfg['nKnotTheta']
            dth=np.pi/(nT-1)
            # bilinear on [ log radius x angle in [-pi/2, pi/2] ]
            tu=(u+uw)/du
            tt=(d+np.pi/2)/dth
            iu=jnp.floor(tu); wu=tu-iu; iu=iu.astype(int)
            it=np.floor(tt); wt=tt-it; it=it.astype(int)
            inside=(tu>=0)&(tu<=cfg['nKnot']-1)&(tt>=0)&(tt<=nT-1)
            c=lambda a,m: jnp.clip(a,0,m-1)
            v00=mother[c(iu,cfg['nKnot']),c(it,nT)]
            v01=mother[c(iu,cfg['nKnot']),c(it+1,nT)]
            v10=mother[c(iu+1,cfg['nKnot']),c(it,nT)]
            v11=mother[c(iu+1,cfg['nKnot']),c(it+1,nT)]
            vals=((1-wu)*((1-wt)*v00+wt*v01)+wu*((1-wt)*v10+wt*v11))
            vals=jnp.where(inside,vals,0.)
          allvals.append(vals)
        vals=jnp.concatenate(allvals,axis=1)                                      # [ nPts x nMothers*nF_mother ]
        norm=jnp.sqrt(jnp.sum(jnp.abs(vals)**2,axis=0,keepdims=True))
        vals=vals/jnp.where(norm>0,norm,1)
        f=jnp.zeros((int(np.prod(pix_dims)),n),dtype=self.nrn.dtype)
        return f.at[flt.index.pix].set(vals.astype(self.nrn.dtype))

    @_logged_training
    def train_multiscale(self,n=None,fourierType=2,nKnot=25,uWidth=2.5,scaleType='tied',kTop=0.25,ratio=2.0,kPeak=None,
                         orientations=None,nKnotTheta=13,knotSmooth=0.,init='loggabor',dtype=None,stimVal=None,
                         nScales=None,nOrientations=None,bInterleave=False,nMothers=1,motherInitNoise=0.1,edgeTaper=0.,
                         bPhase=False):
        """
        learn a multiscale filter bank whose filters are copies of one (or nMothers) non-parametric mother filter: dilated (shifted in
        log frequency) and, for 2D stimuli, rotated (shifted in angle). The mother filter is free: complex spectrum values
        at nKnot points uniform over log(k/k_peak) in [-uWidth, uWidth] (2D: times nKnotTheta angles over [-pi/2, pi/2],
        one-sided), linearly interpolated, zero outside. Filters are unit norm, in the fourier domain (fourierType 1 or 2).

        scaleType - 'tied'  : peak frequencies kTop/ratio^j with kTop and ratio learned (initialized from kTop, ratio)
                    'fixed' : peak frequencies kPeak [ nF ] (default kTop/ratio^j), not learned
                    'free'  : a learned peak frequency per filter (initialized from kTop/ratio^j)
        orientations - 2D: [ nF ] radians from the first stimulus axis (default 0)
        knotSmooth   - weight of a penalty on squared second differences of the mother's knot values (smooth spectra)
        init         - initial mother: 'loggabor' (a real log-gaussian bump, sigma_u = uWidth/3) or 'random'
        nScales, nOrientations - 2D: build the filters as a grid instead of per-filter lists: peaks kTop/ratio^j
                       (j = 0..nScales-1) at orientations o*pi/nOrientations (o = 0..nOrientations-1, wrapped to
                       (-pi/2, pi/2]), n = nScales*nOrientations filters (n need not be given)
        bInterleave  - with the grid: add an interleaved set of the same size at the tritones (half a scale step,
                       kTop/ratio^(j+1/2): half an octave for ratio 2) in the half-rotations (orientations offset by
                       half the orientation step, pi/(2 nOrientations)). All filters are copies of the same mother;
                       with scaleType='tied' the interleaved scales stay halfway (in log frequency) as ratio is learned.
                       Filter order: the grid (scales x orientations, scale first), then the interleaved grid.
        nMothers     - learn this many mother filters simultaneously. Each gets the same set of scale (and orientation)
                       filters, so the bank has nMothers * n filters, ordered mother first; the scales (kTop and ratio,
                       kPeak, or the free peaks) and orientations are shared by all mothers. n (or the grid) counts the
                       filters per mother.
        edgeTaper    - force each mother to vanish at the ends of its log-frequency support: its knots are multiplied by a
                       fixed window that rises (cosine) from 0 at the outermost knots to 1 over this fraction of the knot
                       range at each end (0 = no taper; e.g. 0.2 tapers 5 of 25 knots per end). Without it a mother can
                       move its energy to the edge of the support, making its dilated copies narrowband filters cut off at
                       the boundary. The reported mother, the filters, and knotSmooth use the tapered knots. 2D: the taper
                       is along log radius only.
        motherInitNoise - with nMothers > 1, every mother's initial knots get independent complex gaussian noise of this
                       size relative to the initial bump (seeded from the unit's rng): identical mothers would receive
                       identical gradients and never diverge.
        bPhase       - also learn a phase per filter (exp(i phase) on its spectrum), so the dilated copies of a mother need
                       not share its phase (e.g. phase alignment across scales for a pooled readout)
        Uses this unit's Optimizer (and Objective regType on the generated filters is not applied; use knotSmooth).
        The learned filters become ordinary filters (unit.out); multiscale_values() returns the mother filter and scales.
        Not implemented for split stimuli.
        """
        if fourierType not in (1,2):
            raise Exception('train_multiscale learns in the fourier domain: fourierType must be 1 or 2')
        if scaleType not in ('tied','fixed','free'):
            raise Exception("scaleType must be 'tied', 'fixed', or 'free'")
        bGrid=nScales is not None or nOrientations is not None
        if bInterleave and not bGrid:
            raise Exception('bInterleave needs the filter grid: give nScales and nOrientations')
        scaleExp=None
        if bGrid:
            if nScales is None or nOrientations is None or nScales<1 or nOrientations<1:
                raise Exception('give both nScales and nOrientations (at least 1)')
            if kPeak is not None or orientations is not None:
                raise Exception('with nScales and nOrientations, kPeak and orientations are set by the grid')
            jj,oo=np.meshgrid(np.arange(nScales),np.arange(nOrientations),indexing='ij')
            scaleExp=jj.ravel().astype(float)
            orientations=oo.ravel()*np.pi/nOrientations
            if bInterleave:
                scaleExp=np.concatenate((scaleExp,scaleExp+0.5))
                orientations=np.concatenate((orientations,orientations+np.pi/(2*nOrientations)))
            orientations=(orientations+np.pi/2)%np.pi-np.pi/2                     # wrap to [-pi/2, pi/2)
            orientations=np.where(np.isclose(orientations,-np.pi/2),np.pi/2,orientations)
            nGrid=len(scaleExp)
            if n is not None and n!=nGrid:
                raise Exception('n (' + str(n) + ') does not match the grid (' + str(nGrid) + ' filters)')
            n=nGrid
        elif n is None:
            raise Exception('give n, or nScales and nOrientations')
        if nKnot<3 or (nKnotTheta is not None and nKnotTheta<2):
            raise Exception('nKnot must be at least 3 and nKnotTheta at least 2')
        nMothers=int(nMothers)
        if nMothers<1:
            raise Exception('nMothers must be at least 1')
        if not 0<=edgeTaper<=0.5:
            raise Exception('edgeTaper must be in [0, 0.5]')
        nTot=nMothers*n
        self.pool_p=self.nrn_p=None
        self._finalize(nTot,np.arange(nTot),fourierType=fourierType,dtype=dtype,bSplit=False)
        if len(self.filter.pix_dims) not in (1,2):
            raise Exception('train_multiscale supports 1D and 2D stimuli')
        b2D=len(self.filter.pix_dims)==2
        if bGrid and not b2D:
            raise Exception('nScales, nOrientations, and bInterleave need 2D stimuli (rotation)')
        orientations=np.zeros(n) if orientations is None else np.asarray(orientations,dtype=float)
        if b2D and orientations.shape!=(n,):
            raise Exception('orientations must have one value per filter')
        if scaleExp is None:
            scaleExp=np.arange(n,dtype=float)
        kPeak0=kTop/ratio**scaleExp if kPeak is None else np.asarray(kPeak,dtype=float)
        if kPeak0.shape!=(n,):
            raise Exception('kPeak must have one value per filter')
        rd=jnp.finfo(self.nrn.dtype).dtype
        cfg=dict(scaleType=scaleType,uWidth=float(uWidth),nKnot=int(nKnot),nKnotTheta=int(nKnotTheta),
                 orientations=orientations,kPeak=kPeak0,scaleExp=scaleExp,nMothers=nMothers,
                 knotTaper=self._knot_taper(int(nKnot),float(edgeTaper)),nTot=nTot,knotSmooth=float(knotSmooth))

        ug=np.linspace(-uWidth,uWidth,nKnot)
        if init=='loggabor':
            m=np.exp(-ug**2/(2*(uWidth/3)**2))
            if b2D:
                tg=np.linspace(-np.pi/2,np.pi/2,nKnotTheta)
                m=m[:,None]*np.exp(-tg[None,:]**2/(2*np.deg2rad(20)**2))
        elif init=='random':
            rng=np.random.default_rng(int(np.asarray(jxrandom.key_data(self.rng)).ravel()[-1]))
            m=rng.standard_normal((nKnot,nKnotTheta) if b2D else (nKnot,))
        else:
            raise Exception("init must be 'loggabor' or 'random'")
        m=np.broadcast_to(m,(nMothers,)+np.shape(m)).copy()
        mi=np.zeros(np.shape(m))
        if nMothers>1 and motherInitNoise>0:
            nrng=np.random.default_rng(int(np.asarray(jxrandom.key_data(self.rng)).ravel()[-1])+7919)
            scale=motherInitNoise*np.max(np.abs(m))
            m=m+scale*nrng.standard_normal(m.shape)
            mi=scale*nrng.standard_normal(m.shape)
        params={'mother_re':jnp.asarray(m,dtype=rd),'mother_im':jnp.asarray(mi,dtype=rd)}
        if scaleType=='tied':
            params['log_kTop']=jnp.asarray(np.log(kTop),dtype=rd)
            params['log_ratio_m1']=jnp.asarray(np.log(ratio-1),dtype=rd)
        elif scaleType=='free':
            params['log_kPeak']=jnp.asarray(np.log(kPeak0),dtype=rd)
        if bPhase:
            params['phase']=jnp.zeros(nTot,dtype=rd)
        if self.nrn.bParams:
            params['p']=self._p0()

        params=self._train_generated(params,'multiscale',cfg,stimVal=stimVal)
        self._set_generated(self._generated_filters('multiscale',params,cfg),params)

        mother=np.asarray(self._tapered_mothers(params,cfg))
        if nMothers==1:
            mother=mother[0]                                                      # single mother: [ nKnot (x nKnotTheta) ]
        if scaleType=='tied':
            r=float(1+np.exp(np.asarray(params['log_ratio_m1'])))
            kp=float(np.exp(np.asarray(params['log_kTop'])))/r**scaleExp
        elif scaleType=='fixed':
            r,kp=None,kPeak0
        else:
            r,kp=None,np.exp(np.asarray(params['log_kPeak']))
        tile=lambda v: np.tile(np.asarray(v),nMothers)                           # per mother -> per filter (mother first)
        out={'mother':mother,'log_freq_knots':ug,'k_peak':tile(kp),'ratio':r,'scaleType':scaleType,'fourierType':fourierType}
        out['scale_step']=tile(scaleExp)
        out['nMothers']=nMothers
        out['edgeTaper']=float(edgeTaper)
        out['mother_index']=np.repeat(np.arange(nMothers),n)
        if bPhase:
            out['phase']=np.asarray(params['phase'])
        if b2D:
            out['angle_knots']=np.linspace(-np.pi/2,np.pi/2,nKnotTheta)
            out['orientations']=tile(orientations)
        if bGrid:
            out['nScales'],out['nOrientations'],out['bInterleave']=nScales,nOrientations,bool(bInterleave)
            out['interleaved']=tile(scaleExp%1!=0)
        self.multiscale_out=out

    def multiscale_values(self):
        """the learned mother filter (complex knot values and their log-frequency, and angle, positions) and the peak
        frequencies (cycles/sample) and ratio of the last train_multiscale"""
        if getattr(self,'multiscale_out',None) is None:
            raise Exception('train_multiscale has not been run')
        return self.multiscale_out

    def parametric_values(self):
        """interpretable parameters of the last train_parametric: peak frequencies (cycles/sample), shapes, orientations"""
        if getattr(self,'param_out',None) is None:
            raise Exception('train_parametric has not been run')
        return self.param_out

    @_logged_training
    def train_parametric(self,n,family='morse',fourierType=2,bTied=True,orientations=None,init=None,dtype=None,stimVal=None,
                         bPhase=False):
        """
        learn a parametric filter bank instead of free filters: generalized Morse, |H(k)| = (k/kp)^b exp((b/gamma)(1-(k/kp)^gamma)),
        or log-Gabor, exp(-log(k/kp)^2/(2 sigma_u^2)), with a gaussian angular profile (width sigma_theta, one-sided) around
        each filter's orientation for 2D stimuli. With bTied, the filters are dilations of one mother filter with peak
        frequencies kTop/ratio^j (parameters kTop, ratio, and one shape); otherwise every filter has its own peak and shape.
        orientations [ nF ] radians from the first stimulus axis (2D; default 0). init overrides starting values (kTop, ratio,
        gamma, b, sigma_u, sigma_theta). Uses this unit's Optimizer (optimizerType, lRate0, nIterMax, nStepsPerChunk,
        patience with stimVal). The learned filters become ordinary filters (unit.out), so evaluation, saving, and
        train_recurse (free refinement from the parametric solution) work as usual; parametric_values() returns the
        parameters. Not implemented for split stimuli.
        bPhase - also learn a phase per filter, which multiplies its spectrum by exp(i phase): the profiles are otherwise
                 zero-phase (even filters with fourierType=1; cosine/sine pairs with fourierType=2), and a learned phase
                 gives odd or intermediate filters, or rotates each quadrature pair (which matters with 'phase'
                 normalization and pooled readouts, and through the response-dependent noise)
        """
        if fourierType not in (1,2):
            raise Exception('train_parametric learns in the fourier domain: fourierType must be 1 or 2')
        self.pool_p=self.nrn_p=None
        self._finalize(n,np.arange(n),fourierType=fourierType,dtype=dtype,bSplit=False)
        if len(self.filter.pix_dims) not in (1,2):
            raise Exception('train_parametric supports 1D and 2D stimuli')
        b2D=len(self.filter.pix_dims)==2
        orientations=np.zeros(n) if orientations is None else np.asarray(orientations,dtype=float)
        if b2D and orientations.shape!=(n,):
            raise Exception('orientations must have one value per filter')
        params=self._parametric_init(family,n,bTied,init,b2D)
        if bPhase:
            params['phase']=jnp.zeros(n,dtype=jnp.finfo(self.nrn.dtype).dtype)
        if self.nrn.bParams:
            params['p']=self._p0()
        cfg=dict(family=family,n=n,bTied=bTied,orientations=orientations)
        params=self._train_generated(params,'parametric',cfg,stimVal=stimVal)
        self._set_generated(self._generated_filters('parametric',params,cfg),params)
        e=lambda k: np.exp(np.asarray(params[k]))
        out={'family':family,'bTied':bTied,'fourierType':fourierType}
        if bTied:
            out['ratio']=float(1+e('log_ratio_m1'))
            out['k_peak']=float(e('log_kTop'))/out['ratio']**np.arange(n)
        else:
            out['k_peak']=e('log_kPeak')
        if family=='morse':
            out['gamma'],out['b']=e('log_gamma'),e('log_b')
        else:
            out['sigma_u']=e('log_sigma_u')
        if b2D:
            out['sigma_theta']=e('log_sigma_theta')
            out['orientations']=orientations
        if bPhase:
            out['phase']=np.asarray(params['phase'])
        self.param_out=out

    #- evaluation
    def _prepare_stim(self,stim):
        # other stimuli in this unit's learning domain and precision
        if not self.nrn.bFinalized:
            raise Exception('train (or finalize) the unit first')
        if (stim.nCtg!=self.stim.nCtg or np.shape(stim.Y)!=np.shape(self.stim.Y)
                or not np.allclose(np.asarray(stim.Y),np.asarray(self.stim.Y))):
            raise Exception('stimuli must have the same categories (Y) as the training stimuli')
        if getattr(stim,'Yperiod',None)!=getattr(self.stim,'Yperiod',None):
            raise Exception('stimuli must have the same Yperiod as the training stimuli')
        return copy.copy(stim)._finalize(self.nrn.dtype,None,self.nrn.bFourier,self.nrn.bSplit)

    @partial(jit, static_argnames=['self'])
    def _loss_fun_heldout(self,params,rng_key,stimval,stimweights,yCtg,Y,refval,refweights,refy):
        f=params['f']
        p=params.get('p')
        W=self.nrn.whitening(refval,f,refweights) if self.nrn._bWhiten else None
        G=self.nrn.gain(refval,f,refweights,W,p)

        def one(key):
            obs=self.nrn.main(key,stimval,f,stimweights,W,p,G)
            ref=self.nrn.main(key,refval,f,refweights,W,p,G)
            lAll,Yc=self._lik_parts(self._likelihoods_heldout(obs,ref,refweights,Y,refy))
            return self.objective.lrn_main(lAll,stimweights,yCtg,Y,refweights,Yc),None

        return self._noise_average(one,rng_key)[0]

    def evaluate(self,stim):
        """
        cost of decoding other stimuli (e.g. a held-out test set) with the current filters. This unit's stimuli are the
        training set: the category response distributions (AMA-Gauss), the reference stimuli (full AMA), the prior,
        and the whitening all come from them.
        """
        test=self._prepare_stim(stim)
        return float(self._loss_fun_heldout(self._params_out(),self.rng,test.val,test.weights,test.yCtg,test.Y,
                                            self.stim.val,self.stim.weights,self.stim.yCtg))

    def _log_posterior(self,stim=None):
        # (log posterior, the decoded Stim, the within-category values Yc or None)
        f=self.filter.out_flat
        if stim is None:
            lAll,Yc=self._lik_parts(self._likelihoods(self._nrn_out(),self.stim.weights,self.stim.Y,yRef=self._yRef(self.stim.yCtg)))
            return Objective._posterior__true(lAll,self.stim.weights),self.stim,Yc
        test=self._prepare_stim(stim)
        W=self.nrn.whitening(self.stim.val,f,self.stim.weights) if self.nrn._bWhiten else None
        G=self.nrn.gain(self.stim.val,f,self.stim.weights,W,self._p())
        obs=self.nrn.main(self.rng,test.val,f,test.weights,W,self._p(),G)
        ref=self.nrn.main(self.rng,self.stim.val,f,self.stim.weights,W,self._p(),G)
        lAll,Yc=self._lik_parts(self._likelihoods_heldout(obs,ref,self.stim.weights,self.stim.Y,self._yRef(self.stim.yCtg)))
        return Objective._posterior__true(lAll,self.stim.weights),test,Yc

    def estimates(self,estType='mode',stim=None):
        """
        estimates of the latent variable [ nStim_Ctg x nCtg (x nDim) ] (grouped like Stim.val; see Stim.weights) for the
        training stimuli, or for other stimuli decoded with the training set: 'mode' (MAP), 'mean' (MMSE; circular on
        circular dimensions), 'median', or 'cmean' (circular mean, Y in radians). With Model bWithin, continuous
        estimates within the categories
        """
        lpost,st,Yc=self._log_posterior(stim)
        return np.asarray(getattr(Objective,'_est__'+estType)(lpost,st.Y,st.Yperiod,Yc))

    def performance(self,estType='mode',stim=None):
        """
        estimation performance per latent level for the training stimuli, or for other stimuli decoded with the training
        set: bias, sd, and rmse of the estimates ([ nCtg (x nDim) ]; errors wrap on circular dimensions) and over all
        stimuli (rmseAll); pCorrect and confusion [ true x MAP category ] of the MAP category; and cost, the mean -log
        posterior at the correct level. Errors are measured from each stimulus's own latent value (Stim y), which for
        stimuli without their own values is their category's level
        """
        lpost,st,Yc=self._log_posterior(stim)
        lpost=np.asarray(lpost)
        est=np.asarray(getattr(Objective,'_est__'+estType)(jnp.asarray(lpost),st.Y,st.Yperiod,Yc))
        w=np.asarray(st.weights)>0
        Y=np.asarray(st.Y)
        y=np.asarray(st.yCtg)
        nCtg=len(Y)
        best=np.argmax(lpost,axis=-1)
        confusion=np.stack([np.bincount(best[w[:,c],c],minlength=nCtg) for c in range(nCtg)])
        err=[np.asarray(_wrap(jnp.asarray(est[w[:,c],c]-y[w[:,c],c]),st.Yperiod)) for c in range(nCtg)]
        correct=np.diagonal(lpost,axis1=-2,axis2=-1)
        return {'Y':Y,
                'estimates':est,
                'bias':np.array([e.mean(0) for e in err]),
                'sd':np.array([e.std(0) for e in err]),
                'rmse':np.array([np.sqrt(np.mean(e**2,0)) for e in err]),
                'rmseAll':np.sqrt(np.mean(np.concatenate(err)**2,0)),
                'pCorrect':np.diagonal(confusion)/confusion.sum(1),
                'confusion':confusion,
                'cost':float(-correct[w].mean())}

    def cross_validate(self,n,k=5,seed=0,**train_kw):
        """
        k-fold cross-validation, stratified by category: train n new filters on each training fold with copies of this
        unit's settings, and evaluate them on the held-out fold. Returns train and test costs [ k ] and filters [ k x ... ].
        """
        res={'train':[],'test':[],'filters':[]}
        for i,(train,test) in enumerate(self.stim_full.folds(k,seed)):
            unit=Unit(train,self.nrn.copy(),self.model.copy(),self.objective.copy(),
                      optimizer=None if self.optimizer is None else self.optimizer.copy(),seed=self.seed+i)
            unit.train_new(n,**train_kw)
            res['train'].append(float(unit.loss))
            res['test'].append(unit.evaluate(test))
            res['filters'].append(np.asarray(unit.out))
        return {key:np.array(val) for key,val in res.items()}

    #- save/load
    def save(self,fname):
        """save settings, filters, optimizer state, random keys, and frozen whitening (not the stimuli) with pickle"""
        if not self.nrn.bFinalized:
            raise Exception('train (or finalize) the unit first')
        asnp=lambda tree: None if tree is None else tree_util.tree_map(np.asarray,tree)
        state={'version':1,
               'nrn':_get_copy_dict(self.nrn,_NRN_EXCL),
               'model':_get_copy_dict(self.model),
               'objective':_get_copy_dict(self.objective),
               'optimizer':None if self.optimizer is None else _get_copy_dict(self.optimizer,_OPT_EXCL),
               'finalize':dict(n=self.filter.n,dtype=np.dtype(self.nrn.dtype).name,bFourier=self.nrn.bFourier,
                               bAnalytic=bool(self.nrn.bAnalytic),bSplit=self.nrn.bSplit,
                               stimInd=getattr(self,'_stimInd',None)),
               'out':np.asarray(self.filter.out),
               'last':asnp(self.filter.last),
               'opt_state':asnp(self.opt_state),
               'opt_param_shape':getattr(self,'_opt_param_shape',None),
               'opt_state_key':getattr(self,'_opt_state_key',None),
               'loss_hist':list(getattr(self.optimizer,'loss_hist',[])),
               'restart_costs':self.restart_costs,
               'seed':self.seed,
               'rng':np.asarray(jxrandom.key_data(self.rng)),
               'rng_last':None if self.rng_last is None else np.asarray(jxrandom.key_data(self.rng_last)),
               'W':asnp(self.nrn._W),
               'pool_p':asnp(self.pool_p),
               'nrn_p':asnp(getattr(self,'nrn_p',None)),
               'train_log':list(getattr(self,'train_log',[]) or []),
               'ama_source_sha256':source_sha256(),
               'name':getattr(self,'name',None),
               'multiscale_out':getattr(self,'multiscale_out',None),
               'param_out':getattr(self,'param_out',None),
               'stim_summary':self._stim_summary()}
        with open(fname,'wb') as fh:
            pickle.dump(state,fh)

    @classmethod
    def load(cls,fname,stim):
        """load a unit saved with Unit.save, with its training stimuli"""
        with open(fname,'rb') as fh:
            state=pickle.load(fh)
        asjnp=lambda tree: None if tree is None else tree_util.tree_map(jnp.asarray,tree)
        unit=cls(stim,Nrn(**state['nrn']),Model(**state['model']),Objective(**state['objective'],_bCopy=True),
                 optimizer=None if state['optimizer'] is None else Optimizer(**state['optimizer']),
                 seed=state['seed'],rng=jxrandom.wrap_key_data(state['rng']),
                 rng_last=None if state['rng_last'] is None else jxrandom.wrap_key_data(state['rng_last']),name=state.get('name'))
        fin=state['finalize']
        fourierType=(2 if fin['bAnalytic'] else 1) if fin['bFourier'] else 0
        if unit.optimizer is None:
            unit.optimizer=Optimizer()
        unit._finalize(fin['n'],np.arange(fin['n']),fourierType=fourierType,bSplit=fin['bSplit'],dtype=jnp.dtype(fin['dtype']),
                       stimInd=fin.get('stimInd'))
        if state['optimizer'] is None:
            unit.optimizer=None
        unit.filter.out=jnp.asarray(state['out'])
        unit.filter.last=asjnp(state['last'])
        unit.opt_state=asjnp(state['opt_state'])
        unit._opt_param_shape=state['opt_param_shape']
        unit._opt_state_key=state.get('opt_state_key')                   # older saves: the state is not reused
        if unit.optimizer is not None:
            unit.optimizer.loss_hist=state['loss_hist']
        unit.restart_costs=state['restart_costs']
        unit.nrn._W=asjnp(state['W'])
        unit.pool_p=None if state.get('pool_p') is None else np.asarray(state['pool_p'])
        unit.nrn_p=None if state.get('nrn_p') is None else {k:np.asarray(v) for k,v in state['nrn_p'].items()}
        unit.train_log=list(state.get('train_log') or [])
        unit.name=state.get('name')
        if state.get('multiscale_out') is not None:
            unit.multiscale_out=state['multiscale_out']
        if state.get('param_out') is not None:
            unit.param_out=state['param_out']
        return unit

    #- yaml configuration
    def _stim_summary(self):
        st=self.stim_full
        return {'dims':[int(d) for d in st.dims],'nCtg':int(st.nCtg),'nStim':int(np.sum(np.asarray(st.weights)>0)),
                'Y':_yaml_safe(np.asarray(st.Y)),'Yperiod':_yaml_safe(getattr(st,'Yperiod',None)),
                'bContinuous':bool(getattr(st,'bContinuous',False)),
                'bIsFourier':bool(st.bIsFourier),'nSplit':int(st.nSplit or 0)}

    def config(self):
        """
        this unit's options and settings as a plain dict (yaml-safe): Nrn, Model, Objective and Optimizer settings, the
        seed, the filter layout, a summary of the training stimuli, and the training calls made so far (method and
        arguments, from train_new / train_recurse / train_append / train_parametric / train_multiscale). Rebuild with
        Unit.from_config. Learned filters are not included (use save / load).
        """
        cfg={'ama_config_version':CONFIG_VERSION,'ama_source_sha256':source_sha256(),'name':getattr(self,'name',None),
             'seed':_yaml_safe(self.seed)}
        cfg.update(_components_config(self.nrn,self.model,self.objective,self.optimizer))
        if self.nrn.bFinalized:
            cfg['filters']={'n':int(self.filter.n),'fourierType':(2 if self.nrn.bAnalytic else 1) if self.nrn.bFourier else 0,
                            'dtype':np.dtype(self.nrn.dtype).name,'bSplit':bool(self.nrn.bSplit),
                            'shape':list(np.shape(self.filter.out)),'bPoolWeights':self.pool_p is not None}
        cfg['train']=_yaml_safe(list(getattr(self,'train_log',[]) or []))
        cfg['stim']=self._stim_summary()
        return cfg

    def save_config(self,fname):
        """write config() to a yaml file"""
        import yaml
        with open(fname,'w') as fh:
            yaml.safe_dump(self.config(),fh,sort_keys=False,default_flow_style=None,width=120)

    @classmethod
    def from_config(cls,cfg,stim,bTrain=False,stimVal=None):
        """
        build a unit from a configuration (dict, or the path of a yaml file written by save_config / config_from_saved)
        with these training stimuli. bTrain replays the recorded training calls in order; stimVal is passed to calls that
        were made with validation stimuli. Unknown top-level keys (e.g. project metadata) are ignored.
        """
        if isinstance(cfg,str) or hasattr(cfg,'__fspath__'):
            cfg=load_config(cfg)
        if int(cfg.get('ama_config_version',1))>CONFIG_VERSION:
            raise Exception('config version ' + str(cfg.get('ama_config_version')) + ' is newer than this ama (' + str(CONFIG_VERSION) + ')')
        opt=cfg.get('optimizer')
        unit=cls(stim,Nrn(**cfg['nrn']),Model(**cfg['model']),Objective(**cfg['objective'],_bCopy=True),
                 optimizer=None if opt is None else Optimizer(**opt),seed=cfg.get('seed'),name=cfg.get('name'))
        if bTrain:
            for step in cfg.get('train') or []:
                args=dict(step.get('args') or {})
                if 'stimVal' in args:
                    args['stimVal']=stimVal if args['stimVal'] else None
                if args.get('optimizer') is not None:
                    args['optimizer']=Optimizer(**args['optimizer'])
                if args.get('dtype') is not None:
                    args['dtype']=jnp.dtype(args['dtype'])
                if step['method'] not in ('train_new','train_recurse','train_append','train_parametric','train_multiscale'):
                    raise Exception('unknown training method in config: ' + str(step['method']))
                getattr(unit,step['method'])(**args)
        return unit

    #- whitening
    def freeze_whitening(self):
        """
        fix the whitening matrix at its value for the current filters and stimuli (e.g. the training set), so that other
        stimuli (Unit.split) are evaluated with it rather than with their own. Training clears it.
        """
        if not self.nrn._bWhiten:
            raise Exception("freeze_whitening requires Nrn whitenType 'gram' or 'response'")
        self.nrn._W=self.nrn.whitening(self.stim.val,self.filter.out_flat,self.stim.weights)

    def unfreeze_whitening(self):
        self.nrn._W=None

    #- properties
    @property
    def filter(self):
        return self.nrn.filter

    @property
    def out(self):
        return self.filter.out

    @property
    def last(self):
        return self.filter.last

    @property
    def loss(self):
        return self._loss_fun(self._params_out(),self.rng,self.stim.val,self.stim.weights,self.stim.yCtg,self.stim.Y,self.nrn._W)

    @property
    def penalty(self):
        """filter penalty of Objective regType (without regWeight), which training adds to the cost"""
        if str(self.objective.regType).lower()=='none':
            return 0.
        return float(self._penalty(self.filter.out_flat))

    def _p(self):
        if not self.nrn.bParams:
            return None
        p=self._p_old()
        if not p:
            return None
        return {k:jnp.asarray(v) for k,v in p.items()}

    def _params_out(self):
        out={'f':self.filter.out_flat}
        if self._p() is not None:
            out['p']=self._p()
        return out

    def _penalty(self,f):
        """mean over filters of the regType penalty of the learned coefficients; f [ nPix x (nSplit) x nF ]"""
        flt=self.filter
        F=jnp.reshape(f,flt._shape_exp)
        nLead=len(flt.pix_dims)
        mask=np.zeros(int(np.prod(flt.pix_dims)),dtype=bool)
        mask[flt.index.pix]=True
        mask=np.reshape(mask,flt.pix_dims+(1,)*(F.ndim-nLead))
        reg=str(self.objective.regType).lower()
        if reg=='l1':
            tot=jnp.sum(jnp.abs(F)*mask)
        else:
            tot=0.
            for a in range(nLead):
                valid=mask & np.roll(mask,-1,axis=a)
                edge=np.ones(flt.pix_dims[a],dtype=bool)
                edge[-1]=False
                valid=valid & np.reshape(edge,tuple(-1 if i==a else 1 for i in range(F.ndim)))
                tot=tot+jnp.sum(jnp.abs(F-jnp.roll(F,-1,axis=a))**2*valid)
        return tot/flt.n

    @property
    def responses(self):
        return Response(*self.nrn.main(self.rng,self.stim.val,self.filter.out_flat,self.stim.weights,self.nrn._W,self._p()),self.stim)

    def _nrn_out(self):
        return self.nrn.main(self.rng,self.stim.val,self.filter.out_flat,self.stim.weights,self.nrn._W,self._p())

    @property
    def likelihoods(self):
        return self._likelihoods(self._nrn_out(),self.stim.weights,self.stim.Y)

    @property
    def posterior(self):
        return self.objective._posterior_fun(self.likelihoods,self.stim.weights)

    @property
    def error(self):
        Y,per=self.stim.Y,self.objective._Yperiod
        lAll,Yc=self._lik_parts(self._likelihoods(self._nrn_out(),self.stim.weights,Y,yRef=self._yRef(self.stim.yCtg)))
        lpost=self.objective._posterior_fun(lAll,self.stim.weights)
        return self.objective._err_fun(self.objective._est_fun(lpost,Y,per,Yc),self.stim.yCtg,self.objective.log_target(Y,self.stim.yCtg),
                                       Y,per,self.objective._err_opts())

    #- loss functions
    def _yRef(self,yCtg):
        # the reference stimuli's own latent values, which Model bWithin uses
        return yCtg if self.model.bWithin else None

    @staticmethod
    def _lik_parts(lik):
        # (lAll, Yc) from a likelihood result: lAll, or (lAll, Yc) with Model bWithin
        return lik if isinstance(lik,tuple) else (lik,None)

    def _likelihoods(self,nrn_out,stimweights,Y,rng_ref=None,mstate=None,yRef=None,nbr=None,obs=None):
        """
        log-likelihoods lAll, or (lAll, Yc) with yRef (Model bWithin). rng_ref: the key of the random references (nRef, or
        the tail of nNeighbors); mstate: a mixture fit to warm start from, then (result, the new fit); nbr: the nearest
        references (nNeighbors); obs: (indices [ mMax x nCtg ], weights) of a batch of stimuli to decode against all of
        them (nNeighbors with batchSize), lAll [ mMax x nCtg x nCtg ]
        """
        R,Rm,RVar=[_flatten_responses(x) for x in self.model._response_fun(*nrn_out)]
        noiseCov=self.nrn._corr_fun(RVar,stimweights,self.nrn.rho)
        if mstate is not None:
            # a mixture warm started from mstate: (likelihoods, the new fit)
            lAll,mnew=Model._mix_likelihoods(R,Rm,noiseCov,stimweights,self.model.bLeaveOneOut,self.model,mstate,RVar,Y)
            return Model._with_within(lAll,R,Rm,noiseCov,stimweights,yRef,Y,self.model,self.model.bLeaveOneOut,RVar),mnew
        noiseCorr=self.nrn.corr_matrix(R.shape[0],R.dtype) if self.nrn.corrType=='corr' else None
        if nbr is not None:
            # full AMA from the nearest references and a random tail of each category
            tail=Model._ref_subset(rng_ref,stimweights,min(self.model.nTail,stimweights.shape[0]))
            obsIdx=None
            if obs is not None:
                obsIdx=obs[0]
                R=jnp.take_along_axis(R,jnp.broadcast_to(obsIdx,R.shape[:-2]+obsIdx.shape),axis=-2)
                nbr=nbr[obsIdx,jnp.arange(nbr.shape[1])[None,:]]                 # [ mMax x nCtg x nCtg x K ]
            return Model._full_neighbors(R,Rm,RVar,noiseCorr,stimweights,nbr,tail,self.model.bLeaveOneOut,yRef,Y,self.model._Yperiod,
                                         obsIdx)
        if rng_ref is not None:
            # full AMA with nRef: decode against a random subset of the reference stimuli
            idx=Model._ref_subset(rng_ref,stimweights,self.model.nRef)
            take=lambda A: jnp.take_along_axis(A,jnp.broadcast_to(idx,A.shape[:-2]+idx.shape),axis=-2)
            return self.model.lrn_main(R,take(Rm),take(RVar),noiseCov,noiseCorr,take(stimweights),Y,idx,
                                       None if yRef is None else _take_stim(yRef,idx))
        return self.model.lrn_main(R,Rm,RVar,noiseCov,noiseCorr,stimweights,Y,None,yRef)

    def _bRefSubset(self):
        return self.model.modelType=='full' and self.model.nRef is not None

    def _bNeighbors(self):
        return self.model.modelType=='full' and self.model.nNeighbors is not None

    def _likelihoods_heldout(self,obs_out,ref_out,refweights,Y,refy=None):
        # observed responses of other stimuli, decoded against the reference (training) stimuli; no leave-one-out
        R=_flatten_responses(self.model._response_fun(*obs_out)[0])
        _,Rm,RVar=[_flatten_responses(x) for x in self.model._response_fun(*ref_out)]
        noiseCov=self.nrn._corr_fun(RVar,refweights,self.nrn.rho)
        noiseCorr=self.nrn.corr_matrix(R.shape[0],R.dtype) if self.nrn.corrType=='corr' else None
        return self.model._model_fun(R,Rm,RVar,noiseCov,noiseCorr,refweights,False,self.model,Y,None,
                                     refy if self.model.bWithin else None)

    @partial(jit, static_argnames=['self'])
    def _loss_fun_lrn(self,params,rng_key,prepped,index,stimval,stimweights,yCtg,Y,mstate=None):
        # the reference subset has its own key, so the response noise draws do not change with nRef. mstate: the mixture
        # fit (Model bWarmEM), or the nearest references (nNeighbors), which come back unchanged
        # with batches, nNeighbors' mstate is (nearest references, batch indices, batch weights): the batch is decoded
        # against all stimuli, which set the prior
        rng_ref=jxrandom.fold_in(rng_key,7) if (self._bRefSubset() or self._bNeighbors()) else None
        bNbr=self._bNeighbors()
        obs=mstate[1:] if bNbr and isinstance(mstate,tuple) else None
        nbr=(mstate[0] if obs is not None else mstate) if bNbr else None

        def one(noise_key):
            lik=self._likelihoods(self.nrn.lrn_main(noise_key,stimval,params['f'],prepped,index,stimweights,params.get('p')),
                                  stimweights,Y,rng_ref,None if bNbr else mstate,self._yRef(yCtg),nbr,obs)
            if bNbr:
                lik,mnew=lik,mstate
            else:
                lik,mnew=lik if mstate is not None else (lik,None)
            lAll,Yc=self._lik_parts(lik)
            if obs is None:
                return self.objective.lrn_main(lAll,stimweights,yCtg,Y,None,Yc),mnew
            return self.objective.lrn_main(lAll,obs[1],_take_stim(yCtg,obs[0]),Y,stimweights,Yc),mnew

        noise_key=jxrandom.key(self.model.noiseSeed) if self.model.bFixedNoise else rng_key
        cost,mnew=self._noise_average(one,noise_key)
        if self.objective.regWeight>0 and str(self.objective.regType).lower()!='none':
            cost=cost+self.objective.regWeight*self._penalty(self.nrn.insert(params['f'],prepped,index))
        return cost if mstate is None else (cost,mnew)

    def _noise_average(self,fun,key):
        """
        fun(key) -> (cost, aux), averaged over Model nNoiseSamples noisy observations (independent keys); aux of the first.
        One observation when nNoiseSamples is 1
        """
        S=self.model.nNoiseSamples
        if S<=1:
            return fun(key)
        keys=jxrandom.split(key,S)
        # full AMA's memory is bounded per chunk: its samples run one after the other, the others' in one batch
        cost,aux=lax.map(fun,keys) if self.model.modelType=='full' else jax.vmap(fun)(keys)
        return jnp.mean(cost),tree_util.tree_map(lambda a: a[0],aux)

    @partial(jit, static_argnames=['self'])
    def _nbr_state(self,params,prepped,index,stimval,stimweights):
        # the nearest references of every stimulus (nNeighbors) under the current filters, from the mean responses
        out=self.nrn.lrn_main(jxrandom.key(0),stimval,params['f'],prepped,index,stimweights,params.get('p'))
        _,Rm,RVar=[_flatten_responses(x) for x in self.model._response_fun(*out)]
        noiseCorr=self.nrn.corr_matrix(Rm.shape[0],Rm.dtype) if self.nrn.corrType=='corr' else None
        return Model._full_search(Rm,Rm,RVar,noiseCorr,stimweights,self.model.nNeighbors,self.model.bLeaveOneOut)

    def _mix_state0(self,f0):
        # the cold mixture fit at the initial filters, where a warm-started EM (Model bWarmEM) begins
        if self.model.modelType!='mix' or not self.model.bWarmEM:
            return None
        out=self.nrn.lrn_main(self.rng,self.stim.val,f0,self.filter.prepped_jx,self.filter._insert_index_jx,self.stim.weights,
                              self._p0() if self.nrn.bParams else None)
        Rm=_flatten_responses(self.model._response_fun(*out)[1])
        return tuple(lax.stop_gradient(v) for v in Model._mix_fit(Rm,self.stim.weights,self.model,bDetail=True,Y=self.stim.Y)[1]['state'])

    @partial(jit, static_argnames=['self'])
    def _loss_fun(self,params,rng_key,stimval,stimweights,yCtg,Y,W=None):
        def one(key):
            lAll,Yc=self._lik_parts(self._likelihoods(self.nrn.main(key,stimval,params['f'],stimweights,W,params.get('p')),stimweights,Y,
                                                      yRef=self._yRef(yCtg)))
            return self.objective.lrn_main(lAll,stimweights,yCtg,Y,None,Yc),None
        return self._noise_average(one,rng_key)[0]

    #- plot
    def plot_out(self,bFourier=None,name='f_out'):
        self.filter.plot_out(bFourier=bFourier,name=name)

    def plot_last(self,bFourier=None,name='f_last'):
        self.filter.plot_last(bFourier=bFourier,name=name)

    def _bank_info(self,bankInfo=None):
        """filter-bank metadata for the figures: multiscale_values() or parametric_values(), else bankInfo (a dict with any of
        k_peak, orientations, interleaved, mother, log_freq_knots, nMothers, mother_index), else None"""
        info=None
        for attr in ('multiscale_out','param_out'):
            if getattr(self,attr,None) is not None:
                info=dict(getattr(self,attr))
                break
        if info is None and bankInfo is not None:
            info=dict(bankInfo)
        if info is None:
            return None
        n=int(self.filter.n)
        if 'k_peak' in info and np.size(info['k_peak'])!=n:
            info.pop('k_peak')
        M=info.get('mother')
        info.setdefault('nMothers',1 if M is None or np.ndim(M)==1 else np.shape(M)[0])
        info.setdefault('mother_index',np.repeat(np.arange(info['nMothers']),n//info['nMothers']))
        return info

    def _figure_title(self,name,what):
        name=self.name if name is None else name
        return what if not name else str(name) + '\n' + what

    def plot_filter_bank(self,fname=None,name=None,bankInfo=None,perRow=8,dpi=110):
        """
        figure of the learned filter bank. 1D: one row of implied spatial filters per mother filter (real part solid,
        quadrature pair dotted, in that mother's colour), then a log-frequency and a linear-frequency row in which each
        column (scale) overlays the amplitude spectra of all mothers at that scale; side panels with the pooling weights
        (readoutType resultant/resultant_only, bars in the mother colours), the mother filter(s) of a multiscale bank
        (amplitude solid and phase/pi dotted against log(k/k_peak)), and the pooling-weighted sum of the spectra of each
        scale on linear and log frequency axes. 2D: real part and amplitude spectrum (linear frequency axes) of every
        filter, with peak frequency, orientation and interleaving in the panel titles. name (default unit.name) titles the
        figure; bankInfo supplies bank metadata for units without multiscale_values()/parametric_values(). Saves to fname
        when given; returns the figure.
        """
        flt=self.filter
        dims=tuple(flt.pix_dims)
        g=np.asarray(flt.implied_spatial())
        parts=None
        if flt.bSplit:                                                    # one column per sub-filter (e.g. per eye)
            nSp=int(flt.nSplit)
            g=np.reshape(g,tuple(flt.pix_dims)+(-1,))
            parts=np.tile(np.arange(nSp),g.shape[-1]//nSp) if g.shape[-1]%nSp==0 else None
        n=g.shape[-1]
        F=np.asarray(self.out).reshape(dims+(n,)) if flt.bIsFourier and not flt.bSplit else None
        w=None if self.pool_p is None else np.asarray(Nrn.pool_weights(np.asarray(self.pool_p,float).ravel()))
        mv=self._bank_info(bankInfo)
        per_row=perRow if n%perRow==0 else min(n,perRow)
        rows=int(np.ceil(n/per_row))
        kp=lambda j: '' if mv is None or 'k_peak' not in mv else ' k=%.3f'%np.asarray(mv['k_peak'])[j]
        if len(dims)==1:
            N=dims[0]
            x=np.arange(N)-N/2
            f=np.fft.fftshift(np.fft.fftfreq(N))
            pos=f>0
            amp=np.abs(F) if F is not None else np.abs(np.fft.fftshift(np.fft.fft(np.real(g),axis=0),axes=0))
            nM=int(mv['nMothers']) if mv is not None else 1
            nS=max(1,n//nM)
            midx=np.asarray(mv['mother_index']) if mv is not None else np.zeros(n,int)
            colors=[cm.tab10(i%10) for i in range(nM)]
            # rows: one spatial row per mother, then the log- and linear-frequency rows (all mothers of a scale overlaid)
            fig=plt.figure(figsize=(2.2*nS+3.6,2.6*max(nM+2,4)+1.4))
            outer=fig.add_gridspec(1,2,width_ratios=[nS,1.7],wspace=0.18)
            gs=outer[0,0].subgridspec(nM+2,nS)                             # filters: nM spatial rows, then log and linear
            side=outer[0,1].subgridspec(4,1,hspace=0.45)                   # weights, mother(s), weighted sums (linear, log)
            for j in range(n):
                m_,s_=(int(midx[j]),j%nS) if mv is not None else (0,j)
                ax=fig.add_subplot(gs[m_,s_])
                ax.plot(x,np.real(g[:,j]),lw=0.8,color=colors[m_],label='real')
                ax.plot(x,np.imag(g[:,j]),lw=0.8,ls=':',color=colors[m_],label='quadrature')
                ttl='f'+str(j)+(' m'+str(m_) if nM>1 else '')+('' if parts is None else ' part'+str(parts[j]))+kp(j)
                if w is not None and w.size==n:
                    ttl+='\nw=%.2f'%w[j]
                ax.set_title(ttl,fontsize=6)
                ax.tick_params(labelsize=5)
                ax.set_xlabel('px',fontsize=5)
                if j==0:
                    ax.legend(fontsize=5)
            for s_ in range(nS):
                js=[j for j in range(n) if (int(midx[j]),j%nS)[1]==s_] if mv is not None else [s_]
                for row,(scale,xlab) in enumerate(((True,'cycles/px (log)'),(False,'cycles/px (linear)'))):
                    ax=fig.add_subplot(gs[nM+row,s_])
                    for j in js:
                        (ax.semilogx if scale else ax.plot)(f[pos],amp[pos,j],lw=0.8,color=colors[int(midx[j])])
                    ax.set_xlim(f[pos][0] if scale else 0,0.5)
                    ax.tick_params(labelsize=5)
                    ax.set_xlabel(xlab,fontsize=5)
                    if s_==0:
                        ax.set_ylabel('amplitude',fontsize=5)
            ax=fig.add_subplot(side[0])
            if w is not None:
                ax.bar(np.arange(w.size),w,color=[colors[int(midx[j])] for j in range(min(n,w.size))])
                ax.set_title('pooling weights',fontsize=7)
            else:
                ax.text(0.5,0.5,'no pooling readout',ha='center',va='center',fontsize=7)
                ax.axis('off')
            ax.tick_params(labelsize=6)
            ax=fig.add_subplot(side[1])
            if mv is not None and mv.get('mother') is not None and 'log_freq_knots' in mv:
                M=np.asarray(mv['mother'])
                M=M[None] if M.ndim==1 else M
                for im in range(M.shape[0]):
                    ax.plot(mv['log_freq_knots'],np.abs(M[im]),lw=1,color=colors[im])
                    ax.plot(mv['log_freq_knots'],np.angle(M[im])/np.pi*np.abs(M[im]).max(),lw=0.7,ls=':',color=colors[im])
                handles=[plt.Line2D([],[],color='k',lw=1,label='amplitude'),plt.Line2D([],[],color='k',lw=0.7,ls=':',label='phase')]
                handles+=[plt.Line2D([],[],color=colors[im],lw=1,label='mother '+str(im)) for im in range(M.shape[0])]
                ax.legend(handles=handles,fontsize=5)
                ax.set_xlabel('log(k / k_peak)',fontsize=6)
                ax.set_title('mother filter(s)',fontsize=7)
            else:
                for j in range(n):
                    ax.semilogx(f[pos],amp[pos,j],lw=0.7,color=colors[int(midx[j])])
                ax.set_title('all amplitude spectra (log frequency)',fontsize=7)
                ax.set_xlabel('cycles/px (log)',fontsize=6)
            ax.tick_params(labelsize=6)
            # weighted sum of each scale (pooling weights; uniform when there is no readout), linear above log
            ww=np.ones(n)/n if w is None or w.size!=n else w
            scale_sum=np.stack([sum(ww[j]*amp[:,j] for j in range(n) if (j%nS)==s_) for s_ in range(nS)],1)
            scolors=[cm.viridis(i/max(nS-1,1)) for i in range(nS)]
            for row,scale in ((2,False),(3,True)):
                ax=fig.add_subplot(side[row])
                for s_ in range(nS):
                    (ax.semilogx if scale else ax.plot)(f[pos],scale_sum[pos,s_],lw=0.8,color=scolors[s_],
                                                        label=('scale '+str(s_)) if not scale else None)
                (ax.semilogx if scale else ax.plot)(f[pos],scale_sum[pos].sum(1),lw=1.1,color='k',label='total' if not scale else None)
                ax.set_xlim(f[pos][0] if scale else 0,0.5)
                ax.set_title('weighted sum per scale ('+('log' if scale else 'linear')+' frequency)',fontsize=7)
                ax.set_xlabel('cycles/px ('+('log' if scale else 'linear')+')',fontsize=6)
                ax.tick_params(labelsize=6)
                if not scale:
                    ax.legend(fontsize=5,ncol=2)
        elif len(dims)==2:
            fig=plt.figure(figsize=(1.6*per_row+2.5,3.4*rows+1.2))
            gs=fig.add_gridspec(2*rows,per_row+1)
            for j in range(n):
                r_,c_=divmod(j,per_row)
                ax=fig.add_subplot(gs[2*r_,c_])
                v=np.real(g[...,j])
                lim=np.abs(v).max()
                ax.imshow(v,cmap='RdBu_r',vmin=-lim,vmax=lim)
                ori='' if mv is None or 'orientations' not in mv else ' %d deg'%round(np.degrees(np.asarray(mv['orientations'])[j]))
                il='*' if mv is not None and 'interleaved' in mv and np.asarray(mv['interleaved'])[j] else ''
                ax.set_title('f'+str(j)+il+kp(j)+ori,fontsize=5)
                ax.axis('off')
                ax=fig.add_subplot(gs[2*r_+1,c_])
                A=np.abs(F[...,j]) if F is not None else np.abs(np.fft.fftshift(np.fft.fft2(v)))
                ax.imshow(A,cmap='magma',extent=(-0.5,0.5,0.5,-0.5))
                ax.set_xticks([-0.5,0,0.5])
                ax.set_yticks([-0.5,0,0.5])
                ax.tick_params(labelsize=4)
                if j==0:
                    ax.set_xlabel('cycles/px (linear)',fontsize=5)
            ax=fig.add_subplot(gs[:,per_row])
            if w is not None:
                ax.barh(np.arange(w.size),w)
                ax.set_title('pooling weights',fontsize=7)
            else:
                ax.text(0.5,0.5,'no pooling readout\n* = interleaved\nrows: real part,\none-sided |spectrum|',ha='center',va='center',fontsize=7)
                ax.axis('off')
        else:
            raise Exception('plot_filter_bank supports 1D and 2D filters')
        fig.suptitle(self._figure_title(name,'filters'),fontsize=9)
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname,dpi=dpi)
        return fig

    @staticmethod
    def _embed(feats,method,seed,**kw):
        """2D embedding of [ nStim x nFeatures ]: 't-sne' / 'tsne' (sklearn), 'pacmap', or 'phate'"""
        method=method.lower().replace('-','')
        if method=='tsne':
            from sklearn.manifold import TSNE
            kw.setdefault('perplexity',30)
            return TSNE(n_components=2,init='pca',learning_rate='auto',random_state=seed,**kw).fit_transform(feats),'t-SNE'
        if method=='pacmap':
            import pacmap
            kw.setdefault('n_neighbors',10)
            return pacmap.PaCMAP(n_components=2,random_state=seed,**kw).fit_transform(feats,init='pca'),'PaCMAP'
        if method=='phate':
            import phate
            kw.setdefault('knn',10)
            return phate.PHATE(n_components=2,random_state=seed,verbose=0,**kw).fit_transform(feats),'PHATE'
        raise Exception("method must be 'tsne', 'pacmap' or 'phate'")

    def _response_features(self,stim,nMax,seed,cmap=None):
        """(features [ nStim x nFeatures ], responses u [ nF x nStim ], latent [ nStim ], cmap) for the embeddings:
        noise-free responses of every filter to up to nMax category-balanced stimuli, as unit phasors for normalizeType
        'phase'"""
        st=copy.copy(stim)
        if st.bIsFourier:
            st._ifft()
        if self.filter.bSplit and not st.bIsSplit:
            st.nSplit=int(self.filter.nSplit)                              # stimuli built without nSplit: the filters know it
            st.split()
        elif st.bIsSplit and not self.filter.bSplit:
            st.unsplit()
        rng=np.random.default_rng(seed)
        valid=np.asarray(st.weights)>0                                    # [ nStim_Ctg x nCtg ]
        per=max(1,nMax//st.nCtg)
        cols=[]
        for c in range(st.nCtg):
            idx=np.flatnonzero(valid[:,c])
            k=min(per,idx.size)
            cols.append(np.stack([rng.choice(idx,k,replace=False),np.full(k,c)],1))
        sel=np.concatenate(cols)
        Yv=np.asarray(st.Y) if np.ndim(st.Y)==1 else np.asarray(st.Y)[:,0]      # color by the first latent dimension
        lat=Yv[sel[:,1]]
        g=np.asarray(self.filter.implied_spatial())
        if self.filter.bSplit:
            # each sub-filter (e.g. one eye) responds as its own neuron: [ nSplit x nF ] response dimensions
            nS=int(self.filter.nSplit)
            nP=st.nPix//nS
            g=g.reshape(nP,nS,-1)
            S=np.real(np.asarray(st.val).reshape(nP,nS,st.nStim_Ctg,st.nCtg)[:,:,sel[:,0],sel[:,1]])
            r=np.einsum('psj,psn->sjn',g,S).reshape(-1,S.shape[-1])
        else:
            S=np.real(np.asarray(st.val).reshape(st.nPix,st.nStim_Ctg,st.nCtg)[:,sel[:,0],sel[:,1]])
            r=g.reshape(st.nPix,-1).T@S                                   # [ nF x nStim ]
        u=r/np.maximum(np.abs(r),1e-12) if self.nrn.normalizeType=='phase' else r
        feats=(np.concatenate([u.real,u.imag],0) if np.iscomplexobj(u) else u).T
        if cmap is None:
            bCirc=getattr(st,'Yperiod',None) is not None and st.Yperiod[0] is not None
            cmap='twilight' if bCirc or (Yv.min()>=0 and Yv.max()<2*np.pi+1e-9 and np.ptp(Yv)>np.pi) else 'viridis'
        return feats,u,lat,cmap

    def plot_response_embeddings(self,stim,methods=('tsne','pacmap','phate'),fname=None,name=None,nMax=2000,seed=0,cmap=None,
                                 dpi=110,method_kw=None,**embed_kw):
        """
        one figure with every embedding of the same responses side by side (t-SNE, PaCMAP, PHATE by default), and, with
        pooling weights, the pooled resultant as a final panel. The responses (and the colour scale) are computed once, so
        the panels are comparable. method_kw: {method: its own keyword arguments}; embed_kw apply to every method.
        name (default unit.name) titles the figure. Saves to fname when given; returns the figure.
        """
        feats,u,lat,cmap=self._response_features(stim,nMax,seed,cmap)
        w=None if self.pool_p is None else np.asarray(Nrn.pool_weights(np.asarray(self.pool_p,float).ravel()))
        bPooled=w is not None and w.size==u.shape[0] and np.iscomplexobj(u)
        method_kw=method_kw or {}
        embs=[self._embed(feats,m,seed,**{**embed_kw,**method_kw.get(m,{})}) for m in methods]
        nax=len(embs)+(1 if bPooled else 0)
        fig,axes=plt.subplots(1,nax,figsize=(5.2*nax,5.3),squeeze=False)
        for i,(emb,label) in enumerate(embs):
            sc=axes[0,i].scatter(emb[:,0],emb[:,1],c=lat,cmap=cmap,s=4)
            axes[0,i].set_title(label,fontsize=9)
            axes[0,i].set_xticks([])
            axes[0,i].set_yticks([])
            if i==len(embs)-1:
                fig.colorbar(sc,ax=axes[0,i],label='latent')
        if bPooled:
            z=(w[:,None]*u).sum(0)
            axes[0,-1].scatter(z.real,z.imag,c=lat,cmap=cmap,s=4)
            axes[0,-1].set_aspect('equal')
            axes[0,-1].set_title('pooled resultant sum_j p_j r_j',fontsize=9)
            axes[0,-1].tick_params(labelsize=6)
        fig.suptitle(self._figure_title(name,('phase-normalized ' if self.nrn.normalizeType=='phase' else '')
                                        +'responses of '+str(u.shape[0])+' filters, '+str(feats.shape[0])+' stimuli'),fontsize=9)
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname,dpi=dpi)
        return fig

    def plot_response_embedding(self,stim,method='tsne',fname=None,name=None,nMax=2000,seed=0,cmap=None,dpi=110,**embed_kw):
        """
        2D embedding of the noise-free responses of every filter to up to nMax category-balanced stimuli (a Stim, e.g.
        held-out test stimuli; fourier-domain stimuli are transformed back), coloured by the latent value. method: 'tsne'
        (sklearn t-SNE, perplexity 30, PCA initialization), 'pacmap' (PaCMAP, n_neighbors 10, PCA initialization) or 'phate'
        (PHATE, knn 10); embed_kw go to the method. Responses are unit phasors (real and imaginary parts of r_j / |r_j|) for
        normalizeType 'phase', else the responses themselves. With pooling weights (readoutType resultant/resultant_only) a
        second panel shows the pooled resultant sum_j p_j r_j (what the likelihood sees) in the complex plane. cmap defaults
        to 'twilight' (circular latents) when Y spans more than pi within [0, 2 pi), else 'viridis'. name (default
        unit.name) titles the figure. Saves to fname when given; returns the figure.
        """
        feats,u,lat,cmap=self._response_features(stim,nMax,seed,cmap)
        emb,label=self._embed(feats,method,seed,**embed_kw)
        w=None if self.pool_p is None else np.asarray(Nrn.pool_weights(np.asarray(self.pool_p,float).ravel()))
        bPooled=w is not None and w.size==u.shape[0] and np.iscomplexobj(u)
        fig,axes=plt.subplots(1,2 if bPooled else 1,figsize=(11 if bPooled else 6,5.3),squeeze=False)
        sc=axes[0,0].scatter(emb[:,0],emb[:,1],c=lat,cmap=cmap,s=4)
        axes[0,0].set_title(label+' of '+('phase-normalized ' if self.nrn.normalizeType=='phase' else '')+'responses ('
                            +str(u.shape[0])+' filters, '+str(feats.shape[0])+' stimuli)',fontsize=8)
        axes[0,0].set_xticks([])
        axes[0,0].set_yticks([])
        fig.colorbar(sc,ax=axes[0,0],label='latent')
        if bPooled:
            z=(w[:,None]*u).sum(0)
            axes[0,1].scatter(z.real,z.imag,c=lat,cmap=cmap,s=4)
            axes[0,1].set_aspect('equal')
            axes[0,1].set_title('pooled resultant sum_j p_j r_j (what the likelihood sees)',fontsize=8)
            axes[0,1].tick_params(labelsize=6)
        fig.suptitle(self._figure_title(name,'response '+label),fontsize=9)
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname,dpi=dpi)
        return fig

    def plot_response_tsne(self,stim,fname=None,name=None,**kw):
        """plot_response_embedding with method 'tsne'"""
        return self.plot_response_embedding(stim,'tsne',fname=fname,name=name,**kw)

    def plot_response_pacmap(self,stim,fname=None,name=None,**kw):
        """plot_response_embedding with method 'pacmap'"""
        return self.plot_response_embedding(stim,'pacmap',fname=fname,name=name,**kw)

    def plot_response_phate(self,stim,fname=None,name=None,**kw):
        """plot_response_embedding with method 'phate'"""
        return self.plot_response_embedding(stim,'phate',fname=fname,name=name,**kw)

    def save_figures(self,stem,stim,name=None,bankInfo=None,methods=('tsne','pacmap','phate'),**embed_kw):
        """
        write <stem>_filters.png (plot_filter_bank) and <stem>_<method>.png (plot_response_embedding on stim) for each
        embedding method; closes the figures. With more than one method it also writes <stem>_embeddings.png, all of them in
        one figure (plot_response_embeddings). methods: a sequence of method names, or a dict of method -> its own keyword
        arguments (embed_kw, applying to every method, is for arguments they share, e.g. nMax and seed).
        """
        plt.close(self.plot_filter_bank(str(stem)+'_filters.png',name=name,bankInfo=bankInfo))
        items=dict(methods) if isinstance(methods,dict) else {m:{} for m in methods}
        for m,kw in items.items():
            plt.close(self.plot_response_embedding(stim,m,str(stem)+'_'+m.replace('-','')+'.png',name=name,**{**embed_kw,**kw}))
        if len(items)>1:
            plt.close(self.plot_response_embeddings(stim,tuple(items),str(stem)+'_embeddings.png',name=name,method_kw=items,**embed_kw))



class Response():

    def __init__(self,r,rNs,R,RNs,RVar,stim):
        self.r=r
        self.rNs=rNs
        self.R=R
        self.RNs=RNs
        self.RVar=RVar

        self.bFourier=stim.bIsFourier

        self._stim=stim
        self.yCtgInd=stim.yCtgInd
        self.yCtg=stim.yCtg
        self.weights=stim.weights

    @property
    def nCtg(self):
        return self.r.shape[-1]

    @property
    def nF(self):
        return self.r.shape[0]

    @property
    def nStim_Ctg(self):
        return self.r.shape[-2]

    @property
    def nStim(self):
        return self.nStim_Ctg*self.nCtg

    @property
    def shape(self):
        return self.r.shape

    @property
    def bSplit(self):
        return self.r.ndim==4

    @property
    def nSplit(self):
        if not self.bSplit:
            return 0
        else:
            return self.r.shape[1]

    def _component(self,r,iF,iSplit,iComp,iCtg):
        c=np.real if iComp==0 else np.imag
        mask=np.asarray(self.weights[:,iCtg])>0
        if self.bSplit:
            return c(np.asarray(r[iF,iSplit,:,iCtg]))[mask]
        return c(np.asarray(r[iF,:,iCtg]))[mask]

    def _parts(self):
        splits=range(self.nSplit) if self.bSplit else (0,)
        components=range(2 if np.iscomplexobj(self.r) else 1)
        return splits,components

    def plot_marginal(self,fld='RNs',name='plot_marginal_responses'):
        r=getattr(self,fld)
        colors=cm.rainbow(np.linspace(0,1,self.nCtg))
        splits,components=self._parts()

        for iF, iS, iC in product(range(self.nF),splits,components):
            plt.figure(name + '_' + str(iF) + '_' + str(iS) + '_' + str(iC))
            for i in range(self.nCtg):
                R1=self._component(r,iF,iS,iC,i)
                plt.hist(R1,color=colors[i],alpha=.4)

    def plot_joint(self,fld='RNs',name='plot_joint_responses'):
        """
        R    [ nF    x nStim ] -> [ nF    x nStim_Ctg x nCtg]         [ nF x nSplit x nStim_Ctg x nCtg ]
        """
        # TODO plot marginals at left and bottom
        r=getattr(self,fld)
        colors=cm.rainbow(np.linspace(0,1,self.nCtg))
        splits,components=self._parts()

        # (f1,f2), (split1,split2), (component1,component2)
        pairs=[p + s + c for p, s, c in product(combinations(range(self.nF),2), product(splits,splits), product(components,components))]

        for j,p in enumerate(pairs):
            plt.figure(name + str(j))
            for i in range(self.nCtg):
                R1=self._component(r,p[0],p[2],p[4],i)
                R2=self._component(r,p[1],p[3],p[5],i)
                plt.scatter(R1,R2,color=colors[i],marker='.',alpha=.4)

    def plot_tsne(self,fld='RNs',name='plot_tsne',n_components=2,**kwargs):
        """
        n_components - ndims to plot
        perplexity   - number of neighbors to consider (5-50)
        """
        r=np.asarray(_flatten_responses(getattr(self,fld)))    # [ nF' x nStim_Ctg x nCtg ]
        mask=np.asarray(self.weights).ravel()>0
        X=r.reshape(r.shape[0],-1).T[mask]
        yCtgInd=np.asarray(self.yCtgInd).ravel()[mask]

        from sklearn.manifold import TSNE
        colors=cm.rainbow(np.linspace(0,1,self.nCtg))
        t=TSNE(n_components=n_components,**kwargs).fit_transform(X).T

        fig=plt.figure(name)
        if n_components==3:
            ax=fig.add_subplot(projection='3d')
        elif n_components==2:
            ax=fig.add_subplot()
        else:
            raise Exception('n_components must be 2 or 3')
        for i in range(self.nCtg):
            inds=yCtgInd==i
            ax.scatter(*t[:,inds],color=colors[i])
