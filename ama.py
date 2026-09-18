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
_NRN_EXCL=['filter','corrType','bAnalytic','bFinalized','bSplit','bFourier','dtype','bReadout']
_OPT_EXCL=['loss_hist','tx','val_hist','best_step']


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

def _safe_divide(num,den):
    # zero denominators (e.g. the all-zero padding stimuli) give 0 rather than nan, with finite gradients
    return num/jnp.where(den!=0,den,1)

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
        out.yCtg=self.yCtg[stimInd,:]
        out.yCtgInd=self.yCtgInd[stimInd,:]
        out.nStim_Ctg=len(stimInd)
        out.nStim=out.nStim_Ctg*out.nCtg
        return out

    def __init__(self,x,stimuli,yCtgInd,Y,bStimIsFourier=False,nSplit=0,bStimIsSplit=False,bContrastNormalize=False):
        """
        stimuli [ *dims x nStim ], contrast normalized (zero mean, unit norm; a warning is given otherwise)
        yCtgInd [ nStim ] category label of each stimulus (any integer coding, e.g. 1-based from matlab)
        Y       [ nCtg ]  latent variable value of each category, in sorted label order
        bContrastNormalize - contrast normalize the stimuli (spatial domain only)
        """

        self.bIsFourier=bStimIsFourier
        self.x=x
        self.nSplit=nSplit
        self.bIsSplit=bStimIsSplit

        stimuli=np.asarray(stimuli)
        yCtgInd=np.asarray(yCtgInd).ravel()
        self.ctg,yCtgInd=np.unique(yCtgInd,return_inverse=True) # relabel to 0..nCtg-1
        self.nCtg=len(self.ctg)

        self.Y=jnp.asarray(np.asarray(Y,dtype=float).ravel()) # unique
        if len(self.Y)!=self.nCtg:
            raise Exception('Y has ' + str(len(self.Y)) + ' values but there are ' + str(self.nCtg) + ' categories')
        if stimuli.shape[-1]!=len(yCtgInd):
            raise Exception('last dimension of stimuli must equal the number of labels')

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
        yctg=np.zeros((self.nStim_Ctg,self.nCtg))
        for c in range(self.nCtg):
            val[:,:nStimCtg[c],c]=stimuli[:,yCtgInd==c]
            weights[:nStimCtg[c],c]=1
            yctg[:,c]=self.Y[c]

        self.val=jnp.array(val)
        self.weights=jnp.array(weights)
        self.yCtg=jnp.array(yctg)
        self.yCtgInd=jnp.array(np.broadcast_to(np.arange(self.nCtg),(self.nStim_Ctg,self.nCtg)))
        self.nStim=self.nStim_Ctg*self.nCtg

        if self.bIsSplit:
            self.bIsSplit=False
            self.split()

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
        out.yCtg=jnp.take_along_axis(jnp.asarray(self.yCtg),idx,axis=0)
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
    def load(fname):
        # TODO
        if fname.endswith('.mat'):
            D=loadmat(fname)

            x=filt.X(ndim=1,n=D['s'].shape[0],totS=1)
            return Stim(x,D['s'],D['ctgInd'],D['X'])


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
    rho correlates the noise of all response dimensions (filters, sub-filters, real/imaginary components) equally.
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
                 whitenType='None',whitenMethod='zca',whitenEps=1e-5,readoutType='None'):
        """
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
        readoutType  - pooling of the (normalized) responses across filters, with learned nonnegative weights p
                       (softplus of the parameters Unit.pool_p, normalized to sum 1), before stage-2 noise:
                       'None'
                       'resultant'      - the filter responses and their weighted resultant sum_j p_j R_j
                       'resultant_only' - only the weighted resultant (weighted phase congruency with 'phase')
        """
        # rmax=5.7, var0=0.23 as in burgelab/AMA (paramRSP of AMAdataDisparity.mat)
        self.fano=fano
        self.var0=var0
        self.rmax=rmax
        self.eps=eps

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

        self.rho=rho
        if self.rho is None or ( isinstance(self.rho,str) and self.rho == 'None' ):
            self.corrType='None'
        elif self.rho==0:
            self.corrType='uncorr'
        else:
            self.corrType='corr'

        self.filter=Filter()
        self.bFinalized=False

    @property
    def bAnalytic(self):
        if hasattr(self,'filter') and hasattr(self.filter,'index') and hasattr(self.filter.index,'bAnalytic'):
            return self.filter.index.bAnalytic
        else:
            return None

    def _key(self):
        return (self.fano,self.var0,self.rmax,self.eps,self.nSamples,self.rho,
                self.bNoise_1,self.bNoise_2,self.activationType,self.normalizeType,self.corrType,self.averageType,
                self.whitenType,self.whitenMethod,self.whitenEps,self.readoutType,getattr(self.filter,'pix_dims',None),
                getattr(self,'bFourier',None),getattr(self,'bSplit',None),self.bAnalytic,str(getattr(self,'dtype',None)))

    def copy(self):
       return Nrn(**_get_copy_dict(self,['filter','corrType','bAnalytic','bFinalized','bSplit','bFourier','dtype','bReadout']))

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

    @partial(jit, static_argnames=['self'])
    def main(self,rng,stim,f,weights=None,W=None,p=None):
        """
        returns r, rNs, R, RNs, RVar
            r    mean response before normalization
            rNs  noisy response before normalization
            R    mean response (after normalization)
            RNs  noisy response (after normalization)
            RVar noise variance of the final response given its mean, fano*|R| + var0
        with whitening, responses are flattened real dimensions [ nDim x nStim_Ctg x nCtg ] (see _flatten_responses).
        weights [ nStim_Ctg x nCtg ] mark the valid stimuli (for whitenType='response'); W is a frozen whitening matrix
        """
        rng_key1,rng_key2 = jxrandom.split(rng)

        # respond
        # fourier-domain filters hold half of the frequencies (see _Index.pix). With the orthonormal transform in Stim,
        # the real part of the response is that of the real spatial filter with this spectrum, and the imaginary part is
        # that of its quadrature (hilbert) pair. sqrt(2) gives each of those real filters unit norm when ||f||=1.
        #   fourierType=1: real part only (a real filter)      fourierType=2: both parts (a quadrature pair)
        f=self._scale(f)
        r=self._respond(f,stim)
        # population whitening of the mean responses, before noise
        r=self._whiten_fun(r,f,weights,W)
        r=self._activation_fun(r,self.bFourier)

        # noisey output 1
        rNs = self._average_fun(self._noise_1_fun(r,self.fano,self.var0,self.nSamples,rng_key1,self.rho))

        # normalize
        R   = self._normalize_fun(r,  f,stim,self.eps,self.bSplit)
        RN  = self._normalize_fun(rNs,f,stim,self.eps,self.bSplit)

        # pooled readout across filters
        R   = self._readout_fun(R,p)
        RN  = self._readout_fun(RN,p)

        # noisey output 2
        RNs = self._average_fun(self._noise_2_fun(RN,self.fano,self.var0,self.nSamples,rng_key2,self.rho))

        return r,rNs,R,RNs,self._likelihood_variance(r,R)

    def _likelihood_variance(self,r,R):
        # see the class docstring. r: responses before normalization, R: after
        if not self.bNoise_1:
            return self.variance(R,self.fano,self.var0)
        v1=self.variance(r,self.fano,self.var0)
        if str(self.normalizeType).lower()=='gen':
            # R_i = r_i/D, D = eps + sum_k |r_k|:  var(R_i) ~ sum_j (dR_i/dr_j)^2 v_j  (real responses)
            axes=tuple(range(r.ndim-2))
            D=self.eps + jnp.sum(jnp.abs(r),axis=axes,keepdims=True)
            D=jnp.where(D>0,D,1)
            V=jnp.sum(v1,axis=axes,keepdims=True)
            var=v1/D**2 - 2*jnp.abs(r)*v1/D**3 + r**2*V/D**4
        else:
            # linear normalizations scale each response by |R|/|r| (padding, where r = 0, keeps gain 1)
            r2=jnp.real(r*jnp.conj(r))
            R2=jnp.real(R*jnp.conj(R))
            var=v1*jnp.where(r2>0,R2/jnp.where(r2>0,r2,1),1)
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
    def _activation__relu(R,bComplex):
        return Nrn._componentwise(lambda x: jnp.maximum(x,0),R,bComplex)

    @staticmethod
    def _activation__softplus(R,bComplex):
        return Nrn._componentwise(lambda x: jnp.logaddexp(x,0),R,bComplex)

    @staticmethod
    def _activation__abs(R,bComplex):
        return Nrn._componentwise(jnp.abs,R,bComplex)

    @staticmethod
    def _activation__logistic(R,bComplex):
        return Nrn._componentwise(lambda x: 1/(1+jnp.exp(-x)),R,bComplex)

    @staticmethod
    def _activation__swish(R,bComplex):
        return Nrn._componentwise(lambda x: x/(1+jnp.exp(-x)),R,bComplex)

    @staticmethod
    def _activation__swish2(R,bComplex):
        return Nrn._componentwise(lambda x: x*((1+jnp.tanh(x))/2),R,bComplex)

    @staticmethod
    def _activation__tanh(R,bComplex):
        return Nrn._componentwise(jnp.tanh,R,bComplex)

    @staticmethod
    def _activation__gauss(R,bComplex):
        return Nrn._componentwise(lambda x: jnp.exp(-x**2),R,bComplex)

    @staticmethod
    def _activation__igauss(R,bComplex):
        return Nrn._componentwise(lambda x: 1-jnp.exp(-x**2),R,bComplex)

    #- normalize
    @staticmethod
    def _normalize__none(R,*_):
        return R

    @staticmethod
    def _normalize__gen(R,f,stim,eps,bSplit):
        # divisive normalization by the pooled population response
        axes=tuple(range(R.ndim-2))
        return _safe_divide(R,eps + jnp.sum(jnp.abs(R),axis=axes,keepdims=True))

    @staticmethod
    def _normalize__broad(R,f,stim,eps,bSplit):
        # N_brd = ||s||_2 (stimulus contrast energy), one value per stimulus
        axes=tuple(range(stim.ndim-2))
        N=jnp.sqrt(jnp.sum(jnp.abs(stim)**2,axis=axes))       # [ nStim_Ctg x nCtg ]
        return _safe_divide(R,eps + N)

    @staticmethod
    def _normalize__narrow(R,f,stim,eps,bSplit):
        # N_nrw = A_s^T A_f (amplitude spectra), one value per filter and stimulus; stim and f are fourier-domain
        if bSplit:
            return _safe_divide(R,eps + jnp.einsum('psf,psnc->fsnc',jnp.abs(f),jnp.abs(stim)))
        else:
            return _safe_divide(R,eps + jnp.einsum('pf,pnc->fnc',jnp.abs(f),jnp.abs(stim)))

    @staticmethod
    def _normalize__phase(R,f,stim,eps,bSplit):
        # unit phasors (signs for real responses): contrast- and gain-invariant
        return R/(jnp.abs(R)+eps)

    #- pooled readout over filters (axis 0), weights p [ nF ] (unconstrained; softplus, normalized)
    @staticmethod
    def pool_weights(p):
        w=jnp.logaddexp(p,0.)
        return w/jnp.sum(w)

    @staticmethod
    def _readout__none(R,p):
        return R

    @staticmethod
    def _pool(R,p):
        # weighted sum over filters; real and imaginary parts separately, so the gradient of the real weights is real
        w=Nrn.pool_weights(p).astype(jnp.real(R).dtype)
        if jnp.iscomplexobj(R):
            return (jnp.tensordot(w,R.real,axes=(0,0)) + 1j*jnp.tensordot(w,R.imag,axes=(0,0)))[None]
        return jnp.tensordot(w,R,axes=(0,0))[None]

    @staticmethod
    def _readout__resultant(R,p):
        return jnp.concatenate((R,Nrn._pool(R,p)),axis=0)

    @staticmethod
    def _readout__resultant_only(R,p):
        return Nrn._pool(R,p)

    #- filters and responses
    def _scale(self,f):
        return f*np.sqrt(2) if self.bFourier else f

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
        if rho is not None and rho!=0:
            zf=jnp.reshape(z,(-1,)+z.shape[-3:])                                 # [ nDim' x nStim_Ctg x nCtg x nSamples ]
            L=jnp.linalg.cholesky(rho + (1-rho)*jnp.eye(zf.shape[0],dtype=zf.dtype))
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
        return self.rho + (1-self.rho)*jnp.eye(n,dtype=dtype)

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
        n=RVar.shape[0]
        corrMat=rho + (1-rho)*jnp.eye(n,dtype=RVar.dtype)
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

    Category statistics ('gss', 'student', 'circ'):
    covShrink     - shrink each category covariance toward covTarget by this fraction (0 = sample covariance):
                    Sigma = (1-covShrink) Sigma + covShrink T, before the noise covariance is added
    covTarget     - 'diag' (the covariance's own diagonal) or 'pooled' (the count-weighted mean covariance of all
                    categories)
    ctgPoolWidth  - pool category statistics over neighbouring latent values with a gaussian kernel of this width
                    (in units of Y): each category's covariance is the kernel- and count-weighted pooled scatter,
                    which stabilizes covariances of small categories whose statistics change smoothly with Y
    bPoolMeans    - also pool the category means with the same kernel (biases the means toward neighbours)

    bLeaveOneOut ('full' only) - leave the decoded stimulus out of its own category, so the posterior is Eq 5 with that
                   stimulus removed from the training set. Otherwise each stimulus matches its own mean response, which
                   makes the cost optimistic when noise is low or categories (or batches) are small.
    """
    _model_fun=_id
    _response_fun=_id
    modelType=_TypeFunc()
    responseType=_TypeFunc()

    def __init__(self,modelType='gss',responseType='basic',bLeaveOneOut=False,covShrink=0.,covTarget='diag',df=5.,
                 ctgPoolWidth=None,bPoolMeans=False,circMean='estimate'):
        self.modelType=modelType
        self.responseType=responseType
        self.bLeaveOneOut=bLeaveOneOut
        self.covShrink=float(covShrink)
        self.covTarget=covTarget
        self.df=float(df)
        self.ctgPoolWidth=None if ctgPoolWidth is None else float(ctgPoolWidth)
        self.bPoolMeans=bool(bPoolMeans)
        self.circMean=circMean
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
                self.ctgPoolWidth,self.bPoolMeans,self.circMean)

    def copy(self):
       return Model(**_get_copy_dict(self))

    #-main
    @partial(jit, static_argnames=['self'])
    def lrn_main(self,R,Rm,RVar,noiseCov,noiseCorr,weights,Y):
        return self._model_fun(R,Rm,RVar,noiseCov,noiseCorr,weights,self.bLeaveOneOut,self,Y)

    #- category statistics
    @staticmethod
    def _ctg_kernel(Y,width):
        # [ nCtg x nCtg ] gaussian weights over latent values
        return jnp.exp(-(Y[:,None]-Y[None,:])**2/(2*width**2))

    @staticmethod
    def _ctg_stats(Rm,weights,m,Y,bCentered=True):
        """
        category means [ nF x nCtg ] and covariances [ nCtg x nF x nF ] of the mean responses Rm [ nF x nStim_Ctg x nCtg ]
        (real or complex; covariances are hermitian), with pooling over neighbouring categories and shrinkage (see Model)
        """
        wc=jnp.sum(weights,axis=0)                                               # [ nCtg ]
        mu=jnp.sum(Rm*weights,axis=1)/wc                                         # [ nF x nCtg ]
        if m.ctgPoolWidth is not None and m.bPoolMeans:
            K=Model._ctg_kernel(Y,m.ctgPoolWidth)*wc[None,:]
            mu=jnp.einsum('ik,fk->fi',K,mu)/jnp.sum(K,axis=1)[None,:]
        Dv=(Rm-mu[:,None,:])*jnp.sqrt(weights) if bCentered else Rm*jnp.sqrt(weights)
        S=jnp.einsum('isc,jsc->cij',Dv,jnp.conj(Dv))                             # [ nCtg x nF x nF ] scatter
        dof=(wc-1) if bCentered else wc
        if m.ctgPoolWidth is not None:
            K=Model._ctg_kernel(Y,m.ctgPoolWidth)
            cov=jnp.einsum('ik,kfg->ifg',K,S)/(K@dof)[:,None,None]
        else:
            cov=S/dof[:,None,None]
        if m.covShrink>0:
            if m.covTarget=='pooled':
                T=jnp.broadcast_to(jnp.sum(S,axis=0)/jnp.sum(dof),cov.shape)
            else:
                T=cov*jnp.eye(cov.shape[-1],dtype=cov.dtype)
            cov=(1-m.covShrink)*cov + m.covShrink*T
        return mu,cov

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
    def _model__gss(R,Rm,RVar,noiseCov,noiseCorr,weights,bLeaveOneOut=False,m=None,Y=None):
        #: R, Rm [ nF x nStim_Ctg x nCtg ]
        #: lAll  [ nStim_Ctg x nCtg x nCtg ]
        mu,cov=Model._ctg_stats(Rm,weights,Model() if m is None else m,Y)
        cov=cov + noiseCov

        x=jnp.transpose(R,(1,2,0))[:,:,None,:] - mu.T[None,None,:,:]         # [ nStim_Ctg x nCtg x nCtg x nF ]
        return lmvn0(x,cov[None,None])

    @staticmethod
    def _model__student(R,Rm,RVar,noiseCov,noiseCorr,weights,bLeaveOneOut=False,m=None,Y=None):
        # multivariate t with covariance cov + noiseCov: scale = cov * (df-2)/df
        mu,cov=Model._ctg_stats(Rm,weights,m,Y)
        scale=(cov + noiseCov)*(m.df-2)/m.df
        x=jnp.transpose(R,(1,2,0))[:,:,None,:] - mu.T[None,None,:,:]
        return lmvt0(x,scale[None,None],m.df)

    @staticmethod
    def _model__circ(R,Rm,RVar,noiseCov,noiseCorr,weights,bLeaveOneOut=False,m=None,Y=None):
        # flattened responses hold the real parts of all complex dimensions, then the imaginary parts
        h=R.shape[0]//2
        Rc=R[:h]+1j*R[h:]
        Rmc=Rm[:h]+1j*Rm[h:]
        bZero=m.circMean=='zero'
        mu,cov=Model._ctg_stats(Rmc,weights,m,Y,bCentered=not bZero)
        if bZero:
            mu=jnp.zeros_like(mu)
        # circular noise: the complex variance is the sum of the real and imaginary component (co)variances
        Nc=noiseCov[:,:h,:h] + noiseCov[:,h:,h:]
        cov=cov + Nc.astype(cov.dtype)
        x=jnp.transpose(Rc,(1,2,0))[:,:,None,:] - mu.T[None,None,:,:]
        return lcn0(x,cov[None,None])

    @staticmethod
    def _model__full(R,Rm,RVar,noiseCov,noiseCorr,weights,bLeaveOneOut=False,m=None,Y=None):
        #: lAll[l,k,i] = log mean_j N(R[:,l,k]; Rm[:,j,i], S_ji P S_ji),  S_ji = diag(sqrt(RVar[:,j,i]))
        #: P is the noise correlation matrix (identity when noiseCorr is None). Quadratic forms are expanded in R, so no
        #: [ nF x l x j x i ] tensor is formed.
        nF=R.shape[0]
        wc=jnp.sum(weights,axis=0)
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

        nStim,nCtg=weights.shape
        self_match=jnp.eye(nStim,dtype=bool)[:,:,None]                             # [ l x j x 1 ]

        def per_true_ctg(args):
            # Rk [ nF x nStim_Ctg(l) ] of true category k -> [ nStim_Ctg(l) x nCtg(i) ]
            Rk,k=args
            q=-0.5*quad(Rk) + jnp.einsum('fl,fji->lji',Rk,aPm) + c[None]
            if bLeaveOneOut:
                q=jnp.where(self_match & (jnp.arange(nCtg)==k)[None,None,:],-jnp.inf,q)
            return logsumexp(q,axis=1)

        # dividing by N_i also for the left-out category keeps the posterior (with the prior N_i/N) in Eq 5's sum form
        lAll=lax.map(per_true_ctg,(jnp.moveaxis(R,-1,0),jnp.arange(nCtg))) - jnp.log(wc)[None,None,:]  # [ nCtg(k) x nStim_Ctg x nCtg ]
        return jnp.moveaxis(lAll,0,1)




class Objective(_Static):
    """
    loss(error(estimate(posterior(log-likelihood))))

    errType
        'map' -log p(X_k|R) at the correct level  (0,1 cost / KL, Burge & Jaini 2017 Eq 9)
        'mle' -log p(R|X_k) at the correct level
        'l2'  (Xhat - X_k)^2, estType defaults to posterior 'mean' (MMSE, Eq 13-15)
        'l1'  |Xhat - X_k|,   estType defaults to posterior 'median'
    """
    _posterior_fun=_id
    _est_fun=_id
    _err_fun=_id
    _loss_fun=_id
    bPosterior=_TypeFunc(True)
    estType=_TypeFunc()
    errType=_TypeFunc()
    lossType=_TypeFunc()
    def __init__(self,errType='map',bPosterior=None,estType=None,lossType='mean',regType='None',regWeight=0.,_bCopy=False):
        """
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

        if _bCopy:
            self.bPosterior=bPosterior
            self.estType=estType
            return

        #- posterior
        if   self.errType == 'mle':
            if bPosterior:
                raise Exception('bPosterior must not be set for errType=mle')
            bPosterior=False
        elif self.errType == 'map':
            if bPosterior is False:
                raise Exception('bPosterior must not be False for errType=map')
            bPosterior=True
        elif bPosterior is None:
            bPosterior=True
        self.bPosterior=bPosterior

        #- estType
        if self.errType in ('mle','map'):
            if estType is not None:
                raise Exception('estType must not be set for errType=' + self.errType)
        elif estType is None:
            estType='median' if self.errType=='l1' else 'mean'
        self.estType=estType


    def _key(self):
        return (self.errType,self.bPosterior,self.estType,self.lossType,str(self.regType).lower(),self.regWeight)

    def copy(self):
       return Objective(**_get_copy_dict(self),_bCopy=True)

    @partial(jit, static_argnames=['self'])
    def lrn_main(self,lAll,stimweights,yCtg,Y,priorweights=None):
        # the prior comes from priorweights (the training stimuli) when decoding other stimuli
        prior=stimweights if priorweights is None else priorweights
        return self._loss_fun(self._err_fun(self._est_fun(self._posterior_fun(lAll,prior),Y),yCtg),stimweights)

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

    @staticmethod
    def _est__none(lpost,Y):
        return lpost

    @staticmethod
    def _est__median(lpost,Y):
        # cdf over latent values in ascending order (Y need not be sorted)
        order=jnp.argsort(Y)
        cdf=jnp.cumsum(Objective._prob(lpost)[...,order],axis=-1)
        interp=lambda c: jnp.interp(0.5,c,Y[order])
        return jnp.vectorize(interp,signature='(n)->()')(cdf)

    @staticmethod
    def _est__mean(lpost,Y):
        return Objective._prob(lpost) @ Y

    @staticmethod
    def _est__mode(lpost,Y):
        # MAP estimate; piecewise constant, so provides no gradient
        return Y[jnp.argmax(lpost,axis=-1)]

    @staticmethod
    def _est__cmean(lpost,Y):
        # circular mean, Y in radians
        return jnp.angle(Objective._prob(lpost) @ jnp.exp(1j*Y))

    #- error
    @staticmethod
    def _err__mle(lAll,_):
        return -Objective._at_correct(lAll)

    @staticmethod
    def _err__map(lpost,_):
        return -Objective._at_correct(lpost)

    @staticmethod
    def _err__l1(yHat,yCtg):
        return jnp.abs(yHat-yCtg)

    @staticmethod
    def _err__l2(yHat,yCtg):
        return jnp.abs(yHat-yCtg)**2

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


class Optimizer():
    def __init__(self,optimizerType='adam',projectionType=['l2_sphere',1],lRate0=1e-1,nIterMax=1000,f0_jxrand_fun=['ball',1],
                 batchSize=None,nStepsPerChunk=100,bVerbose=True,nBatchMinCtg=2,patience=None):
        """
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

    def copy(self):
        return Optimizer(**_get_copy_dict(self,['loss_hist','tx','val_hist','best_step']))

    @property
    def _projection(self):
        #l2_sphere, l2_ball, l1_all, l1_sphere
        return getattr(optax.projections,'projection_' + self.projectionType[0])

    @property
    def _projection_params(self):
        return tuple(self.projectionType[1:])

    @property
    def optimizer(self):
        return getattr(optax,self.optimizerType)

    @property
    def tx(self):
        # one optax transformation per setting, shared by all Optimizers: a new transformation object would compile the
        # training step again (e.g. for every new Unit or cross-validation fold)
        key=(self.optimizerType,self.lRate0)
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
    def _sample_batch(rng,mMax,mask,stimval,stimweights,yCtg):
        # a random subset of the valid (non-padding) stimuli within each category
        score=jxrandom.uniform(rng,stimweights.shape) + jnp.where(stimweights>0,0.,jnp.inf)
        idx=jnp.argsort(score,axis=0)[:mMax]                                    # [ mMax x nCtg ]
        val=jnp.take_along_axis(stimval,jnp.broadcast_to(idx,stimval.shape[:-2]+idx.shape),axis=-2)
        w=jnp.take_along_axis(stimweights,idx,axis=0)*mask
        y=jnp.take_along_axis(yCtg,idx,axis=0)
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
    @partial(jit, static_argnames=['tx','loss_fun','proj_fun','proj_params','nSteps','mMax','bTangent'])
    def _run_chunk(tx,loss_fun,proj_fun,proj_params,nSteps,mMax,bTangent,nActive,
                   params,opt_state,rng,prepped,prepped_exp,index,index_exp,batch_mask,stimval,stimweights,yCtg,Y):
        def body(carry,i):
            params,opt_state,rng=carry
            rng,rng_batch,rng_key=jxrandom.split(rng,3)
            if mMax is None:
                val,w,y=stimval,stimweights,yCtg
            else:
                val,w,y=Optimizer._sample_batch(rng_batch,mMax,batch_mask,stimval,stimweights,yCtg)

            loss_value,grads=value_and_grad(loss_fun)(params,rng_key,prepped,index,val,w,y,Y)
            # complex params: jax returns the conjugate of the ascent direction
            grads=tree_util.tree_map(jnp.conjugate,grads)
            if bTangent:
                grads=dict(grads,f=Optimizer._tangent(grads['f'],params['f']))
            updates,new_state=tx.update(grads,opt_state,params)
            new_params=Optimizer.insert_project_extract(optax.apply_updates(params,updates),prepped_exp,index_exp,proj_fun,proj_params)

            # iterations past nIterMax in the last chunk leave the state unchanged
            keep=lambda new,old: tree_util.tree_map(lambda a,b: jnp.where(i<nActive,a,b),new,old)
            return (keep(new_params,params),keep(new_state,opt_state),rng),loss_value

        (params,opt_state,rng),losses=lax.scan(body,(params,opt_state,rng),jnp.arange(nSteps))
        return params,opt_state,rng,losses

    def minimize(self,f0,rng,stim,filter,loss_fun,opt_state=None,extra_params=None,val_fun=None):
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

        # a fixed chunk length keeps one compiled trace; the last chunk masks its extra iterations
        nSteps=max(1,min(self.nStepsPerChunk,self.nIterMax))
        self.loss_hist=[]
        self.val_hist=[]
        self.best_step=None
        best=None
        nBad=0
        step=0
        while step < self.nIterMax:
            nActive=min(nSteps,self.nIterMax-step)
            params,opt_state,rng,losses=self._run_chunk(tx,loss_fun,proj_fun,proj_params,nSteps,mMax,bTangent,nActive,
                                                        params,opt_state,rng,
                                                        filter.prepped_jx,filter.prepped_exp_jx,index,index_exp,batch_mask,
                                                        stim.val,stim.weights,stim.yCtg,stim.Y)
            self.loss_hist.extend(np.asarray(losses)[:nActive].tolist())
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
                    break

        if best is not None:
            _,params,opt_state,self.best_step=best
        return params,opt_state,rng

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
        self.train_log=[]         # training calls, for config() / from_config()

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

    def split(self,stimInd=None):

        nrn=self.nrn.copy()
        model=self.model.copy()
        objective=self.objective.copy()
        optimizer=None if self.optimizer is None else self.optimizer.copy()

        stim=self.stim if stimInd is None else self.stim._subset(stimInd)

        self.rng,rng_key = jxrandom.split(self.rng)
        unit=Unit(stim,nrn,model,objective,optimizer=optimizer,seed=self.seed,rng=rng_key,rng_last=self.rng_last)

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

        stim=self.stim_full._finalize(dtype,None,bFourier,bSplit)
        self.stim=stim if stimInd is None else stim._subset(stimInd)

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
        nOut={'resultant':self.nrn.filter.n+1,'resultant_only':1}.get(str(self.nrn.readoutType).lower(),self.nrn.filter.n)
        return nOut*(2 if self.nrn.bAnalytic else 1)*(self.nrn.filter.nSplit if self.nrn.bSplit else 1)

    def _check_batches(self):
        if self.optimizer.batchSize is None or self.model.modelType not in ('gss','student','circ'):
            return
        _,mask=self.optimizer._batch_plan(self.stim.weights)
        mMin=int(np.asarray(mask).sum(0).min())
        if mMin < self._nDim+1:
            warnings.warn('batches have as few as ' + str(mMin) + ' stimuli in a category, fewer than the ' + str(self._nDim+1)
                          + ' needed for a full-rank AMA-Gauss covariance of ' + str(self._nDim) + ' response dimensions; '
                          'increase batchSize or Optimizer nBatchMinCtg',stacklevel=3)

    def _check(self):
        counts=np.asarray(self.stim.weights).sum(0)
        if self.model.modelType in ('gss','student','circ') and np.any(counts<2) and self.model.ctgPoolWidth is None:
            raise Exception("modelType='" + self.model.modelType + "' needs at least 2 stimuli in every category (fewest: " + str(int(counts.min())) + ')')
        if self.model.modelType=='student' and not self.model.df>2:
            raise Exception("modelType='student' needs df > 2 (its covariance is matched to the category covariance)")
        if self.model.modelType=='circ' and self.nrn.bFinalized:
            if not self.nrn.bAnalytic:
                raise Exception("modelType='circ' needs complex quadrature-pair responses (fourierType=2)")
            if self.nrn._bWhiten:
                raise Exception("modelType='circ' can not be combined with whitening, which mixes real and imaginary components")
        if self.model.bLeaveOneOut is False and self.model.modelType not in ('gss','full','student','circ'):
            raise Exception("modelType must be 'gss', 'full', 'student', or 'circ'")
        if self.model.modelType=='full' and self.nrn.corrType=='None':
            raise Exception("modelType='full' needs response noise; rho=None (no noise) is only supported with modelType='gss'")
        if self.nrn.corrType=='corr' and self.nrn.bFinalized:
            nDim=self._nDim
            lo=-1/(nDim-1) if nDim>1 else -np.inf
            if not (lo < self.nrn.rho < 1):
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
        if str(self.nrn.readoutType).lower() not in ('none','resultant','resultant_only'):
            raise Exception("readoutType must be 'None', 'resultant', or 'resultant_only'")
        if self.nrn.bReadout and self.nrn.bNoise_1:
            raise Exception('stage-1 noise (bNoise_1) is not modeled through a pooled readout')
        if self.nrn.bReadout and self.nrn._bWhiten:
            raise Exception('a pooled readout can not be combined with whitening')
        if self.model.bLeaveOneOut:
            if self.model.modelType!='full':
                raise Exception("bLeaveOneOut is only implemented for modelType='full'")
            if self.objective.errType=='mle':
                raise Exception("bLeaveOneOut defines a leave-one-out posterior and can not be used with errType='mle'")
            if np.any(counts<2):
                raise Exception('bLeaveOneOut needs at least 2 stimuli in every category')
        if (self.nrn.bNoise_1 and str(self.nrn.normalizeType).lower()=='gen' and self.nrn.bFinalized
                and self.nrn.bAnalytic and not self.nrn._bWhiten):
            raise Exception("stage-1 noise (bNoise_1) with normalizeType='gen' is only modeled for real responses (not fourierType=2)")

#- LEARN MODES
    def _p0(self):
        # initial pooling-weight parameters: previous values, extended with zeros (equal weights) for new filters
        n=self.nrn.filter.n
        p=np.zeros(n)
        if self.pool_p is not None:
            old=np.asarray(self.pool_p).ravel()[:n]
            p[:len(old)]=old
        return jnp.asarray(p,dtype=jnp.finfo(self.nrn.dtype).dtype)

    def _val_fun(self,stimVal):
        test=self._prepare_stim(stimVal)
        def fun(params):
            full={'f':self.filter.insert(params['f'])}
            if 'p' in params:
                full['p']=params['p']
            return self._loss_fun_heldout(full,self.rng,test.val,test.weights,test.yCtg,test.Y,self.stim.val,self.stim.weights)
        return fun

    def _run(self,f0,rng,opt_state=None,stimVal=None):
        self._check_batches()
        extra={'p':self._p0()} if self.nrn.bReadout else None
        val_fun=None if stimVal is None else self._val_fun(stimVal)
        self.out_params,self.opt_state,self.rng_last=self.optimizer.minimize(f0,rng,self.stim,self.filter,_BoundLoss(self,'_loss_fun_lrn'),
                                                                             opt_state=opt_state,extra_params=extra,val_fun=val_fun)
        self._opt_param_shape=self.out_params['f'].shape
        self.filter.extract(self.out_params['f'])
        if 'p' in self.out_params:
            self.pool_p=np.asarray(self.out_params['p'])

    def _train_random(self,rng,nRestarts,stimVal=None):
        # learn from random initial filters; with restarts, keep the run with the lowest cost on the training stimuli
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
            cost=float(self.loss)
            self.restart_costs.append(cost)
            if best is None or cost<best[0]:
                best=(cost,jnp.asarray(self.filter.out),self.out_params,self.opt_state,self.rng_last,
                      list(self.optimizer.loss_hist),self._opt_param_shape,self.pool_p)
        _,self.filter.out,self.out_params,self.opt_state,self.rng_last,self.optimizer.loss_hist,self._opt_param_shape,self.pool_p=best

    @_logged_training
    def train_new(self,n,fourierType=None,bSplit=None,stimInd=None,dtype=None,optimizer=None,nRestarts=1,stimVal=None):
        """learn n new filters. stimVal: validation stimuli for early stopping (Optimizer patience)"""
        if optimizer is not None:
            self.optimizer=optimizer
        self.pool_p=None

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
        opt_state=self.opt_state if getattr(self,'_opt_param_shape',None)==f0.shape else None

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

    def _train_generated(self,params,build,penalty=None,stimVal=None):
        """
        optimize params (a dict of arrays; 'p' holds readout pooling weights) of filters f=build(params) [ nPix x nF ] on
        this unit's cost, plus penalty(params) if given, with this unit's Optimizer (optimizerType, lRate0, nIterMax,
        nStepsPerChunk, bVerbose, patience with stimVal). Returns the final (or best validation) params.
        """
        opt=self.optimizer
        tx=opt.tx
        stim=self.stim

        def full(prm):
            out={'f':build(prm)}
            if 'p' in prm:
                out['p']=prm['p']
            return out

        def loss(prm):
            cost=self._loss_fun(full(prm),self.rng,stim.val,stim.weights,stim.yCtg,stim.Y,None)
            return cost if penalty is None else cost+penalty(prm)

        step_fun=jit(value_and_grad(loss))
        val_fun=None
        if stimVal is not None:
            test=self._prepare_stim(stimVal)
            val_fun=jit(lambda prm: self._loss_fun_heldout(full(prm),self.rng,test.val,test.weights,test.yCtg,test.Y,
                                                          stim.val,stim.weights))

        opt_state=tx.init(params)
        opt.loss_hist,opt.val_hist,opt.best_step=[],[],None
        best,nBad=None,0
        for it in range(opt.nIterMax):
            value,grads=step_fun(params)
            # complex params: jax returns the conjugate of the ascent direction
            grads=tree_util.tree_map(jnp.conjugate,grads)
            updates,opt_state=tx.update(grads,opt_state,params)
            params=optax.apply_updates(params,updates)
            opt.loss_hist.append(float(value))
            if (it+1) % max(1,opt.nStepsPerChunk)==0 or it==opt.nIterMax-1:
                if opt.bVerbose:
                    print(f'step {it}, loss: {float(value)}')
                if val_fun is not None:
                    v=float(val_fun(params))
                    opt.val_hist.append(v)
                    if best is None or v<best[0]:
                        best,nBad=(v,params,it),0
                    else:
                        nBad+=1
                    if opt.patience is not None and nBad>=opt.patience:
                        break
        if best is not None:
            _,params,opt.best_step=best
        return params

    def _set_generated(self,f,params):
        # generated filters [ nPix x nF ] become this unit's ordinary filters
        self.filter.out=jnp.reshape(f,self.filter._shape_exp)
        self.out_params={'f':f[self.filter._insert_index_jx]}
        self._opt_param_shape=self.out_params['f'].shape
        self.opt_state=None
        if 'p' in params:
            self.pool_p=np.asarray(params['p'])

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
                         nScales=None,nOrientations=None,bInterleave=False,nMothers=1,motherInitNoise=0.1,edgeTaper=0.):
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
        self.pool_p=None
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
                 knotTaper=self._knot_taper(int(nKnot),float(edgeTaper)))

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
        if self.nrn.bReadout:
            params['p']=self._p0()

        penalty=None
        if knotSmooth>0:
            def penalty(prm):
                tot=0.
                # per mother, relative to that mother's energy (scale-free), on the tapered knots
                tm=self._tapered_mothers(prm,cfg)
                energy=jnp.sum((jnp.abs(tm)**2).reshape(nMothers,-1),axis=1)+1e-12
                for v in (jnp.real(tm),jnp.imag(tm)):
                    d=jnp.sum((jnp.diff(v,n=2,axis=1)**2).reshape(nMothers,-1),axis=1)
                    if b2D:
                        d=d+jnp.sum((jnp.diff(v,n=2,axis=2)**2).reshape(nMothers,-1),axis=1)
                    tot=tot+jnp.sum(d/energy)
                return knotSmooth*tot

        build=lambda prm: self._multiscale_filters(prm,nTot,cfg)
        params=self._train_generated(params,build,penalty=penalty,stimVal=stimVal)
        self._set_generated(build(params),params)

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
    def train_parametric(self,n,family='morse',fourierType=2,bTied=True,orientations=None,init=None,dtype=None,stimVal=None):
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
        """
        if fourierType not in (1,2):
            raise Exception('train_parametric learns in the fourier domain: fourierType must be 1 or 2')
        self.pool_p=None
        self._finalize(n,np.arange(n),fourierType=fourierType,dtype=dtype,bSplit=False)
        if len(self.filter.pix_dims) not in (1,2):
            raise Exception('train_parametric supports 1D and 2D stimuli')
        b2D=len(self.filter.pix_dims)==2
        orientations=np.zeros(n) if orientations is None else np.asarray(orientations,dtype=float)
        if b2D and orientations.shape!=(n,):
            raise Exception('orientations must have one value per filter')
        params=self._parametric_init(family,n,bTied,init,b2D)
        if self.nrn.bReadout:
            params['p']=self._p0()
        params=self._train_generated(params,lambda prm: self._parametric_filters(prm,family,n,bTied,orientations),
                                     stimVal=stimVal)
        f=self._parametric_filters(params,family,n,bTied,orientations)
        self._set_generated(f,params)
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
        self.param_out=out

    #- evaluation
    def _prepare_stim(self,stim):
        # other stimuli in this unit's learning domain and precision
        if not self.nrn.bFinalized:
            raise Exception('train (or finalize) the unit first')
        if stim.nCtg!=self.stim.nCtg or not np.allclose(np.asarray(stim.Y),np.asarray(self.stim.Y)):
            raise Exception('stimuli must have the same categories (Y) as the training stimuli')
        return copy.copy(stim)._finalize(self.nrn.dtype,None,self.nrn.bFourier,self.nrn.bSplit)

    @partial(jit, static_argnames=['self'])
    def _loss_fun_heldout(self,params,rng_key,stimval,stimweights,yCtg,Y,refval,refweights):
        f=params['f']
        p=params.get('p')
        W=self.nrn.whitening(refval,f,refweights) if self.nrn._bWhiten else None
        obs=self.nrn.main(rng_key,stimval,f,stimweights,W,p)
        ref=self.nrn.main(rng_key,refval,f,refweights,W,p)
        return self.objective.lrn_main(self._likelihoods_heldout(obs,ref,refweights,Y),stimweights,yCtg,Y,refweights)

    def evaluate(self,stim):
        """
        cost of decoding other stimuli (e.g. a held-out test set) with the current filters. This unit's stimuli are the
        training set: the category response distributions (AMA-Gauss), the reference stimuli (full AMA), the prior,
        and the whitening all come from them.
        """
        test=self._prepare_stim(stim)
        return float(self._loss_fun_heldout(self._params_out(),self.rng,test.val,test.weights,test.yCtg,test.Y,
                                            self.stim.val,self.stim.weights))

    def _log_posterior(self,stim=None):
        f=self.filter.out_flat
        if stim is None:
            return Objective._posterior__true(self.likelihoods,self.stim.weights),self.stim
        test=self._prepare_stim(stim)
        W=self.nrn.whitening(self.stim.val,f,self.stim.weights) if self.nrn._bWhiten else None
        obs=self.nrn.main(self.rng,test.val,f,test.weights,W,self._p())
        ref=self.nrn.main(self.rng,self.stim.val,f,self.stim.weights,W,self._p())
        return Objective._posterior__true(self._likelihoods_heldout(obs,ref,self.stim.weights,self.stim.Y),self.stim.weights),test

    def estimates(self,estType='mode',stim=None):
        """
        estimates of the latent variable [ nStim_Ctg x nCtg ] (grouped like Stim.val; see Stim.weights) for the training
        stimuli, or for other stimuli decoded with the training set: 'mode' (MAP), 'mean' (MMSE), 'median', or
        'cmean' (circular mean, Y in radians)
        """
        lpost,st=self._log_posterior(stim)
        return np.asarray(getattr(Objective,'_est__'+estType)(lpost,st.Y))

    def performance(self,estType='mode',stim=None):
        """
        estimation performance per latent level for the training stimuli, or for other stimuli decoded with the training
        set: bias, sd, and rmse of the estimates; pCorrect and confusion [ true x MAP category ] of the MAP category;
        and cost, the mean -log posterior at the correct level
        """
        lpost,st=self._log_posterior(stim)
        lpost=np.asarray(lpost)
        est=np.asarray(getattr(Objective,'_est__'+estType)(jnp.asarray(lpost),st.Y))
        w=np.asarray(st.weights)>0
        Y=np.asarray(st.Y)
        nCtg=len(Y)
        best=np.argmax(lpost,axis=-1)
        confusion=np.stack([np.bincount(best[w[:,c],c],minlength=nCtg) for c in range(nCtg)])
        err=[est[w[:,c],c]-Y[c] for c in range(nCtg)]
        correct=np.diagonal(lpost,axis1=-2,axis2=-1)
        return {'Y':Y,
                'estimates':est,
                'bias':np.array([e.mean() for e in err]),
                'sd':np.array([e.std() for e in err]),
                'rmse':np.array([np.sqrt(np.mean(e**2)) for e in err]),
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
               'nrn':_get_copy_dict(self.nrn,['filter','corrType','bAnalytic','bFinalized','bSplit','bFourier','dtype','bReadout']),
               'model':_get_copy_dict(self.model),
               'objective':_get_copy_dict(self.objective),
               'optimizer':None if self.optimizer is None else _get_copy_dict(self.optimizer,['loss_hist','tx','val_hist','best_step']),
               'finalize':dict(n=self.filter.n,dtype=np.dtype(self.nrn.dtype).name,bFourier=self.nrn.bFourier,
                               bAnalytic=bool(self.nrn.bAnalytic),bSplit=self.nrn.bSplit),
               'out':np.asarray(self.filter.out),
               'last':asnp(self.filter.last),
               'opt_state':asnp(self.opt_state),
               'opt_param_shape':getattr(self,'_opt_param_shape',None),
               'loss_hist':list(getattr(self.optimizer,'loss_hist',[])),
               'restart_costs':self.restart_costs,
               'seed':self.seed,
               'rng':np.asarray(jxrandom.key_data(self.rng)),
               'rng_last':None if self.rng_last is None else np.asarray(jxrandom.key_data(self.rng_last)),
               'W':asnp(self.nrn._W),
               'pool_p':asnp(self.pool_p),
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
        unit._finalize(fin['n'],np.arange(fin['n']),fourierType=fourierType,bSplit=fin['bSplit'],dtype=jnp.dtype(fin['dtype']))
        if state['optimizer'] is None:
            unit.optimizer=None
        unit.filter.out=jnp.asarray(state['out'])
        unit.filter.last=asjnp(state['last'])
        unit.opt_state=asjnp(state['opt_state'])
        unit._opt_param_shape=state['opt_param_shape']
        if unit.optimizer is not None:
            unit.optimizer.loss_hist=state['loss_hist']
        unit.restart_costs=state['restart_costs']
        unit.nrn._W=asjnp(state['W'])
        unit.pool_p=None if state.get('pool_p') is None else np.asarray(state['pool_p'])
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
                'Y':_yaml_safe(np.asarray(st.Y)),'bIsFourier':bool(st.bIsFourier),'nSplit':int(st.nSplit or 0)}

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
        return None if (self.pool_p is None or not self.nrn.bReadout) else jnp.asarray(self.pool_p)

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

    @property
    def likelihoods(self):
        return self._likelihoods(self.nrn.main(self.rng,self.stim.val,self.filter.out_flat,self.stim.weights,self.nrn._W,self._p()),self.stim.weights,self.stim.Y)

    @property
    def posterior(self):
        return self.objective._posterior_fun(self.likelihoods,self.stim.weights)

    @property
    def error(self):
        return self.objective._err_fun(self.objective._est_fun(self.posterior,self.stim.Y),self.stim.yCtg)

    #- loss functions
    def _likelihoods(self,nrn_out,stimweights,Y):
        R,Rm,RVar=[_flatten_responses(x) for x in self.model._response_fun(*nrn_out)]
        noiseCov=self.nrn._corr_fun(RVar,stimweights,self.nrn.rho)
        noiseCorr=self.nrn.corr_matrix(R.shape[0],R.dtype) if self.nrn.corrType=='corr' else None
        return self.model.lrn_main(R,Rm,RVar,noiseCov,noiseCorr,stimweights,Y)

    def _likelihoods_heldout(self,obs_out,ref_out,refweights,Y):
        # observed responses of other stimuli, decoded against the reference (training) stimuli; no leave-one-out
        R=_flatten_responses(self.model._response_fun(*obs_out)[0])
        _,Rm,RVar=[_flatten_responses(x) for x in self.model._response_fun(*ref_out)]
        noiseCov=self.nrn._corr_fun(RVar,refweights,self.nrn.rho)
        noiseCorr=self.nrn.corr_matrix(R.shape[0],R.dtype) if self.nrn.corrType=='corr' else None
        return self.model._model_fun(R,Rm,RVar,noiseCov,noiseCorr,refweights,False,self.model,Y)

    @partial(jit, static_argnames=['self'])
    def _loss_fun_lrn(self,params,rng_key,prepped,index,stimval,stimweights,yCtg,Y):
        cost=self.objective.lrn_main(self._likelihoods(self.nrn.lrn_main(rng_key,stimval,params['f'],prepped,index,stimweights,params.get('p')),stimweights,Y),stimweights,yCtg,Y)
        if self.objective.regWeight>0 and str(self.objective.regType).lower()!='none':
            cost=cost+self.objective.regWeight*self._penalty(self.nrn.insert(params['f'],prepped,index))
        return cost

    @partial(jit, static_argnames=['self'])
    def _loss_fun(self,params,rng_key,stimval,stimweights,yCtg,Y,W=None):
        return self.objective.lrn_main(self._likelihoods(self.nrn.main(rng_key,stimval,params['f'],stimweights,W,params.get('p')),stimweights,Y),stimweights,yCtg,Y)

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
        lat=np.asarray(st.Y)[sel[:,1]]
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
            Yv=np.asarray(st.Y)
            cmap='twilight' if Yv.min()>=0 and Yv.max()<2*np.pi+1e-9 and np.ptp(Yv)>np.pi else 'viridis'
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
