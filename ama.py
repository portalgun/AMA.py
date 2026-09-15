"""
f                         [ nPix x nF ]                       [(nPix / nSplit) x nSplit x nF]
stim [ nPix  x nStim ] -> [ nPix  x nStim_Ctg x nCtg ]        [(nPix / nSplit) x nSplit x nStim_Ctg x nCtg]
R    [ nF    x nStim ] -> [ nF    x nStim_Ctg x nCtg]         [ nF x nSplit x nStim_Ctg x nCtg ]
lAll [ nStim x nCtg ]  -> [ nStim x nCtg x nCtg]
pCor [ nStim x 1 ]
yHat [ nStim x 1 ]
iters
     appendages (new filters)
     recursions                 - learn
         batch
     minimize                   - iter

        #:
"""


import copy
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax.random as jxrandom
import numpy.random as random
from jax import grad,jit,vmap,lax,value_and_grad,tree_util,profiler
from jax.scipy.stats import multivariate_normal as mvn
from jax.scipy.stats import mode
import statsmodels.stats.moment_helpers as mh
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import optax
from functools import partial
import Filter as filt
from jax._src.numpy.util import promote_dtypes_inexact
from dataclasses import dataclass
from scipy.io import loadmat
from itertools import combinations, product
from sklearn.manifold import TSNE 

#log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
#writer = tf.summary.create_file_writer(log_dir)

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


def _id(ins,*_):
    return ins

class _Static:
    # jit treats these config objects as static arguments: equal configurations share compiled traces,
    # and a changed configuration gets a new trace
    def __hash__(self):
        return hash((type(self),self._key()))

    def __eq__(self,other):
        return type(self) is type(other) and self._key()==other._key()

def __is_integer(value):
    return isinstance(value, int) or (isinstance(value, float) and value.is_integer())

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

        setattr(instance,self.wrap_fun_name,self.func)

    @property
    def wrap_fun_name(self):
        return '_' + self._base_func_name + '_fun'


    @property
    def instance_class(self):
        return type(self.instance).__name__

    @property
    def func(self):
        return getattr(self.instance,self.func_name)

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

    def __init__(self,x,stimuli,yCtgInd,Y,bStimIsFourier=False,nSplit=0,bStimIsSplit=False):
        """
        stimuli [ *dims x nStim ]
        yCtgInd [ nStim ] category label of each stimulus (any integer coding, e.g. 1-based from matlab)
        Y       [ nCtg ]  latent variable value of each category, in sorted label order
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
        stimuli=np.concatenate((a,b),len(dims))
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

    The likelihood (Model) always assumes noise of variance fano*|R| + var0 on the final responses, including when no
    noise is sampled (responseType='mean', the mean-response approximation of Burge & Jaini 2017). Noise sampled before
    normalization (bNoise_1) is not accounted for by the likelihood; with both stages on, only stage 2 is.
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
    def __init__(self,fano=1.36,var0=0.23,rmax=5.7,normalizeType='None',activationType='None',bNoise_1=False,bNoise_2=False,rho=0,eps=0.001,averageType='full',nSamples=1,
                 whitenType='None',whitenMethod='zca',whitenEps=1e-5):
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
                self.whitenType,self.whitenMethod,self.whitenEps,getattr(self.filter,'pix_dims',None),
                getattr(self,'bFourier',None),getattr(self,'bSplit',None),self.bAnalytic,str(getattr(self,'dtype',None)))

    def copy(self):
       return Nrn(**_get_copy_dict(self,['filter','corrType','bAnalytic','bFinalized','bSplit','bFourier','dtype']))

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
    def lrn_main(self,rng,stim,fIn,prepped,index,weights=None):
        # prepped and index are arguments rather than read from self.filter, so traces survive re-finalizing
        return self.main(rng,stim,self.insert(fIn,prepped,index),weights)

    @partial(jit, static_argnames=['self'])
    def main(self,rng,stim,f,weights=None,W=None):
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

        # noisey output 2
        RNs = self._average_fun(self._noise_2_fun(RN,self.fano,self.var0,self.nSamples,rng_key2,self.rho))

        return r,rNs,R,RNs,self.variance(R,self.fano,self.var0)

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
    The prior p(X_i)=N_i/N is applied by Objective, so the posterior equals Eq 5 exactly.
    """
    _model_fun=_id
    _response_fun=_id
    modelType=_TypeFunc()
    responseType=_TypeFunc()

    def __init__(self,modelType='gss',responseType='basic'):
        self.modelType=modelType
        self.responseType=responseType

    def _key(self):
        return (self.modelType,self.responseType)

    def copy(self):
       return Model(**_get_copy_dict(self))

    def _finalize(self,*_):
        pass

    #-main
    @partial(jit, static_argnames=['self'])
    def lrn_main(self,R,Rm,RVar,noiseCov,noiseCorr,weights):
        return self._model_fun(R,Rm,RVar,noiseCov,noiseCorr,weights)

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
    def _model__gss(R,Rm,RVar,noiseCov,noiseCorr,weights):
        #: R, Rm [ nF x nStim_Ctg x nCtg ]
        #: lAll  [ nStim_Ctg x nCtg x nCtg ]
        nF=R.shape[0]
        wc=jnp.sum(weights,axis=0)                                           # [ nCtg ]
        mu=jnp.sum(Rm*weights,axis=1)/wc                                     # [ nF x nCtg ]
        D=(Rm-mu[:,None,:])*jnp.sqrt(weights)
        cov=jnp.einsum('isc,jsc->cij',D,D)/(wc-1)[:,None,None]               # [ nCtg x nF x nF ]
        cov=cov + noiseCov

        x=jnp.transpose(R,(1,2,0))[:,:,None,:] - mu.T[None,None,:,:]         # [ nStim_Ctg x nCtg x nCtg x nF ]
        return lmvn0(x,cov[None,None])

    @staticmethod
    def _model__full(R,Rm,RVar,noiseCov,noiseCorr,weights):
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

        def per_true_ctg(Rk):
            # Rk [ nF x nStim_Ctg(l) ] -> [ nStim_Ctg(l) x nCtg(i) ]
            q=-0.5*quad(Rk) + jnp.einsum('fl,fji->lji',Rk,aPm) + c[None]
            return logsumexp(q,axis=1)

        lAll=lax.map(per_true_ctg,jnp.moveaxis(R,-1,0)) - jnp.log(wc)[None,None,:]  # [ nCtg(k) x nStim_Ctg x nCtg ]
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
    def __init__(self,errType='map',bPosterior=None,estType=None,lossType='mean',_bCopy=False):

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
        return (self.errType,self.bPosterior,self.estType,self.lossType)

    def copy(self):
       return Objective(**_get_copy_dict(self),_bCopy=True)

    @partial(jit, static_argnames=['self'])
    def lrn_main(self,lAll,stimweights,yCtg,Y):
        return self._loss_fun(self._err_fun(self._est_fun(self._posterior_fun(lAll,stimweights),Y),yCtg),stimweights)

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
                 batchSize=None,nStepsPerChunk=100,bVerbose=True):
        """
        batchSize      - None for full-batch learning, or the approximate number of stimuli per iteration (AMA-SGD,
                         Burge & Jaini 2017). Each iteration draws a new random batch, stratified so every category keeps
                         its share of the training set (the prior) with at least 2 stimuli. Posteriors are computed
                         against the batch, so full AMA costs O(batchSize^2) per iteration. loss_hist holds batch costs.
        nStepsPerChunk - iterations compiled into one lax.scan; the loss is reported once per chunk
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
        self.nStepsPerChunk=nStepsPerChunk
        self.bVerbose=bVerbose

    def copy(self):
        return Optimizer(**_get_copy_dict(self,['loss_hist','tx']))

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
        # one optax transformation per setting; a new object each call would force recompilation
        key=(self.optimizerType,self.lRate0)
        if getattr(self,'_tx_key',None)!=key:
            self._tx=self.optimizer(self.lRate0)
            self._tx_key=key
        return self._tx

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
        m=np.maximum(2,np.round(self.batchSize*counts/counts.sum())).astype(int)
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

    def minimize(self,f0,rng,stim,filter,loss_fun,opt_state=None):
        tx=self.tx
        proj_fun=self._projection
        proj_params=self._projection_params
        bTangent=self.projectionType[0]=='l2_sphere'
        index=tuple(jnp.asarray(i) for i in filter._insert_index_jx)
        index_exp=tuple(jnp.asarray(i) for i in filter._insert_index_exp_jx)

        # f0
        params=self.insert_project_extract({'f':f0},filter.prepped_exp_jx,index_exp,proj_fun,proj_params)
        if opt_state is None:
            opt_state=tx.init(params)

        if self.batchSize is None:
            mMax,batch_mask=None,None
        else:
            mMax,batch_mask=self._batch_plan(stim.weights)

        # a fixed chunk length keeps one compiled trace; the last chunk masks its extra iterations
        nSteps=max(1,min(self.nStepsPerChunk,self.nIterMax))
        self.loss_hist=[]
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

        return params,opt_state,rng

class Unit(_Static):
    def __init__(self,stim,nrn,model,objective,optimizer=None,seed=None,rng=None,rng_last=None):
        self.nrn=nrn
        self.stim=stim
        self.stim_full=stim
        self.model=model
        self.objective=objective
        self.optimizer=optimizer
        self.opt_state=None

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


    def _check(self):
        counts=np.asarray(self.stim.weights).sum(0)
        if self.model.modelType=='gss' and np.any(counts<2):
            raise Exception("modelType='gss' needs at least 2 stimuli in every category (fewest: " + str(int(counts.min())) + ')')
        if self.model.modelType=='full' and self.nrn.corrType=='None':
            raise Exception("modelType='full' needs response noise; rho=None (no noise) is only supported with modelType='gss'")
        if self.nrn.corrType=='corr' and self.nrn.bFinalized:
            nDim=self.nrn.filter.n*(2 if self.nrn.bAnalytic else 1)*(self.nrn.filter.nSplit if self.nrn.bSplit else 1)
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
                nDim=self.nrn.filter.n*(2 if self.nrn.bAnalytic else 1)*(self.nrn.filter.nSplit if self.nrn.bSplit else 1)
                if nDim >= counts.sum():
                    raise Exception("whitenType='response' needs more stimuli (" + str(int(counts.sum())) + ') than response dimensions (' + str(nDim) + ')')

#- LEARN MODES
    def _run(self,f0,rng,opt_state=None):
        self.out_params,self.opt_state,self.rng_last=self.optimizer.minimize(f0,rng,self.stim,self.filter,self._loss_fun_lrn,opt_state=opt_state)
        self._opt_param_shape=self.out_params['f'].shape
        self.filter.extract(self.out_params['f'])

    def train_new(self,n,fourierType=None,bSplit=None,stimInd=None,dtype=None,optimizer=None):
        if optimizer is not None:
            self.optimizer=optimizer

        self._finalize(n,
                      np.arange(n),
                      (),
                      (),
                      fourierType=fourierType,
                      stimInd=stimInd,
                      dtype=dtype,
                      bSplit=bSplit
        )

        rng,rng_key = jxrandom.split(self.rng)
        f0=self.filter.get_f0(rng_key,self.optimizer._f0_jxrand_fun)
        self._run(f0,rng)

    def train_recurse(self,ind_rec=None,fourierType=None,bSplit=None,stimInd=None,dtype=None,optimizer=None):
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
        self._run(f0,rng,opt_state)

    def train_append(self,n_append,fourierType=None,bSplit=None,stimInd=None,dtype=None,optimizer=None):
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

        rng,rng_key = jxrandom.split(self.rng if self.rng_last is None else self.rng_last)
        f0=self.filter.get_f0(rng_key,self.optimizer._f0_jxrand_fun)
        self._run(f0,rng)

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
        return self._loss_fun({'f':self.filter.out_flat},self.rng,self.stim.val,self.stim.weights,self.stim.yCtg,self.stim.Y,self.nrn._W)

    @property
    def responses(self):
        return Response(*self.nrn.main(self.rng,self.stim.val,self.filter.out_flat,self.stim.weights,self.nrn._W),self.stim)

    @property
    def likelihoods(self):
        return self._likelihoods(self.nrn.main(self.rng,self.stim.val,self.filter.out_flat,self.stim.weights,self.nrn._W),self.stim.weights)

    @property
    def posterior(self):
        return self.objective._posterior_fun(self.likelihoods,self.stim.weights)

    @property
    def error(self):
        return self.objective._err_fun(self.objective._est_fun(self.posterior,self.stim.Y),self.stim.yCtg)

    #- loss functions
    def _likelihoods(self,nrn_out,stimweights):
        R,Rm,RVar=[_flatten_responses(x) for x in self.model._response_fun(*nrn_out)]
        noiseCov=self.nrn._corr_fun(RVar,stimweights,self.nrn.rho)
        noiseCorr=self.nrn.corr_matrix(R.shape[0],R.dtype) if self.nrn.corrType=='corr' else None
        return self.model.lrn_main(R,Rm,RVar,noiseCov,noiseCorr,stimweights)

    @partial(jit, static_argnames=['self'])
    def _loss_fun_lrn(self,params,rng_key,prepped,index,stimval,stimweights,yCtg,Y):
        return self.objective.lrn_main(self._likelihoods(self.nrn.lrn_main(rng_key,stimval,params['f'],prepped,index,stimweights),stimweights),stimweights,yCtg,Y)

    @partial(jit, static_argnames=['self'])
    def _loss_fun(self,params,rng_key,stimval,stimweights,yCtg,Y,W=None):
        return self.objective.lrn_main(self._likelihoods(self.nrn.main(rng_key,stimval,params['f'],stimweights,W),stimweights),stimweights,yCtg,Y)

    #- plot
    def plot_out(self,bFourier=None,name='f_out'):
        self.filter.plot_out(bFourier=bFourier,name=name)

    def plot_last(self,bFourier=None,name='f_last'):
        self.filter.plot_last(bFourier=bFourier,name=name)



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
