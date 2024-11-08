#: f                         [ nPix x nF ]                       (nPix / nSplit) x nSplit x nF
#: stim [ nPix  x nStim ] -> [ nPix  x nStim_Ctg x nCtg ]        (nPix / nSplit) x nSplit x nStim_Ctg x nCtg
#: R    [ nF    x nStim ] -> [ nF    x nStim_Ctg x nCtg]
#: lAll [ nStim x nCtg ]  -> [ nStim x nCtg x nCtg]
#: pCor [ nStim x 1 ]
#: yHat [ nStim x 1 ]
#: iters
#:      appendages (new filters)
#:      recursions                 - learn
#:          batch
#:      minimize                   - iter


import copy
import numpy as np
import jax.numpy as jnp
import jax.random as jxrandom
import numpy.random as random
from jax import grad,jit,vmap,lax,value_and_grad,tree_util,profiler
from jax.scipy.stats import multivariate_normal as mvn
import statsmodels.stats.moment_helpers as mh
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import optax
from functools import partial
import Filter as filt
from jax._src.numpy.util import promote_dtypes_inexact
from dataclasses import dataclass
from scipy.io import loadmat
from itertools import combinations

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
    return (-1/2 * jnp.einsum('...i,...i->...', y, y) - cov.shape[-1]/2 * jnp.log(2*np.pi)
            - jnp.log(L.diagonal(axis1=-1, axis2=-2)).sum(-1))

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
        return '_' + self._base_func_name + '__' + self._inst_func_name


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
        return jnp.reshape(self.val[:,index[0],index[1]],self.dims)

    def __getitem__(self,index):
        yCtgInd=self.yCtgInd[index].flatten(order='F')
        return Stim(self.x,self.val[...,index[0],index[1]],yCtgInd, self.Y, self.bIsFourier, _bShape=True, _weights=self.weights[index],_yCtg=self.yCtg[index])

    def __init__(self,x,stimuli,yCtgInd,Y,bStimIsFourier=False,_bShape=False,_weights=np.empty,_yCtg=np.empty,nSplit=0,bStimIsSplit=False):

        self.bIsFourier=bStimIsFourier

        self.x=x

        self.ctg=jnp.unique(yCtgInd)
        self.nCtg=len(self.ctg)
        self.nSplit=nSplit
        self.bIsSplit=bStimIsSplit

        self.Y=Y # unique

        if _bShape:
            self.dims=stimuli.shape[:-2]
        else:
            self.dims=stimuli.shape[:-1]

        self.ndim=len(self.dims)
        self.nPix=jnp.multiply.reduce(jnp.array(self.dims))


        if not _bShape:
            nStimCtg=np.zeros((self.nCtg),dtype=int)
            for c in range(self.nCtg):
                nStimCtg[c]=len(np.where(yCtgInd==c)[0])
            self.nStim_Ctg=np.max(nStimCtg)

            stimuli=np.reshape(stimuli,(self.nPix,jnp.shape(yCtgInd)[0]))
            weights=np.zeros((self.nStim_Ctg,self.nCtg))
            yctg=np.zeros((self.nStim_Ctg,self.nCtg))
            val=np.zeros((self.nPix,self.nStim_Ctg,self.nCtg))
            for c in range(self.nCtg):
                val[:,:nStimCtg[c],c]=stimuli[:,np.where(yCtgInd==c)[0]]
                weights[:nStimCtg[c],c]=1
                yctg[:nStimCtg[c],c]=Y[c]
            self.val=jnp.array(val)
            self.weights=jnp.array(weights)
            self.yCtg=yctg
        else:
            self.nStim_Ctg=jnp.shape(stimuli)[-2]
            self.val=stimuli
            self.weights=_weights
            self.yCtg=_yCtg

        yCI=jnp.zeros((self.nStim_Ctg,self.nCtg),dtype=int)
        for c in range(self.nCtg):
            yCI=yCI.at[:,c].set(c)
        self.yCtgInd=jnp.array(yCI)
        self.nStim=self.nStim_Ctg*self.nCtg

    def _fft(self):
        if self.bIsFourier:
            raise Exception('Stim is already in the fourier domain')
        self.val=filt.fft(self.x,self.val)
        self.bIsFourier=True

    def _ifft(self):
        if not self.bIsFourier:
            raise Exception('Stim is already out of the fourier domain')
        self.val=filt.ifft(self.x,self.val)
        self.bIsFourier=False

    def finalize(self,dtype,index,bFourier,bSplit):
        if bSplit and not self.bIsSplit:
            self.split()
        elif not bSplit and  self.bIsSplit:
            self.unsplit()

        if bFourier and not self.bIsFourier:
            self._fft()
        elif not bFourier and self.bIsFourier:
            self._ifft()

        if not ( isinstance(self.val,jnp.ndarray) and jnp.issubdtype(self.val,dtype)):
            if np.any(np.imag(self.val) != 0) and dtype!=jnp.complex64:
                self.val=jnp.array(np.real(self.val).astype(dtype)) + 1j * jnp.array(np.imag(self.val).astype(dtype))
            else:
                self.val=jnp.array(self.val,dtype=dtype)

        if index:
            return self.index(index)
        else:
            return self
    def split(self):
        #[ nPix  x nStim_Ctg x nCtg ]    (nPix / nSplit) x nSplit x nStim_Ctg x nCtg
        # check shape
        self.val=jnp.reshape(self.val,(self.nPix/self.nSplit, self.nSplit, self.nStim_Ctg, self.nCtg))
        self.bIsSplit=True

        return

    def unsplit(self):
        self.val=jnp.reshape(self.val,(self.nPix, self.nStim_Ctg, self.nCtg))
        self.bIsSplit=False
        return

    def plot_all(self):
        plt.imshow(jnp.reshape(self.val,(np.multiply.reduce(self.dims), self.nStim )))


    def plot(self,index=(0,0),bFourier=False):
        if bFourier and not self.bIsFourier:
            stim=filt.fft(self.x,self.index(index))
        elif not bFourier and self.bIsFourier:
            stim=filt.fft(self.x,self.index(index))
        else:
            stim=self.index(index)

        if self.ndim==1:
            filt.plotFT(self.x.sl[0],stim)
        elif self.ndim==2:
            plt.imshow(self.index(index),extent=self.x.extents)
        elif self.ndim==3:
            raise Exception('TODO')
        elif self.ndim==4:
            raise Exception('TODO')

    @staticmethod
    def load(fname):
        # TODO
        if fname.endswith('.mat'):
            D=loadmat(fname)

            x=filt.X(n=D['s'].shape[0],totS=1)
            return Stim(x,D['s'],D['ctgInd'],D['X'])


    @classmethod
    def gen_test(cls,dims=(8,9),nStim=101):
        a=random.randn(*(dims+(int(jnp.ceil(nStim/2)),)))
        b=random.rand( *(dims+(int(jnp.floor(nStim/2)),)))
        stimuli=jnp.concatenate((a,b),2)
        yCtgInd=jnp.concatenate((jnp.zeros(a.shape[-1],dtype=int),jnp.ones(b.shape[-1],dtype=int)),0)
        X=jnp.array([1, 2])

        x=filt.X(n=(dims),totS=1)
        return Stim(x,stimuli,yCtgInd,X,False)

class _Index():
    _parent=None
    n=_ParentProp()
    stim=_ParentProp()
    ndim=_ParentProp()
    dims=_ParentProp()
    x=_ParentProp()
    nPix=_ParentProp()
    bNew=_ParentProp()

    def __init__(self,parent,ind_lrn,ind_fix,ind_rec,bAnalytic=True):
        self._parent=parent

        #- parse
        ind_lrn=_Index.__parse_ind(ind_lrn)
        ind_fix=_Index.__parse_ind(ind_fix)
        ind_rec=_Index.__parse_ind(ind_rec)

        if len(ind_rec)==0 and len(ind_lrn)==0:
            raise Exception('a lrn or rec index is required')

        #- No overlap
        lst=[]
        if len(ind_lrn)!=0:
            lst.append(ind_lrn)
        if len(ind_fix)!=0:
            lst.append(ind_fix)
        if len(ind_rec)!=0:
            lst.append(ind_rec)

        if len(lst)>1:
            comb=np.concat(lst)
            if len(np.unique(comb))!=len(comb):
                raise Exception('filter indices need to be mutually exclusive')
        else:
            comb=lst

        if np.any(np.isin(np.arange(self.n),comb,invert=True)):
            raise Exception('indeces contain value outside of range (n=' + str(self.n) + ')')

        self.ind_rec=ind_rec # old and learning
        self.ind_lrn=ind_lrn # new and learning
        self.ind_fix=ind_fix # old and helping

        self.bAnalytic=bAnalytic

    @staticmethod
    def __parse_ind(ind):
        if not isinstance(ind,np.ndarray):
            ind=np.array(ind,dtype=int,ndmin=1)


        if not np.issubdtype(ind.dtype,np.integer):
            ind.astype(int)

        return ind

    @property
    def shape_exp(self):
        return self.dims + (self.n,)

    #- prepped
    @property
    def recover(self):
        return np.union1d(self.ind_fix,self.ind_rec) # to get from f_out


    #- in
    @property
    def insert_exp(self):
    # jaxxed
        #- in
        if self.bAnalytic:
            inds=[jnp.where(x>=0)[0] for x in self.x.fl]
        else:
            inds=[np.arange(self.shape_exp[i]) for i in range(self.ndim)]

        in_index=jnp.union1d(self.ind_lrn,self.ind_rec) # to insert into f_prepped
        inds.append(in_index)
        return jnp.ix_(*inds)

    @property
    def insert(self):
    # jaxxed

        mie=self.insert_exp

        mi=jnp.ravel_multi_index(mie[:-1],self.dims)
        return (mi,) + (mie[-1],)

class Filter():
    index=None
    _insert_index_jx=None
    def __init__(self):
        self.last=None
        self.out=None

    def finalize(self,stim,dtype,n,ind_lrn,ind_fix,ind_rec=(),bAnalytic=True,last=None,bSplit=False):
        # XXX bSplit
        #[ nPix x nF ]                   (nPix / nSplit) x nSplit x nF

        self.x=stim.x
        self.bIsFourier=stim.bIsFourier
        self.nSplit=stim.nSplit

        self.dtype=dtype
        self.dims=stim.dims
        self.ndim=len(self.dims)

        self.n=n

        if bSplit:
            self._shape=(int(stim.nPix)/self.nSplit,self.nSplit,self.n)
            self._shape_exp=self.dims/self.nSplit + (self.nSplit,self.n,) # XXX CHECK
        else:
            self._shape=(int(stim.nPix),self.n)
            self._shape_exp=self.dims + (self.n,)

        if last is not None:
            self.last=last


        self.bNew=self.last is None or len(self.last)==0

        if self.bNew and len(ind_rec) !=0:
            raise Exception('ind_rec cannot be set if there are no previous filter')

        self.index=_Index(self,ind_lrn,ind_fix,ind_rec,bAnalytic)

        # out
        if not self.bNew:
            self.out=self.last

            # extend
            lshape=np.shape(self.last)[-1]
            if self.n > lshape:
                nshape=self._shape_exp[:-1] + (self.n-lshape,)
                self.out=np.concat((self.out, np.empty(nshape) * np.nan), axis=1)
            elif self.n < lshape:
                raise Exception('specified n is smaller than shape of last: this should not happen!')

        else:
            self.out=np.empty(self._shape_exp) * jnp.nan

        # prepped
        prepped=np.zeros(self._shape_exp)
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
        if self.dtype==jnp.complex64:
            f00=jnp.array(rand_fun[0](rng_key,*rand_fun[1:],shape=self._shape,dtype=jnp.float32),jnp.complex64)
        else:
            f00=rand_fun[0](rng_key,*rand_fun[1:],shape=self._shape,dtype=self.dtype)

        # XXX why?
        if len(jnp.shape(f00)) > 2:
            f00=f00[:,:,0]

        return f00[self.index.insert]

    def insert(self,fIn):
    # jaxxed
        return self.prepped_jx.at[self._insert_index_jx].set(fIn)



    def extract(self,fOut):
    # jaxxed
        self.out=jnp.reshape(self.insert(fOut),self._shape_exp)

#- PLOT

    def plot_out(self,bFourier=None,name='f_out'):
        self._plot(self.out,clim=(-1,1),bFourier=bFourier,name=name)

    def plot_fprepped(self,bFourier=None):
        self._plot(self.prepped_jx,clim=(0,1),bFourier=bFourier)

    def plot_indices(self):
        A=np.ones(self._shape)*-1
        if self._insert_index_jx is not None:
            A[self._insert_index_jx]=2
            if len(self.index.ind_fix )>0:
                A[:,self.index.ind_fix]=0

        # XXX reshape
        self._plot(A,clim=(-1,2))


    def _plot(self,fld,clim=None,bFourier=None,name=''):
        if isinstance(fld,str):
            y=getattr(self,fld)
        else:
            y=fld

        if bFourier is None:
            bFourier=self.bIsFourier

        if np.shape(y)==self._shape:
            y=np.reshape(y,self._shape_exp)

        if bFourier and not self.bIsFourier:
            y=filt.fft(self.x,y)
            y=yn/filt.maxC(y)
        elif not bFourier and self.bIsFourier:
            y=filt.ifft(self.x,y)
            y=y/filt.maxC(y)

        if self.ndim==1:
            if bFourier:
                x=self.x.fl[0]
            else:
                x=self.x.sl[0]
        else:
            if bFourier:
                x=self.x.f[0]
            else:
                x=self.x.s[0]

        for i in range(y.shape[-1]):
            plt.figure(name + '_' + str(i))
            #plt.subplot(1,self.n,i+1)
            if self.ndim == 1:
                filt.plotFT(x,y[:,i])
                if clim:
                    plt.ylim(*clim)
            elif self.ndim == 2:
                plt.imshow(y[:,:,i],extent=self.x.extentf)
                if clim:
                    plt.clim(*clim)




class Nrn():
    _noise_fun=_id
    _normalize_fun=_id
    _rectify_fun=_id
    _corr_fun=_id
    _combine_fun=id
    bNoise=_TypeFunc(True)
    bRectify=_TypeFunc(True)
    normalizeType=_TypeFunc()
    corrType=_TypeFunc()
    bCombine=_TypeFunc(True)
    def __init__(self,fano=1.36,var0=0.23,rmax=5.7,normalizeType='None',bRectify=False,bNoise=False,rho=0,eps=0.001):
        # TODO GET better rMax value
        # TODO check eps value
        self.fano=fano
        self.var0=var0
        self.rmax=rmax
        self.eps=eps

        self.bNoise=bNoise
        self.normalizeType=normalizeType
        self.bRectify=bRectify

        self.rho=rho
        if self.rho is None or ( isinstance(self.rho,str) and self.rho == 'None' ):
            self.corrType='None'
        elif self.rho==0:
            self.corrType='uncorr'
        elif self.rho != 0:
            self.corrType='corr'

        self.filter=Filter()
        self.bFinalized=False

    def copy(self):
       return Nrn(**_get_copy_dict(self,['filter','corrType','bAnalytic','bFinalized','bCombine','bSplit','bFourier']))

    def finalize(self,stim,dtype,n,ind_lrn,ind_fix,ind_rec,bFourier=False,bAnalytic=False,bSplit=False,last=None):
        self.dtype=dtype
        self.bFourier=bFourier
        self.bCombine=bFourier
        self.bSplit=bSplit

        self.filter.finalize(stim,dtype,n,ind_lrn,ind_fix,ind_rec,bAnalytic=bAnalytic,last=last,bSplit=bSplit)

        n=self.filter.n
        self._covMat_upper_index=jnp.array(jnp.triu(jnp.ones((n,n)), 1),dtype=bool)
        self._covMat_lower_index=jnp.array(jnp.tril(jnp.ones((n,n)),-1),dtype=bool)

        self.bFinalized=True

    @property
    def bAnalytic(self):
        if hasattr(self,'filter') and hasattr(self.filter,'index') and hasattr(self.filter.index,'bAnalytic'):
            return self.filter.index.bAnalytic
        else:
            return None


    @staticmethod
    @partial(jit, static_argnames=['rmax'])
    def respond(f,stim,rmax):
        return rmax*jnp.einsum('ij,ikl->jkl',jnp.conjugate(f),stim)

    @staticmethod
    @jit
    def insert(fIn,f_prepped,index):
        return f_prepped.at[index].set(fIn)

    @partial(jit, static_argnames=['self'])
    def lrn_main(self,rng_key,stim,fIn):

        return self.main(rng_key,stim,self.insert(fIn,self.filter.prepped_jx,self.filter._insert_index_jx))

    @partial(jit, static_argnames=['self'])
    def main(self,rng_key,stim,f):

        rm   = self._rectify_fun(self._combine_fun(self.respond(f,stim,self.rmax)))

        Rm   = self._normalize_fun(rm,stim,f,self.eps)

        # XXX  rm or Rm?
        nsVar = self._get_noise_var(self.fano,self.var0,rm)
        covMat=self._corr_fun(nsVar,jnp.ones_like(f,shape=f.shape[-1]),self.rho)

        return self._noise_fun(Rm,rng_key,nsVar), Rm, covMat, nsVar

    #- split
    @staticmethod
    @jit
    def _split__none(r):
        return r

    #- split
    @jit
    def _split__true(self,f,stim):
        # stim [ nPix  x nStim_Ctg x nCtg ]
        # [ nPix x nF ]

        out=jnp.zeros_like(f,shape=(stim.shape[0],stim.shpae[1],f.shape[1]*self.nSplit))
        for i in range(self.nSplit):
            inds=self.allInds[:,i]
            out[:,:,i]=stim[inds,:]*f[inds,:]
        # XXX
        return r

    #- combine
    @staticmethod
    @jit
    def _combine__none(r):
        return r

    @staticmethod
    @jit
    def _combine__true(r):
        return jnp.concat(r.imag,r.real,axis=0)

    #- recify
    @staticmethod
    @jit
    def _rectify__none(R):
        return R

    @staticmethod
    @jit
    def _rectify__true(R):
        return R.at[R < 0].set(0)


    #- max
    # RM??
    @staticmethod
    @jit
    def _max__none(Rn,*_):
        return Rn

    @staticmethod
    @jit
    def _max__1(Rn,r):
        return Rn/jnp.max(Rn)*jnp.max(r)

    @staticmethod
    @jit
    def _max__2(Rn,*_):
        return Rn/jnp.max(Rn)

    @staticmethod
    @jit
    def _max__3(Rn,*_):
        ind=abs(Rn)>1;
        Rn[ind]=1*jnp.sign(Rn[ind]);
        return Rn




    #- normalize
    @staticmethod
    @jit
    def _normalize__none(R,*_):
        return R

    @staticmethod
    @jit
    def _normalize__broad(R,stim,_,err):
        # TODO VMAP
        for i in range(jnp.shape(R)[0]):
            R=R.at[i,:].set(R.at[i,:]/(err+jnp.norm(stim.val[:,i])))

        return R

    @staticmethod
    @jit
    def _normalize__narrow(R,stim,f,eps):
        ## check this is the correct one
        # TODO
        # Rn=r./(S.As' * abs(obj.f));
        return R/(eps + jnp.einsum('ij,ikl->jkl',jnp.abs(f),stim))
    #- noise
    @staticmethod
    @partial(jit, static_argnames=['fano','var0'])
    def _get_noise_var(fano,var0,rm):
        return fano * jnp.abs(rm) + var0

    @staticmethod
    @jit
    def _noise__true(Rm,rng_key,var):
        [_,rng_key] = jxrandom.split(rng_key)
        # XXX is this right? does the first out need to be returned?

        return Rm +  (jxrandom.normal(rng_key,Rm.shape) * jnp.sqrt(var))

    @staticmethod
    @jit
    def _noise__none(Rm,*_):
        return Rm

    #- corr
    @staticmethod
    @jit
    def _corr__none(*_):
        return 0

    @staticmethod
    @jit
    def _corr__uncorr(nsVar,covDiag0,*_):
        # noise variance
        # MATCH CONSTANT ADDITIVE TO AVERAGE SCALED ADDITIVE NOISE VARIANCE
        # INTERNAL FILTER RESPONSE COVARIANCE MATRIX (ASSUMING UNCORRELATED NOISE)

        return jnp.diag(jnp.mean(jnp.square(nsVar)) * covDiag0)

    # TODO pytree-ize
    @partial(jit, static_argnames=['self','rho'])
    def _corr__corr(self,nsVar,covDiag0,rho):
        covMat=Nrn._corr__uncorr(nsVar,covDiag0)

        C = mh.corr2cov(covMat,rho)
        covMat[self._covMat_upper_index] = C
        covMat[self._covMat_lower_index] = C
        return covMat


class Model():
    _post_fun=_id
    _est_fun=_id
    _model_fun=_id
    _response_fun=_id
    modelType=_TypeFunc()
    responseType=_TypeFunc()
    def __init__(self,modelType='gss',responseType='basic'):
        self.modelType=modelType
        self.responseType=responseType

    def copy(self):
       return Model(**_get_copy_dict(self))

    def finalize(self,stim,dtype):
        #self._lAll0=jnp.zeros((stim.nStim_Ctg, stim.nCtg, stim.nCtg),dtype=dtype)
        pass


    @partial(jit, static_argnames=['self'])
    def lrn_main(self,R,Rm,covMat,nsVar):
        return self._model_fun(*self._response_fun(R,Rm,covMat,nsVar))

    @staticmethod
    @jit
    def _response__mean(_,Rm,covMat,nsVar):
        return Rm,Rm,covMat,nsVar

    @staticmethod
    @jit
    def _response__basic(R,Rm,covMat,nsVar):
        return R, Rm, covMat,nsVar

    #- models

    #@partial(jit, static_argnames=['self'])
    @staticmethod
    @jit
    def _model__gss(R,Rm,covMat,_):
        #: R   [ nF   x nStim_Ctg x ctg]
        #: lAll[nStim_ctg x ctg x ctg]

        R0=np.transpose(R-jnp.mean(R,axis=1,keepdims=True),(1,2,0))
        mvn=vmap(lambda cov_diag : lax.exp(lmvn0(R0,cov = jnp.diag(cov_diag) + covMat)), in_axes=(1,),out_axes=2)
        # XXX is abs right for complex values?  could for sure merge responses across nF, could also average
        #return jnp.abs(mvn(jnp.var(Rm,axis=1)))
        return mvn(jnp.var(Rm,axis=1))


    @staticmethod
    @jit
    def _model__full(R,Rm,_,nsVar):
        #nF, nStm_ctg, nCtg = R.shape
        ##pAll [ nStim x nCtg x nCtg]

        #Y = jnp.zeros_like(R,shape=(nStm_ctg,nCtg,nCtg))

        #for k in range(nCtg):
        #    for l in range(nStm_ctg):
        #        for i in range(nCtg):
        #            for j in range(nStm_ctg):
        #                Y[l,k,i] += jnp.prod(1/sigma[:,j,i],axis=0) * jnp.exp(-0.5 * jnp.sum(((R[:,l,k] - Rm[:, j,i]) / sigma[:, j,i]) ** 2, axis=0))


        R_exp        =R[:,:,None,:,None]
        Rm_exp      =Rm[:,None,:,None,:]
        sigma_exp=jnp.sqrt(nsVar[:,None,:,None,:])
        Y = jnp.prod(1/sigma_exp,axis=0) * jnp.exp(-0.5 * jnp.sum(((R_exp - Rm_exp) / sigma_exp) ** 2, axis=0))
        return Y / jnp.sum(Y,axis=2,keepdims=True)




class Objective:
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
        if isinstance(errType,(int, float)):
            self.errType='l'
            self.l=errType
        else:
            self.errType=errType
        self.lossType=lossType

        #- posterior
        if _bCopy:
            pass
        elif   self.errType == 'mle':
            if bPosterior is not None:
                assert Exception('postType must not be set for for errType=mle')
            bPosterior=False
        elif self.errType == 'map':
            if bPosterior is not None:
                assert Exception('postType must not be set for errType=map')
            bPosterior=True
        elif bPosterior is None:
            bPosterior=True
        self.bPosterior=bPosterior

        #- estType
        if _bCopy:
            pass
        elif   self.errType == 'mle':
            if estType is not None:
                assert Exception('estType must not be set for for errType=mle')
            estType=None
        elif self.errType == 'map':
            if estType is not None:
                assert Exception('estTyep must not be set for for errType=map')
            estType=None
        self.estType=estType


    def copy(self):
       return Objective(**_get_copy_dict(self))

    def finalize(self,stim,dtype):
        pass
        #self._yHat0=jnp.zeros((stim.nStim_Ctg,stim.nCtg),dtype=dtype)
        #self._correct0=jnp.zeros((stim.nStim_Ctg,stim.nCtg),dtype=dtype)

    @partial(jit, static_argnames=['self'])
    def lrn_main(self,lAll,stimweights,yCtg,Y):
        return self._loss_fun(self._err_fun(self._est_fun(self._posterior_fun(lAll),Y),yCtg),stimweights)

    #- post_fun
    @staticmethod
    @jit
    def _posterior__true(lAll):
        return lAll/jnp.sum(lAll,axis=1,keepdims=True)

    @staticmethod
    @jit
    def _posterior__none(lAll):
        return lAll

    #- estimation
    @staticmethod
    @jit
    def _est__none(post,Y):
        return post

    @staticmethod
    @jit
    def _est__median(post,Y):
    # XXX check
        return jnp.interp(
            0.5 * jnp.ones((post.shape[0], 1)),
            jnp.cumsum(post,axis=1),
            Y,
            left=jnp.min(Y),
            right=jnp.max(Y)
        )

    @staticmethod
    @jit
    def _est__mean(post,Y):
    # XXX check
        return jnp.dot(post,Y)

    @staticmethod
    @jit
    def _est__mode(post,Y):
    # XXX check
        return jnp.dot(jnp.max(post,axis=0),Y)

    @staticmethod
    @jit
    def _est__cmean(post,Y):
        return jnp.angle( jnp.sum( jnp.transpose(post) * jnp.exp(1j*Y) ) / sum(jnp.transpose(post)) )

    #- error
    @staticmethod
    @jit
    def _err__mle(lAll,_):
        return jnp.log(Objective._at_correct(lAll))

    @staticmethod
    @jit
    def _err__map(pAll,_):
        return -jnp.log(Objective._at_correct(pAll))

    @staticmethod
    @jit
    def _err__l2(yHat,yCtg):
        return jnp.pow(jnp.abs(yHat-yCtg),2)

    #- loss
    @staticmethod
    @jit
    def _loss__mean(err,stimweights):
        return jnp.mean(err*stimweights)

    @staticmethod
    @jit
    def _loss__median(err,stimweights):
        return jnp.median(err*stimweights)

    #- helpers
    @staticmethod
    @jit
    def _at_correct(inAll):
        return jnp.diagonal(inAll, axis1=-2, axis2=-1)

        # LOOP VERSION
        #lCorrect=jnp.zeros_like(inAll,shape=inAll.shape[:-1])
        #for c in range(jnp.shape(inAll)[2])
        #    lCorrect=lCorrect.at[:,c].set(inAll.at[:,c,c].get())
        #return lCorrect

class Optimizer():
    def __init__(self,optimizerType='adam',projectionType=['l2_ball',1],lRate0=1e-1,nIterMax=1000,f0_jxrand_fun=['ball',1]):
        # ball is le instead of eq
        self.optimizerType=optimizerType
        self.projectionType=projectionType
        self.lRate0=lRate0
        self.nIterMax=nIterMax
        self.f0_jxrand_fun=f0_jxrand_fun
        self._f0_jxrand_fun=f0_jxrand_fun
        if not isinstance(f0_jxrand_fun[0],str):
            self._f0_jxrand_fun[0]=f0_jxrand_fun[0]
        else:
            self._f0_jxrand_fun[0]=getattr(jxrandom,f0_jxrand_fun[0])

    def copy(self):
        return Optimizer(**_get_copy_dict(self))

    @property
    def _projection(self):
        #l2_sphere, l2_ball, l1_all, l1_sphere
        return getattr(optax.projections,'projection_' + self.projectionType[0])

    @property
    def _projection_params(self):
        if len(self.projectionType)==1:
            return []
        else:
            return self.projectionType[1:]

    @property
    def optimizer(self):
        return getattr(optax,self.optimizerType)

    @staticmethod
    @partial(jit, static_argnames=['proj_fun'])
    def insert_project_extract(params,prepped,index,proj_fun,proj_params):
        out=params.copy()
        fNew=prepped.at[index].set(params['f'])

        vectorized_proj_fun = vmap(lambda f: proj_fun({'f': f}, *proj_params)['f'], in_axes=-1, out_axes=-1)
        fNew_updated = fNew.at[..., :].set(vectorized_proj_fun(fNew))

        out['f']=fNew_updated.at[index].get()
        return out


    def minimize(self,f0,rng,stim,filter,loss_fun,opt_state=None):
        opt_fun=self.optimizer

        proj_fun_full=lambda params: self.insert_project_extract(params,filter.prepped_exp_jx,filter._insert_index_exp_jx,self._projection,self._projection_params)

        optimizer = opt_fun(self.lRate0)

        # f0
        params = {"f": f0}
        if opt_state is None:
            params = proj_fun_full(params)
            opt_state=optimizer.init(params)


        for step in range(self.nIterMax):
            params, loss_value, opt_state,  rng = self.step(optimizer,params,loss_fun,proj_fun_full,opt_state,rng,stim)

            if step % 100==0 or step==self.nIterMax-1:
                print(f'step {step}, loss: {loss_value}')


        return params, opt_state, rng


    @staticmethod
    @partial(jit, static_argnames=['stim','optimizer','loss_fun','proj_fun_full'])
    def step(optimizer,params,loss_fun,proj_fun_full,opt_state,rng,stim):
        [rng,rng_key] = jxrandom.split(rng)
        loss_value, grads = value_and_grad(loss_fun,holomorphic=False)(params, rng_key, stim.val, stim.weights, stim.yCtg, stim.Y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        cparams = proj_fun_full(params)

        return cparams, loss_value, opt_state, rng

class Unit:
    def __init__(self,stim,nrn,model,objective,optimizer=None,seed=None,rng=None,rng_last=None):
        self.nrn=nrn
        self.stim=stim
        self.model=model
        self.objective=objective
        self.optimizer=optimizer

        if seed is None:
            seed=666
        self.seed=seed

        if rng is None:
            self.rng=jxrandom.key(seed)
        else:
            self.rng=rng

        self.rng_last=rng_last

    def split(self,stimInd=None):

        nrn=self.nrn.copy()
        model=self.model.copy()
        objective=self.objective.copy()
        if self.optimizer is None:
            optimizer=None
        else:
            optimizer=self.optimizer.copy()

        if self.nrn.bFinalized:
            dtype=self.nrn.dtype
            n=self.nrn.filter.n

            ind=self.nrn.filter.index

            stim=self.stim.finalize(self.nrn.dtype,
                                    stimInd,
                                    self.nrn.bFourier,
                                    self.nrn.bSplit
            )
            nrn.finalize(stim,
                         self.nrn.dtype,
                         self.nrn.filter.n,
                         ind.ind_lrn,
                         ind.ind_fix,
                         ind.ind_rec,
                         self.nrn.bFourier,
                         self.nrn.bAnalytic,
                         last=self.last
            )
            nrn.filter.out=self.nrn.filter.out
        else:
            stim=self.stim

        [self.rng,rng_key] = jxrandom.split(self.rng)

        return Unit(stim,nrn,model,objective,optimizer=optimizer,seed=self.seed,rng=rng_key,rng_last=self.rng_last)

    def _finalize(self,n,ind_lrn,ind_fix=(),ind_rec=(),fourierType=None,bSplit=None,stimInd=None,dtype=None,last=None):

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
            if self.nrn.bFinalized:
                bSplit=self.nrn.bSplit
            else:
                bSplit=False

        if dtype is None:
            if self.nrn.bFinalized:
                dtype=self.nrn.dtype
            elif bFourier:
                dtype=jnp.complex64
            else:
                dtype=jnp.float32


        self.stim=self.stim.finalize(dtype,
                                     stimInd,
                                     bFourier,
                                     bSplit,
        )

        self.nrn.finalize(self.stim,
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

        if self.optimizer is None:
            raise Exception('Optimizer was not provided')


#- LEARN MODES
    def train_new(self,n,fourierType=None,bSplit=None,stimInd=None,dtype=None,optimizer=None):
        if optimizer is not None:
            self.optimizer=optimizer

        ind_lrn=np.arange(n)
        ind_fix=()
        ind_rec=()

        self._finalize(n,
                      ind_lrn,
                      ind_fix,
                      ind_rec,
                      fourierType=fourierType,
                      stimInd=stimInd,
                      dtype=dtype,
                      bSplit=bSplit
        )

        [rng,rng_key] = jxrandom.split(self.rng)
        f0=self.filter.get_f0(rng_key,self.optimizer._f0_jxrand_fun)

        self.out_params,self.opt_state,self.rng_last=self.optimizer.minimize(f0,rng,self.stim,self.filter,self.loss_fun_lrn)
        self.filter.extract(self.out_params['f'])

    def train_recurse(self,ind_rec=None,fourierType=None,bSplit=None,stimInd=None,dtype=None,optimizer=None):
        if optimizer is not None:
            self.optimizer=optimizer

        n=self.nrn.filter.n
        if ind_rec is None:
            ind_rec=jnp.arange(n)
            ind_fix=()
            opt_state=self.opt_state
        else:
            full=jnp.arange(n)
            ind_fix=full(jnp.isin(full,ind_rec,invert=True))
            opt_state=None

        ind_lrn=()
        self._finalize(n,
                      ind_lrn,
                      ind_fix,
                      ind_rec,
                      fourierType=fourierType,
                      bSplit=bSplit,
                      stimInd=stimInd,
                      dtype=dtype,
                      last=self.out

        )


        [rng,rng_key] = jxrandom.split(self.rng_last)

        self.out_params,self.opt_state,self.rng_last=self.optimizer.minimize(self.out,rng,self.stim,self.filter,self.loss_fun_lrn,opt_state=opt_state)
        self.filter.extract(self.out_params['f'])

    def train_append(self,n_append,fourierType=None,bSplit=None,stimInd=None,dtype=None,optimizer=None):
        if optimizer is not None:
            self.optimizer=optimizer

        n0=self.nrn.filter.n
        n=n0+n_append

        ind_lrn=jnp.arange(n0,n)
        ind_fix=jnp.arange(n0)
        ind_rec=()

        self._finalize(n,
                      ind_lrn,
                      ind_fix,
                      ind_rec,
                      fourierType=fourierType,
                      bSplit=bSplit,
                      stimInd=stimInd,
                      dtype=dtype,
                      last=self.out
        )

        [rng,rng_key] = jxrandom.split(self.rng_last)
        f0=self.filter.get_f0(rng_key,self.optimizer._f0_jxrand_fun)

        self.out_params,self.opt_state,self.rng_last=self.optimizer.minimize(f0,rng,self.stim,self.filter,self.loss_fun_lrn)
        self.filter.extract(self.out_params['f'])

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
        return self.loss_fun({'f':self.filter.out},self.rng,self.stim.val,self.stim.weights,self.stim.yCtg,self.stim.Y)

    @property
    def response(self):
        return self.nrn.main(self.rng,self.stim.val,self.filter.out)

    @property
    def error(self):
        return self.objective._err_fun(self.objective._est_fun(self.objective._posterior_fun(self.response),self.stim.Y),self.stim.yCtg)

    @partial(jit, static_argnames=['self'])
    def loss_fun_lrn(self,params,rng_key,stimval,stimweights,yCtg,Y):
        return self.objective.lrn_main(self.model.lrn_main(*self.nrn.lrn_main(rng_key,stimval,params['f'])),stimweights,yCtg,Y)

    @partial(jit, static_argnames=['self'])
    def loss_fun(self,params,rng_key,stimval,stimweights,yCtg,Y):
        return self.objective.lrn_main(self.model.lrn_main(*self.nrn.main(rng_key,stimval,params['f'])),stimweights,yCtg,Y)

    #- out
    def plot_out(self,bFourier=None,name='f_out'):
        self.filter.plot_out(bFourier=bFourier,name=name)

    def plot_spinner(self,name='plot_spinner'):
    # R = [nF x nStimCtg x nCtg]
        R,Rm,covMat=self.response

        colors=cm.rainbow(np.linspace(0,1,self.stim.nCtg))
        pairs=list(combinations(range(Rm.shape[0]),2))
        for j in range(len(pairs)):
            plt.figure(name + str(j))
            for i in range(self.stim.nCtg):
                if self.stim.bIsFourier:
                    plt.subplot(1,2,1)
                    plt.scatter( Rm[pairs[j][0],:,i].real,Rm[pairs[j][1],:,i].real,color=colors[i],marker='.')
                    plt.subplot(1,2,2)
                    plt.scatter( Rm[pairs[j][0],:,i].imag,Rm[pairs[j][1],:,i].imag,color=colors[i],marker='.')
                else:
                    plt.scatter( Rm[pairs[j][0],:,i],Rm[pairs[j][1],:,i],color=colors[i],marker='.')



