#: f    [ nPix x nF ]
#: stim [ nPix  x nStim ] -> [ nPix x nStim_Ctg x ctg ]
#: R    [ nF    x nStim ] -> [ nF   x nStim_Ctg x ctg]
#: lAll [ nStim x nCtg ]  ->
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
from jax import grad,jit,vmap,lax
from jax.scipy.stats import multivariate_normal as mvn
import statsmodels.stats.moment_helpers as mh
import matplotlib.pyplot as plt
import optax
from functools import partial
import Filter as filt
from jax._src.numpy.util import promote_dtypes_inexact


def _get_copy_dict(instance,excl=[]):
    flds = [attr for attr in vars(instance) if not attr.startswith('_') and attr not in excl]

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
        self.val=self.default

    def __set_name__(self, _, name):
        if self.bBinary and name[0]!='b':
            raise Exception('binary _TypeFuncs must have a name that begins with "b"')
        elif not self.bBinary and name[-4:]!='Type':
            raise Exception('_TypeFuncs must have a name that ends with "Type"')
        self.name = name

    def __get__(self, instance, _):
        if instance is not None:
            return self.val
        return self

    def __set__(self, instance, value):
        self.instance = instance
        if self.bBinary:
            if not isinstance(value,bool) and value is not None:
                raise Exception('Value must be bool')
        elif not isinstance(value,str) and value is not None:
            raise Exception('Value must be a string')

        self.val = value
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
        if self.bBinary:
            if self.val:
                return 'true'
            else:
                return 'none'
        else:
            if not self.val or ( isinstance(self.val,str) and self.val.lower()=='none'):
                return 'none'
            else:
                return self.val

class Stim:
    def index(self,index=(0,0)):
        return jnp.reshape(self.val[:,index[0],index[1]],self.dims)

    def __getitem__(self,index):
        yCtgInd=self.yCtgInd[index].flatten(order='F')
        return Stim(self.x,self.val[...,index[0],index[1]],yCtgInd, self.Y, self.bIsFourier, _bShape=True, _weights=self.weights[index])

    def __init__(self,x,stimuli,yCtgInd,Y,bStimIsFourier=False,_bShape=False,_weights=np.empty):

        self.bIsFourier=bStimIsFourier

        self.x=x

        self.ctg=jnp.unique(yCtgInd)
        self.nCtg=len(self.ctg)

        self.Y=Y # unique
        self._Y_transpose=jnp.transpose(Y)
        self.yCtg=Y[yCtgInd]

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
            val=np.zeros((self.nPix,self.nStim_Ctg,self.nCtg))
            for c in range(self.nCtg):
                val[:,:nStimCtg[c],c]=stimuli[:,np.where(yCtgInd==c)[0]]
                weights[:nStimCtg[c],c]=1
            self.val=jnp.array(val)
            self.weights=jnp.array(weights)
        else:
            self.nStim_Ctg=jnp.shape(stimuli)[-2]
            self.val=stimuli
            self.weights=_weights

        yCI=jnp.zeros((self.nStim_Ctg,self.nCtg),dtype=int)
        for c in range(self.nCtg):
            yCI=yCI.at[:,c].set(c)
        self.yCtgInd=jnp.array(yCI)
        self.nStim=self.nStim_Ctg*self.nCtg

    def _fft(self):
        if self.bStimIsFourier:
            raise Exception('Stim is already in the fourier domain')
        self.val=filt.fft(self.x,self.val)
        self.bStimIsFourier=True

    def _ifft(self):
        if not self.bStimIsFourier:
            raise Exception('Stim is already out of the fourier domain')
        self.val=filt.ifft(self.x,self.val)
        self.bStimIsFourier=False

    def finalize(self,dtype,index,bFourier):
        if bFourier and not self.bIsFourier:
            self._fft()
        elif not bFourier and self.bIsFourier:
            self._ifft()

        if not ( isinstance(self.val,jnp.ndarray) and jnp.issubdtype(self.val,dtype)):
            self.val=jnp.array(self.val,dtype=dtype)

        if index:
            return self.index(index)
        else:
            return self

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
            filt.plotFT(self.x,stim)
        elif self.ndim==2:
            plt.imshow(self.index(index),extent=self.x.extents)
        elif self.ndim==3:
            raise Exception('TODO')
        elif self.ndim==4:
            raise Exception('TODO')

    @classmethod
    def load(cls,fname):
        # TODO
        return Stim(x,stimuli,yCtgInd,X)

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
    # not jaxxed
        #- in
        if self.bAnalytic:
            inds=[jnp.where(x>=0)[0] for x in self.x.fl]
        else:
            inds=[np.arange(self.shape_exp[i]) for i in range(self.ndim)]

        in_index=np.union1d(self.ind_lrn,self.ind_rec) # to insert into f_prepped
        inds.append(in_index)
        return np.ix_(*inds)

    @property
    def insert(self):
    # jaxxed

        mie=self.insert_exp

        mi=np.ravel_multi_index(mie[:-1],self.dims)
        return (jnp.array(mi),) + (jnp.array(mie[-1]),)

class Filter():
    index=None
    _insert_index_jx=None
    def __init__(self):
        self.last=None

    def finalize(self,stim,dtype,n,ind_lrn,ind_fix,ind_rec=(),bAnalytic=True):
        self.x=stim.x

        self.dtype=dtype
        self.dims=stim.dims
        self.ndim=len(self.dims)

        self.n=n

        self._shape=(int(stim.nPix),self.n)
        self._shape_exp=self.dims + (self.n,)
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
                self.out=np.concat((self.out, np.empty(shape) * np.nan), axis=1)
            elif self.n < lshape:
                raise Exception('specified n is smaller than shape of last: this should not happen!')

        else:
            self.out=np.empty(self._shape_exp) * jnp.nan

        # prepped
        prepped=np.zeros(self._shape_exp)
        prepped[...,self.index.recover]=self.out[...,self.index.recover]
        self.prepped_jx=jnp.array(np.reshape(prepped,self._shape),dtype=dtype)

        # save insert index as a constant
        self._insert_index_jx=self.index.insert

    def get_f0(self,rng_key,rand_fun):
    # jaxxed
        f00=rand_fun[0](rng_key,*rand_fun[1:],shape=self._shape,dtype=self.dtype)

        # XXX why?
        if len(jnp.shape(f00)) > 2:
            f00=f00[:,:,0]


        return f00[self.index.insert]

    def insert(self,fIn):
    # jaxxed
        fNew=self.prepped_jx.at[self._insert_index_jx].set(fIn)
        return fNew

    def extract(self,fOut):
        # NOT jaxxed
        self.out=jnp.reshape(self.insert(fOut),self._shape_exp)
#- PLOT

    def plot_out(self):
        self._plot(self.out,clim=(-1,1))

    def plot_fprepped(self):
        self._plot(self.f_prepped,clim=(0,1))

    def plot_indices(self):
        A=np.ones(self._shape)*-1
        if self._insert_index_jx is not None:
            A[self._insert_index_jx]=2
            if len(self.index.ind_fix )>0:
                A[:,self.index.ind_fix]=0

        # XXX reshape
        self._plot(A,clim=(-1,2))


    def _plot(self,fld,clim=None):
        if isinstance(fld,str):
            y=getattr(self,fld)
        else:
            y=fld

        if np.shape(y)==self._shape:
            y=np.reshape(y,self._shape_exp)

        for i in range(self.n):
            plt.figure(i)
            #plt.subplot(1,self.n,i+1)
            plt.imshow(y[:,:,i],extent=self.x.extentf)
            if clim:
                plt.clim(*clim)




class Nrn():
    _noise_fun=_id
    _normalize_fun=_id
    _rectify_fun=_id
    _corr_fun=_id
    bNoise=_TypeFunc(True)
    bRectify=_TypeFunc(True)
    normalizeType=_TypeFunc()
    corrType=_TypeFunc()
    def __init__(self,fano=1.36,sigma0=0.23,rmax=5.7,normalizeType='None',bRectify=False,bNoise=False,rho=0,eps=0.001):
        # TODO GET better rMax value
        # TODO check eps value
        self.fano=fano
        self.sigma0=sigma0
        self.rmax=rmax
        self.eps=eps

        self.bNoise=bNoise
        self.normalizeType=normalizeType
        self.bRectify=bRectify

        self.rho=rho
        if self.rho is None:
            self.corrType='None'
        elif self.rho==0:
            self.corrType='uncorr'
        elif self.rho != 0:
            self.corrType='corr'

        self.filter=Filter()
    def copy(self):
       return Nrn(**_get_copy_dict(self,['filter']))

    def finalize(self,stim,dtype,n,ind_lrn,ind_fix,ind_rec,bAnalytic=True):
        self.dtype=dtype

        self.filter.finalize(stim,dtype,n,ind_lrn,ind_fix,ind_rec,bAnalytic)

        n=self.filter.n
        self._rShape=(n,stim.nStim_Ctg,stim.nCtg)
        self._covDiag0=jnp.ones(n,dtype=dtype)
        self._covMat_upper_index=jnp.array(jnp.triu(jnp.ones((n,n)), 1),dtype=bool)
        self._covMat_lower_index=jnp.array(jnp.tril(jnp.ones((n,n)),-1),dtype=bool)


    def lrn_main(self,rng_key,stim,fIn):
        f=self.filter.insert(fIn)


        rm   = self._rectify_fun(self.rmax*jnp.einsum('ij,ikl->jkl',jnp.conjugate(f),stim))

        # XXX todo, mv error
        Rm   = self._normalize_fun(rm,stim,f,self.eps)

        nsVar = self.fano * jnp.abs(rm) + self.sigma0
        covMat=self._corr_fun(nsVar)

        return Rm + self._noise_fun(rng_key,nsVar), Rm, covMat

    #- max
    # RM??
    def _max__none(self,Rn,*_):
        return Rn

    def _max__1(self,Rn,r):
        return Rn/jnp.max(Rn)*jnp.max(r)

    def _max__2(self,Rn,*_):
        return Rn/jnp.max(Rn)

    def _max__3(self,Rn,*_):
        ind=abs(Rn)>1;
        Rn[ind]=1*jnp.sign(Rn[ind]);
        return Rn


    #- noise
    def _noise__true(self,rng_key,var):

        [_,rng_key] = jxrandom.split(rng_key)
        # XXX is this right? does the first out need to be returned?

        return jnp.reshape(
            jxrandom.multivariate_normal(
                rng_key,
                jnp.zeros(var.size)[:,jnp.newaxis],
                var.ravel()[:,jnp.newaxis,jnp.newaxis]
            ),
            self._rShape
        )
    def _noise__none(self,*_):
        return 0

    #- recify
    def _rectify__none(self,R):
        return R

    def _rectify__true(self,R):
        R[R < 0]=0
        return R

    #- normalize
    def _normalize__none(self,R,*_):
        return R

    def _normalize__broad(self,R,stim,_,err):
        # TODO VMAP
        for i in range(jnp.shape(R)[0]):
            R=R.at[i,:].set(R.at[i,:]/(err+jnp.norm(stim.val[:,i])))

        return R

    def _normalize__narrow(self,R,stim,f,err):
        ## check this is the correct one
        # TODO
        return R/(err + jnp.vdot(jnp.transpose(stim.as_transpose),jnp.abs(f)))

    def _corr__none(self,*_):
        return 0

    def _corr__uncorr(self,nsVar):
        # noise variance
        # MATCH CONSTANT ADDITIVE TO AVERAGE SCALED ADDITIVE NOISE VARIANCE
        # INTERNAL FILTER RESPONSE COVARIANCE MATRIX (ASSUMING UNCORRELATED NOISE)
        return jnp.diag(jnp.mean(jnp.square(nsVar)) * self._covDiag0)

    def _corr_fun__corr(self,nsVar):
        covMat=self._corr__uncorr(nsVar)

        C = mh.corr2cov(covMat,self.rho)
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
        # XXX rm?
        self._lAll0=jnp.zeros((stim.nStim_Ctg, stim.nCtg, stim.nCtg),dtype=dtype)

    def lrn_main(self,R,Rm,covMat):
        return self._model_fun(*self._response_fun(R,Rm,covMat))

    def _response__mean(self,_,Rm,covMat):
        return Rm,Rm,covMat

    def _response__basic(self,R,Rm,covMat):
        return R, Rm, covMat

    #- models
    def _model__gss(self,R,Rm,covMat):
        # XXX check
        #: R   [ nF   x nStim_Ctg x ctg]
        #: lAll[nStim_ctg x ctg x ctg]



        #def compute_pdf(R,r):
        nStim_ctg=jnp.shape(R)[1]
        nCtg=jnp.shape(R)[2]
        nF=jnp.shape(R)[0]
        out=jnp.zeros((nStim_ctg,nCtg,nCtg))

        R0=np.transpose(R-jnp.mean(R,axis=1,keepdims=True),(1,2,0))
        #print(R0.shape)     #  3,51,2
        #print(out.shape)    # 51, 2,2
        #print(covMat.shape) # 3,3
        vr=jnp.var(Rm,axis=1)
        for c in range(nCtg):
            out=out.at[:,:,c].set(lax.exp(lmvn0(R0,cov=jnp.diag(vr[:,c])+covMat)))


        #out=vmap(compute_pdf)(R,R,(None, 2),2)
        return out



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
       return Objective(**_get_copy_dict(self,['l']))

    def finalize(self,stim,dtype):
        self._Y_transpose=stim._Y_transpose
        self._Y=stim.Y
        self._yMin=jnp.min(stim.Y)

        self._yHat0=jnp.zeros((stim.nStim_Ctg,stim.nCtg),dtype=dtype)
        self._correct0=jnp.zeros((stim.nStim_Ctg,stim.nCtg),dtype=dtype)

    def lrn_main(self,lAll,stimweights,yCtg):
        return self._loss_fun(self._err_fun(self._est_fun(self._posterior_fun(lAll)),yCtg),stimweights)

    #- post_fun
    def _posterior__true(self,lAll):
        return lAll/jnp.sum(lAll,axis=1,keepdims=True)

    def _posterior__none(self,lAll):
        return lAll

    #- estimation
    def _est__none(self,post):
        return post

    def _est___med(self,post):
        # XXX TODO and check
        yHat  = self._yHat0.copy()
        for i in range(self._nStim):
            yHat=yHat.at[i].set(jnp.interp1(jnp.cumsum(post[i,:]), self._Y ,0.5,'linear',self._yMin))
        return yHat

    def _est__mean(self,post):
        # XXX check
        return post*self._Y_transpose

    def _est__mode(self,post):
        # XXX check
        return jnp.max(post,axis=0)*self._Y_transpose

    def _est__cmean(self,post):
        return jnp.angle( jnp.sum( jnp.transpose(post) * jnp.exp(1j*self._Y_transpose) ) / sum(jnp.transpose(post)) )

    #- error
    def _err__mle(self,lAll,_):
        return jnp.log(self._at_correct(lAll))

    def _err__map(self,pAll,_):
        return -jnp.log(self._at_correct(pAll))

    def _err__l(self,yHat,yCtg):
        return jnp.jnp.pow(jnp.abs(yHat-yCtg),self.l)

    #- loss
    def _loss__mean(self,err,stimweights):
        return jnp.mean(err*stimweights)

    def _loss__median(self,err,stimweights):
        return jnp.median(err*stimweights)

    #- helpers
    def _at_correct(self,inAll):
        lCorrect=self._correct0.copy()

        # [nStim_ctg x ctg_stim x ctg ]
        # [ nStim_ctg x ctg_stim ]

        #    lCorrect.at[:,c].set(inAll[:,:,c])
        #atC=lambda c : lCorrect.at[:,c].set(inAll[:,:,c])
        nCtg=jnp.shape(inAll)[2]
        for c in range(nCtg):
            lCorrect=lCorrect.at[:,c].set(inAll.at[:,c,c].get())
        return lCorrect

class Optimizer():
    def __init__(self,optimizerType='adam',projectionType=['l2_sphere',1],lRate0=1e-1,nIterMax=1000,f0_jxrand_fun=['ball',1]):
        self.optimizerType=optimizerType
        self.projectionType=projectionType
        self.lRate0=lRate0
        self.nIterMax=nIterMax
        self._f0_jxrand_fun=f0_jxrand_fun
        if not isinstance(f0_jxrand_fun[0],str):
            self._f0_jxrand_fun[0]=f0_jxrand_fun[0]
        else:
            self._f0_jxrand_fun[0]=getattr(jxrandom,f0_jxrand_fun[0])

    def copy(self):
        return Optimizer(**_get_copy_dict(self))

    @property
    def f0_jxrand_fun(self):
        self._f0_jxrand_fun.__name__

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

    def minimize(self,rng,f0,stim,loss_fun):
        opt_fun=self.optimizer
        proj_fun=self._projection
        proj_params=self._projection_params

        optimizer = opt_fun(self.lRate0)

        # f0
        [rng,rng_key] = jxrandom.split(rng)
        params = {"f": f0}

        opt_state=optimizer.init(params)

        for _ in range(self.nIterMax):
            [rng,rng_key] = jxrandom.split(rng)
            grads = grad(loss_fun)(params, rng_key, stim.val, stim.weights, stim.yCtg)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            params = proj_fun(params,*proj_params)

        return params, opt_state

class Trn:
    def __init__(self,nrn,model,objective,stim,seed=666):
        self.nrn=nrn
        self.model=model
        self.objective=objective
        self.stim=stim

        self.seed=seed
        self.rng=jxrandom.key(seed)

    def finalize(self,n,ind_lrn,ind_fix=(),ind_rec=(),fourierType=2,stimInd=None,dtype='float32'):
        bFourier=fourierType==0
        bAnalytic=fourierType==2

        # copy
        nrn=self.nrn.copy()
        model=self.model.copy()
        objective=self.objective.copy()

        # finalize
        stim=self.stim.finalize(dtype,stimInd,bFourier)
        nrn.finalize(stim,dtype,n,ind_lrn,ind_fix,ind_rec,bAnalytic)
        model.finalize(stim,dtype)
        objective.finalize(stim,dtype)

        [self.rng,rng_key] = jxrandom.split(self.rng)

        return Iter(stim,nrn,model,objective,rng_key)

class Iter:
    def __init__(self,stim,nrn,model,objective,rng_key):
        self.nrn=nrn;
        self.stim=stim
        self.model=model
        self.objective=objective
        self.rng=rng_key

    def minimize(self,optimizer):
        # XXX is this type of copy sufficient?
        self.optimizer=optimizer.copy()

        [self.rng,rng_key] = jxrandom.split(self.rng)
        f0=self.nrn.filter.get_f0(rng_key,self.optimizer._f0_jxrand_fun)

        [self.rng,rng_key] = jxrandom.split(self.rng)
        self.out_params,self.optstate=self.optimizer.minimize(rng_key,f0,self.stim,self.get_loss_fun())
        self.nrn.filter.extact(self.out_params['f'])

    def get_loss_fun(self):
        return lambda params, rng_key, stimval,stimweights, yCtg : self.objective.lrn_main(self.model.lrn_main(*self.nrn.lrn_main(rng_key,stimval,params['f'])),stimweights,yCtg)
