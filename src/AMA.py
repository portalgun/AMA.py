#from scipy.misc import derivative as deriv
from numpy import asarray, array
import jax.numpy as np
#import numpy as np
import numpy.random as nprandom
import jax.random as jxrandom
#from scipy.optimize import approx_fprime
from jax import value_and_grad, jacfwd, jit
#import tensorflow as tf

##import cython
##cimport numpy as cnp

##@cython.boundscheck(False)
##@cython.wraparound(False)

class AMA:
    def __init__(self,S,X,ctgInd,amaOpts,consOpts):
        # NOTE S should already reshaped [Nlvl,Ni,nPix]

        self.bJax=amaOpts.bJax
        if self.bJax:
            # NOTE OVERWRITTING np
            self.rng=jxrandom.PRNGKey(123) #
            np.asarray(S)

        else:
            self.rng=123

        self.consOpts=consOpts

        #cdef const int    self.nPix, self.xPix, self.yPix self.nStim, self.Nf, self.nX, self.Ni, self.nLvl
        self.S    = S
        self.Nf   =int(amaOpts.Nf) # also known as Nq
        self.Nlvl =int(np.max(ctgInd)+1)
        self.nStim=int(ctgInd.shape[0])
        self.Ni   =int(self.nStim/self.Nlvl) # num stim per lvl

        #
        self.nPix =int(S.shape[2])
        self.xPix =int(np.sqrt(self.nPix)) #TODO
        self.yPix =int(np.sqrt(self.nPix)) #TODO


        #cdef const np.ndarray[cnp.double_t, ndim=3] self.S
        # S  [Nlvl, Ni, nPix]


        #cdef const np.ndarray[cnp.double_t, ndim=2] self.X # TODO dim?
        #cdef const np.ndarray[cnp.int_t,    ndim=2] self.labels
        #cdef const np.ndarray[cnp.int_t,    ndim=1] self.ctgInd

        self.X=np.transpose(X)
        self.ctgInd=np.array([ctgInd])
        self.labels=np.reshape(X[ctgInd],(self.Nlvl,self.Ni))

        self.get_Ac()
        self.get_filter_ind()

        #cdef const double self.alpha, self.s0, self.Nf, rmax
        #cdef const int errType, normType, bRectify
        self.alpha   =amaOpts.alpha
        self.s0      =amaOpts.s0
        self.rmax    =amaOpts.rmax
        self.errType =amaOpts.errType
        self.normType=amaOpts.normType
        self.bRectify=amaOpts.bRectify
        self.bMean   =amaOpts.bMean
        self.bPrint  =amaOpts.bPrint
        self.bNormF  =consOpts.bNormF
        self.bPrintCons=consOpts.bPrint

        self.eps=np.sqrt(np.finfo(float).eps)



        ## in/out variable
        #cdef np.ndarray[cnp.double_t, ndim=2] self.f
        #cdef double self.mCost

        ## variable init
        #cdef np.ndarray[cnp.double_t, ndim=3] self.R
        #cdef np.ndarray[cnp.double_t, ndim=3] self.r
        #cdef np.ndarray[cnp.double_t, ndim=3] self.var
        #cdef np.ndarray[cnp.double_t, ndim=3] self.PPstm
        #cdef np.ndarray[cnp.double_t, ndim=3] self.Y

        #np.empty((self.Nlvl,self,Ni,self.Nf),  dtype=double) self.R
        #np.empty((self.Nlvl,self,Ni,self.Nf),  dtype=double) self.r
        #np.empty((self.Nlvl,self,Ni,self.Nf),  dtype=double) self.var
        #np.empty((self.Nlvl,self,Ni,self.Nlvl),  dtype=double) self.PPstm
        #np.empty((self.Nlvl,self,Ni,self.Nlvl),  dtype=double) self.Y

        self.x=np.array([0,0])

        self.R   =np.empty([self.Nlvl,self.Ni,self.Nf])
        self.r   =np.empty([self.Nlvl,self.Ni,self.Nf]) # mean rsp foreach stm
        self.var =np.empty([self.Nlvl,self.Ni,self.Nf]) # var  rsp foreach stm
        self.PPstm  =np.empty([self.Nlvl,self.Ni,self.Nlvl]) # Posterior foreach stm @ each lvl
        self.Y  =np.empty([self.Nlvl,self.Ni,self.Nlvl]) # Posterior foreach stm @ each lvl

        #if self.bJax:
        if self.bJax:
            self.jac=value_and_grad(self.objective_fun_core)
            #self.jac=jacfwd(lambda x: self.objective_fun_core(x))


    def get_filter_ind(self):
        #cdef const np.ndarray[cnp.int_t, ndim=2] self.S2
        self.filter_ind=np.repeat( np.arange(0,self.Nf), self.nPix , axis=0)

    def get_Ac(self):
        #cdef const np.ndarray[cnp.double_t, ndim=2] self.S2
        S2D=self.S.reshape(self.Nlvl,self.Ni,self.xPix,self.yPix)

        #cdef const np.ndarray[cnp.double_t, ndim=2] self.Ac
        self.Ac=np.abs(np.fft.fft2(S2D,axes=(2,3))) # constant, piecwize, XXX
##
    def get_f0(self):
        #cdef np.rand( (nPix,nF),      dtype=double) f0

        # HERE XXX
        if self.bJax:
            sz=(self.nPix*self.Nf,)
            self.rng, rng_input = jxrandom.split(self.rng)
            f0=jxrandom.normal(rng_input,sz)
        else:
            f0=nprandom.randn(self.nPix*self.Nf)

        f0=self.normalize_f(f0)
        return f0

    def normalize_f(self,f):
        sums=np.bincount(self.filter_ind, weights=np.square(f))
        sums=np.repeat(sums,self.nPix)
        f=f/np.sqrt(sums)
        return f

    def objective_fun(self,f,grad):
        #grad is same size as f, partial derivatives at f

        print(grad.size)
        print(self.bJax)

        if grad.size > 0 and self.bJax:
            c,g=self.jac(f)
            cost =   asarray(c).item()
            grad[:] = asarray(g)
        elif self.bJax:
            c=self.objective_fun_core(f)
            cost =   asarray(c).item()
        elif grad.size > 0:
            grad[:]=approx_fprime(f,self.objective_fun_core,e)
            cost=self.objective_fun_core(f)
        else:
            cost=self.objective_fun_core(f)

        if self.bPrint:
            self.print(f,cost)

        return cost

    def objective_fun_core(self,f):
        if self.bNormF:
            f=self.normalize_f(f)
        self.get_response(f)
        self.get_posterior()
        self.get_mean_cost()

        #for property, value in vars(self.mCost).items():
        #    print(property, ":", value)

        return self.mCost

    def get_response(self,f):
        if self.bJax:
            self.rng, rng_input = jxrandom.split(self.rng)
        else:
            rng_input=None

        self.R,self.r,self.var=get_response(f,self.S,self.s0,self.alpha,self.rmax,self.normType,self.Ac,self.Nf,self.nPix,self.xPix,self.yPix,self.nStim,self.bRectify,rng_input)


    def get_posterior(self):
        # each stimulus response gets its own posterior distribution across categories
        # R    [Nlvl, Ni, Nf]
        # r    [Nlvl, Ni, Nf]
        #
        # single LPstm  [Nlvl]
        # PPstm         [Nlvl, Ni, NLvl]

        self.pStd=1/np.prod(np.sqrt(self.var),2)
        if self.bMean==1:
            R=self.r
        else:
            R=self.R


        #for k in range(self.Nlvl):
        #    for l in range(self.Ni):
        #        self.Y[k,l,:]=get_Y_dist(R[k,l,:],self.r,self.var,self.pStd)

        self.Y=np.apply_along_axis(lambda x: self.my_get_Y_dist(x),2,R)


        Z = np.tile(np.sum(self.Y,2)[:,:,np.newaxis],[1,1,self.Nlvl])
        self.PPstm=self.Y/Z

    def my_get_Y_dist(self,R):
        out=get_Y_dist(R,self.r,self.var,self.pStd)
        return out

    def get_mean_cost(self):
        if self.errType==0:
            cost=self.get_map_cost()
        elif self.errType==2:
            cost=self.get_mse_cost()
        self.mCost=np.mean(cost,(0,1))

    def get_map_cost(self):
        # XXX only for pp not ppall
        cost=-np.log(self.PPstm);
        return cost

    def get_mse_cost(self):
        # PPstm   [Nlvl, Ni, NLvl]
        P=np.dot(self.PPstm,self.X)
        # P [Nlvl, Ni]
        cost=np.square(P-self.labels)
        return cost

    def print(self,f,cost):
        for y in range(6):
            for x in range(10):
                print(f'{x}\r', end="")
        print('Cost: ', cost)
        if self.consOpts and self.bPrintCons:
            self.consOpts.print_cons_res(f)

    def reshape_f(self,f):
        f=np.reshape(f,(self.nPix,self.Nf))
        f=np.reshape(f,(self.xPix,self.yPix,self.Nf))
        return f

    def reshape_S(self):
        S=self.S.reshape(self.Nlvl,self.Ni,self.xPix,self.yPix)
        return S

## FUNCTIONS
def get_response(f,S,s0,alpha,rmax,normType,Ac,Nf,nPix,xPix,yPix,nStim,bRectify,rng):
    # X      [Nlvl, 1]
    # ctgInd [Nstm]
    # S      [Nlvl,Ni,nPix]
    # f      [nPix Nf]
    #
    # R      [Nlvl, Ni, Nf}
    #
    # normType1: Ac Nf
    # normType2: Ac,[], xpix,ypix,nstim

    #cdef np.ndarray[cnp.double_t, ndim=2] r
    #cdef np.ndarray[cnp.double_t, ndim=2] s
    #cdef np.ndarray[cnp.double_t, ndim=2] N
    f=np.reshape(f,(nPix,Nf))

    r = rmax*np.dot(S,f)
    var = alpha * abs(r) + s0
    # HERE XXX
    if rng is None:
        N = np.random.normal(0,var)
    elif np.any(rng):
        cov=var.ravel()[:,np.newaxis,np.newaxis]
        m=np.zeros(var.size)[:,np.newaxis]
        N=jxrandom.multivariate_normal(rng,m,cov)
        N= np.reshape(N,r.shape)

    R = r+N

    if bRectify:
        R[R < 0]=0

    if normType==1:
        # Ac Nf
        r=normalize_rsp_brd(r,Ac,Nf)
    elif normType==2:
        # Ac,[],,xpix,ypix,nstim
        r=normalize_rsp_nrw(r,Ac,xPix,yPix,nStim)

    return R,r,var

def normalize_rsp_brd(R,Ac,Nf):
    Nbrd=np.sqrt(np.sum(np.square(Ac),(2,3)))
    Nbrd=np.tile(Nbrd[:,:,np.newaxis],(1,1,Nf))
    R=R/Nbrd
    return R

def normalize_rsp_nrw(R,Ac,f,xPix,yPix,nStim):
    f2=np.reshape(f,(xPix,yPix,nStim))
    Af=abs(fft2(f2),axes=(0,1))
    R=R/np.dot(Ac,Af)
    return R

def get_Y_dist(R,r,var,pStd):
    # R [Nf]
    # r [Nlvl, Ni, Nf]
    # LPstm [Nlvl]
    Y=np.sum(pStd*np.exp(-0.5*np.sum(np.square(R-r)/var,2)),1)
    return Y
