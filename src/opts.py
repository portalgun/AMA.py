import jax.numpy as np
from numpy import asarray, array
from jax import grad as autograd
from jax import jacfwd
class opt_opts:
    def __init__(self,lib='nlopt',algorithm='SLSQP',xtolAbs=1e-6,xtolRel=1e-6,ftolAbs=1e-6,ftolRel=1e-6,maxeval=100,maxtime=1000,step=None,stopval=0):

        #SBLX
        #BOBYQA
        #PRAXIS
        bLB=0
        bUB=0
       
        self.lib=lib
        self.algorithm=algorithm
        self.xtolAbs=xtolAbs
        self.xtolRel=xtolRel
        self.ftolAbs=ftolAbs
        self.ftolRel=ftolRel
        self.maxeval=maxeval
        self.maxtime=maxtime
        self.step=step
        self.stopval=stopval


class cons_opts:
    def __init__(self,bSumE=1,bMultE=0,bLB=1,bUB=1,sumTol=1e-6,LB=-.4,UB=.4,bAug=0,bNormF=0,bPrint=0):

        bPrint=1

        self.bAug=bAug
        self.bNormF=bNormF
        self.bSumE=bSumE
        self.bLB=bLB
        self.bUB=bUB
        self.LB=LB
        self.UB=UB
        self.sumTol=sumTol
        self.bPrint=bPrint

        self.eps=np.sqrt(np.finfo(float).eps)

        self.jac=jacfwd(lambda x: self.vec_mag_one_fun_core(x))
        #self.jac=autograd(lambda x: self.vec_mag_one_fun_core(x))
        #self.bounds=Bounds(-1,1,True)

    def print_cons_res(self,f):
        #if self.bLB:
        print("  LB   " + str(self.LB) + "   " + str(np.min(f)) )

        #if self.bUB:
        print("  UB    " + str(self.UB) + "    " + str(np.max(f)) )

        print("  SUM  " + str(self.vec_mag_one_fun_core(f)))

    #def vec_mag_one_fun(self,x,grad):
    #    out=np.prod(np.bincount(self.filter_ind, weights=np.square(x)))-1
    #    return out

    def vec_mag_one_fun(self,result,x,grad):
        result[:]=asarray(self.vec_mag_one_fun_core(x))
        if grad.size > 0:
            g=self.jac(x)
            grad[:]=asarray(g[np.newaxis,:])

            #grad=asarray(grad)
            #result=asarray(result)
            #grad[:]=approx_fprime(x,self.vec_mag_fun_core,e)
        return result

    def vec_mag_one_fun_core(self,x):
        return np.bincount(self.filter_ind, weights=np.square(x))-1

    def set_filter_ind(self,AMA):
        self.filter_ind=AMA.filter_ind
        self.nPix=AMA.nPix
        self.Nf=AMA.Nf
        self.m=self.nPix*self.Nf

class ama_opts:
    def __init__(self,stimFname,bStimNorm=0,btchSz=100, NfFix=0, NfFset=4, errType=2, normType=0, bMean=1, alpha=1.3600, s0=0.230, rmax=5.70, bRectify=0,bPrint=2,bJax=1):
        self.stimFname=stimFname
        self.bStimNorm=int(bStimNorm)
        self.btchSz=int(btchSz)
        self.NfFix=int(NfFix)
        self.NfFset=int(NfFset)
        self.bPrint=bPrint

        self.errType=int(errType)
        self.normType=int(normType)
        self.bMean=int(bMean)
        self.bRectify=int(bRectify)

        self.alpha=alpha
        self.s0=s0
        self.rmax=rmax

        self.Nf=int(self.NfFset+self.NfFix)

        self.bJax=bJax
