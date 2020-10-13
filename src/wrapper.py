#/usr/bin/env python3
import time
import numpy as NP
import jax.numpy as np
#import cython
from scipy.optimize import minimize

from scipy.io       import loadmat
import matplotlib.pyplot as plot
from itertools import combinations as cb

from .AMA import AMA
from .nl_wrapper import nl_wrapper
from .opts import *

#@cython.boundscheck(False)
#@cython.wraparound(False)

class wrapper:
    def __init__(self,stimFname,consOpts=None,amaOpts=None,optOpts=None):

        if not consOpts:
            consOpts=cons_opts()

        if not amaOpts:
            amaOpts=ama_opts(stimFname)
        else:
            amaOpts.stimFname=stimFname

        if not optOpts:
            optOpts=opt_opts()


        S, X, ctgInd=load_stim(amaOpts.stimFname)


        if amaOpts.bStimNorm:
            S=self.contrast_normalize_stim(S) #TODO CHECK IF NEEDS

        S=self.reshape_S(S,ctgInd)

        ama=AMA(S,X,ctgInd,amaOpts,consOpts)
        if not optOpts.maxeval:
            optOpts.maxeval=ama.nPix * 800

        if optOpts.lib=='nlopt':
            self.OPT=nl_wrapper(ama,optOpts,consOpts)
        print('INITIALISED\n')

    def reshape_S(self,S,ctgInd):

        nStim=int(ctgInd.shape[0])
        Nlvl =int(NP.max(ctgInd)+1)
        Ni   =int(nStim/Nlvl) # num stim per lvl
        nPix =int(S.shape[0])

        Snew=NP.empty([Nlvl,Ni,nPix])
        S=NP.transpose(S)
        for i in NP.unique(ctgInd):
            Snew[i,:,:]=S[ctgInd==i,:]

        return Snew


    def optimize(self):
        print('OPTIMIZING\n')
        self.OPT.optimize()

##
    def plot_S(self,lvl,n):
        S=self.OPT.AMA.reshape_S()
        plot.imshow(S[lvl,n,:,:],cmap="gray")
        plot.show()

    def plot_Ac(self,lvl,n):
        #amplitude spectra
        Ac=self.OPT.AMA.Ac
        plot.imshow(Ac[lvl,n,:,:],cmap="gray")
        plot.colorbar()
        plot.show()

    def plot_f_raw(self):
        f =self.OPT.AMA.f
        Nf=self.OPT.AMA.Nf
        for i in range(Nf):
            plot.subplot(1,Nf,i+1)
            ff=f[:,i]
            plot.imshow(ff[:,NP.newaxis])

        plot.show()
        return 0

    def plot_f(self):
        f=self.OPT.AMA.reshape_f(self.OPT.fopt)
        for i in range(self.OPT.AMA.Nf):
            plot.subplot(1,self.OPT.AMA.Nf,i+1)
            plot.imshow(f[:,:,i], cmap="gray")

        plot.show()
        return 0

    def plot_r(self):
        self.plot_response(self.AMA.r)

    def plot_R(self):
        self.plot_response(self.AMA.R)

    def plot_response(self,rsp):
        IND=NP.array(list(cb(NP.arange(self.Nf),2)))
        N=IND.shape[0]
        colors = plot.cm.rainbow(NP.linspace(0, 1, self.AMA.Nlvl))

        fig,axes=plot.subplots(self.Nf,self.Nf)
        for i in range(N):
            a=IND[i,0]
            b=IND[i,1]

            for j in range(self.AMA.Nlvl):

                r=rsp[j,:,:]
                axes[a,b].scatter(r[:,a],r[:,b],marker='.',color=colors[j])

        plot.show()
        return 0


    def plot_mean_posteriors(self):
        #plot.imshow(self.AMA.PPstm[0,:,:])
        colors = plot.cm.rainbow(NP.linspace(0, 1, self.AMA.Nlvl))
        for i in range(self.AMA.Nlvl):
            plot.plot(NP.sum(self.AMA.PPstm[i,:,:],axis=0)/self.AMA.nStim,color=colors[i])

        plot.show()
        return 0


def load_filters(fname):
    f=load_filters_mat(fname)
    return f

def load_filters_mat(fname):
    Sx=loadmat(fname)
    f=Sx['f']
    return f

def flatten_filters(f):
    f=f.ravel()
    return f

def load_stim(fname):
    # TODO, handle other types
    # X      [Nlvl, 1]
    # ctgInd [Nstm]
    # S      [nPix, nStm]
    S,X,ctgInd=load_mat(fname)
    # dd
    return S, X, ctgInd

def load_mat(fname):
    Sx=loadmat(fname)
    X=Sx['X'][0,:]
    ctgInd=Sx['ctgInd'][:,0]-1
    S=Sx['s']

    return S, X, ctgInd


def contrast_normalize_stim(self,S):
    # TODO SWITCH FOR PREPROCESSED
    # mean
    #m=np.mean(S,axis=0)

    #S=(S-m)/m

    #N=np.sum(S,axis=0)

    #S=S/N
    return S
