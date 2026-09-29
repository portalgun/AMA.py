"""
The parts of Filter (github.com/portalgun/Filter.py) that ama uses: the stimulus sample grid X and plotFT.
Shipped with ama so it installs without the full package. X keeps Filter's constructor and coordinate attributes, so
code written against Filter.X keeps working.
"""
import numpy as np


class _Dim:
    """one dimension of the sample grid: n samples spanning totS (space), centered at sample ceil((n-1)/2)"""
    def __init__(self,n=None,totS=None,totF=None,smpS=None,smpF=None,dS=None,dF=None,bCtrS=True,bCtrF=True):
        self.bCtrS=bCtrS
        self.bCtrF=bCtrF
        if totS and (smpS or totF) and not n:
            n=int(totS*(totF if totF else smpS))
        elif totS is None and n:
            if smpS is not None:
                totS=n/smpS
            elif dS is not None:
                totS=n*dS
            elif totF is not None:
                totS=n/totF
            elif smpF is not None:
                totS=smpF
            elif dF is not None:
                totS=1/dF
        self.totS=totS if totS else 1
        self.n=int(n) if n else 1000

    # sampling rate in space is the extent in frequency, and the reverse
    totF=property(lambda self: self.n/self.totS)
    smpS=property(lambda self: self.totF)
    smpF=property(lambda self: self.totS)
    dS=property(lambda self: self.totS/self.n)
    dF=property(lambda self: self.smpS/self.n)
    ctrS=property(lambda self: np.ceil((self.n-1)/2)*int(self.bCtrS))
    ctrF=property(lambda self: np.ceil((self.n-1)/2)*int(self.bCtrF))

    @property
    def s(self):
        c=self.ctrS
        return np.linspace(-c*self.dS,(self.n-c-1)*self.dS,self.n)

    @property
    def f(self):
        c=self.ctrF
        return np.linspace(-c*self.dF,(self.n-c-1)*self.dF,self.n)


class X:
    """
    sample grid of a stimulus: X(ndim=1,n=64,totS=1) or X(n=(8,9),totS=1). Each argument is one value for every
    dimension or one per dimension. s and f are meshgrids (xy indexing) of the space and frequency coordinates, sl and
    fl the per-dimension coordinate vectors, lims/limf their end points; other attributes (n, totS, dS, ...) are the
    value of the single dimension, or a tuple over dimensions
    """
    def __init__(self,ndim=None,**kwargs):
        if ndim is None:
            lens=[len(v) for v in kwargs.values() if isinstance(v,(list,tuple))]
            ndim=max(lens) if lens else 1
        per={}
        for key,value in kwargs.items():
            value=list(value) if isinstance(value,(list,tuple)) else [value]
            if len(value)==1:
                value=value*ndim
            elif len(value)!=ndim:
                raise Exception(key + ' should have a length of 1 or ' + str(ndim) + '. has ' + str(len(value)))
            per[key]=value
        self._dims=[_Dim(**{k: v[i] for k,v in per.items()}) for i in range(ndim)]

    ndim=property(lambda self: len(self._dims))
    shape=property(lambda self: self.n)
    s=property(lambda self: np.meshgrid(*[d.s for d in self._dims],indexing='xy'))
    f=property(lambda self: np.meshgrid(*[d.f for d in self._dims],indexing='xy'))
    sl=property(lambda self: tuple(d.s for d in self._dims))
    fl=property(lambda self: tuple(d.f for d in self._dims))
    lims=property(lambda self: [d.s[np.array([0,-1])] for d in self._dims])
    limf=property(lambda self: [d.f[np.array([0,-1])] for d in self._dims])
    extents=property(lambda self: np.concatenate(self.lims))
    extentf=property(lambda self: np.concatenate(self.limf))

    def __getattr__(self,attr):
        if attr.startswith('__') or attr=='_dims':
            raise AttributeError(attr)
        if len(self._dims)==1:
            return getattr(self._dims[0],attr)
        return tuple(getattr(d,attr) for d in self._dims)

    def __getitem__(self,ind):
        return self._dims[ind]


def plotFT(x,y,colorR='blue',colorI='red',colorA='black',linestyleR='-',linestyleI='-',linestyleA=':',**kwargs):
    """plot y over x: the real part, and for complex y also the imaginary part and the magnitude"""
    import matplotlib.pyplot as plt
    if np.any(np.imag(y)!=0):
        plt.plot(x,np.real(y),color=colorR,linestyle=linestyleR,**kwargs)
        plt.plot(x,np.imag(y),color=colorI,linestyle=linestyleI,**kwargs)
        plt.plot(x,np.abs(y),color=colorA,linestyle=linestyleA,**kwargs)
    else:
        plt.plot(x,np.real(y),color=colorR,linestyle=linestyleR,**kwargs)
