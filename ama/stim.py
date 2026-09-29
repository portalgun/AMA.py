"""Stimuli (Stim), filter indexing (_Index) and filters (Filter)."""
from ._base import *


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
        if Y.ndim==2 and (Y.shape[1]==1 or (Y.shape[0]==1 and self.nCtg>1)):
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
            if len(i)<2:
                raise Exception('train_test needs at least 2 stimuli in every category (one for each part)')
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


__all__=['Stim', '_Index', 'Filter']
