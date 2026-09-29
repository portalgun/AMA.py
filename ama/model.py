"""The likelihood models (Model)."""
from ._base import *

_FULL_CHUNK=2**24                            # full AMA: elements of the [ observed x reference x category ] terms per chunk


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
        # a zero mean variance (no noise) keeps its scale 1: sqrt(0)'s infinite derivative times a zero cotangent is NaN
        pos=v[None]>0
        sc=jnp.sqrt(jnp.where(pos,vL/jnp.where(pos,v[None],1),1.))
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
        Li=_tri_inv(L)
        solve=lambda v: Li@v                                                   # L^-1 v, v [ nCtg x nF x nStim_Ctg ]
        u,z=solve(d),solve(x)
        uu=jnp.sum(jnp.abs(u)**2,axis=1).T                                       # [ nStim_Ctg x nCtg ]
        zz=jnp.sum(jnp.abs(z)**2,axis=1).T
        uz=jnp.abs(jnp.sum(jnp.conj(u)*z,axis=1)).T**2
        den=1-a*uu
        q=zz+a*uz/den
        logdet=2*jnp.sum(jnp.log(jnp.real(jnp.diagonal(L,axis1=-2,axis2=-1))),axis=-1)[None]+jnp.log(den)
        return q,logdet

    @staticmethod
    def _loo_pooled_low_rank(m):
        # whether leave-one-out with statistics shared across categories (_loo_needs_all) can use _loo_quad_all
        return (Model._loo_needs_all(m) and m.covRank is None and not (m.covShrink>0 and m.covTarget=='diag')
                and not (m.ctgPoolWidth is not None and m.bPoolMeans))

    @staticmethod
    def _loo_quad_all(R,Rm,noiseCov,noiseL,weights,m,Y,bCentered=True):
        """
        q = x^H M^-1 x and log det M [ nStim_Ctg x nCtg(k) x nCtg(i) ] of every category's likelihood with stimulus (l,k)
        left out, when that changes the other categories' statistics (ctgPoolWidth without bPoolMeans, the pooled
        shrinkage target; not covRank). Category i's covariance is (1-s) P_i/Q_i + s T + noise_i, with P_i = sum_j K_ij C_j
        over the categories' own scatters, Q_i = sum_j K_ij dof_j, and T = sum_j C_j / sum_j dof_j. Leaving out a stimulus
        of weight 1 from category k downdates C_k by c d d^H (d = Rm - o_k, c = N_k/(N_k-1) about the mean) and dof_k by 1,
        so M = B_ik - a_ik d d^H with
            B_ik = (1-s) P_i/(Q_i-K_ik) + s sum_j C_j/(D-1) + noise_i,   a_ik = c_k ((1-s) K_ik/(Q_i-K_ik) + s/(D-1)),
        one matrix per pair of categories instead of one per stimulus and category; q and log det M follow from B's
        Cholesky factor as in _loo_quad. noiseL [ nStim_Ctg x nCtg x nF x nF ] (bLooNoise): the own category's noise
        covariance without the stimulus, which is not a low-rank change, so the own category's term is computed directly.
        Stim weights are 0 (padding, whose value is ignored) or 1.
        """
        nF,_,nC=Rm.shape
        noiseCov=jnp.broadcast_to(noiseCov,(nC,nF,nF))
        wc=jnp.sum(weights,axis=0)                                               # [ nCtg ]
        o=jnp.sum(Rm*weights,axis=1)/wc if bCentered else jnp.zeros((nF,nC),dtype=Rm.dtype)   # [ nF x nCtg ]
        Dv=(Rm-o[:,None,:])*jnp.sqrt(weights)
        C=jnp.einsum('isc,jsc->cij',Dv,jnp.conj(Dv))                             # [ nCtg x nF x nF ] own scatters
        dof=(wc-1) if bCentered else wc
        K=jnp.eye(nC,dtype=weights.dtype) if m.ctgPoolWidth is None else Model._ctg_kernel(Y,m.ctgPoolWidth,m._Yperiod).astype(weights.dtype)
        s=m.covShrink                                                            # the pooled target (see _loo_pooled_low_rank)
        QL=(K@dof)[:,None]-K                                                     # [ i x k ]
        DL=jnp.sum(dof)-1
        B=((1-s)*jnp.einsum('ij,jfg->ifg',K,C)[:,None]/QL[...,None,None] + s*jnp.sum(C,axis=0)/DL
           + noiseCov.astype(C.dtype)[:,None])                                   # [ i x k x nF x nF ]
        c=wc/(wc-1) if bCentered else jnp.ones_like(wc)
        a=c[None,:]*((1-s)*K/QL + s/DL)                                          # [ i x k ]

        d=jnp.transpose(Rm-o[:,None,:],(2,0,1))                                  # [ k x nF x l ]
        eye=jnp.eye(nC,dtype=weights.dtype)
        x=jnp.transpose(R,(2,0,1))[None]-o.T[:,None,:,None]                       # [ i x k x nF x l ]: R - o_i
        if bCentered:
            # the own category's mean without the stimulus: o_k - w d/(N_k-w)
            x=x+eye[:,:,None,None]*(d*(weights/(wc-weights)).T[:,None,:])[None]
        L=jnp.linalg.cholesky(B)
        Li=_tri_inv(L)
        solve=lambda v: Li@v
        u,z=solve(jnp.broadcast_to(d[None],x.shape)),solve(x)
        aw=a[...,None]*weights.T[None]                                           # [ i x k x l ]: 0 for padding
        uu=jnp.sum(jnp.abs(u)**2,axis=2)
        den=1-aw*uu
        q=jnp.sum(jnp.abs(z)**2,axis=2)+aw*jnp.abs(jnp.sum(jnp.conj(u)*z,axis=2))**2/den
        logdet=2*jnp.sum(jnp.log(jnp.real(jnp.diagonal(L,axis1=-2,axis2=-1))),axis=-1)[...,None]+jnp.log(den)
        q,logdet=jnp.transpose(q,(2,1,0)),jnp.transpose(logdet,(2,1,0))           # [ l x k x i ]
        if noiseL is not None:
            k=jnp.arange(nC)
            dl=jnp.transpose(d,(2,0,1))                                          # [ l x k x nF ]
            M=(B[k,k]-noiseCov.astype(C.dtype))[None] + noiseL.astype(C.dtype) \
              - aw[k,k].T[...,None,None]*dl[...,:,None]*jnp.conj(dl)[...,None,:]
            xo=jnp.transpose(x[k,k],(2,0,1))                                     # [ l x k x nF ]
            Lo=jnp.linalg.cholesky(M)
            y=_tri_apply(_tri_inv(Lo),xo)
            qo=jnp.sum(jnp.abs(y)**2,axis=-1)
            ldo=2*jnp.sum(jnp.log(jnp.real(jnp.diagonal(Lo,axis1=-2,axis2=-1))),axis=-1)
            q,logdet=Model._set_own(q,qo),Model._set_own(logdet,ldo)
        return q,logdet

    @staticmethod
    def _lpdf_quad(q,logdet,n,m):
        # log density from q = x^H M^-1 x and log det M of the covariance M (n dimensions; complex ones for 'circ')
        if m.modelType=='student':
            # scale (df-2)/df M: q scales by df/(df-2), log det by n log((df-2)/df)
            df=m.df
            return (jax.scipy.special.gammaln((df+n)/2)-jax.scipy.special.gammaln(df/2)-n/2*np.log(df*np.pi)
                    -(logdet+n*np.log((df-2)/df))/2-(df+n)/2*jnp.log1p(q/(df-2)))
        if m.modelType=='circ':
            return -q-n*np.log(np.pi)-logdet
        return -q/2-n/2*np.log(2*np.pi)-logdet/2

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
        if bLeaveOneOut and Model._loo_pooled_low_rank(m):
            q,logdet=Model._loo_quad_all(R,Rm,noiseCov,Model._noise_loo(noiseCov,RVar,weights) if m.bLooNoise else None,
                                         weights,m,Y)
            lAll=Model._add_own(Model._lpdf_quad(q,logdet,R.shape[0],m),Model._loo_prior(weights))
        elif bLeaveOneOut and Model._loo_needs_all(m):
            muL,covL=Model._ctg_stats_loo_all(Rm,weights,m,Y)
            noiseL=Model._noise_all(noiseCov,Model._noise_loo(noiseCov,RVar,weights,m))
            lAll=Model._add_own(lmvn0(jnp.transpose(R,(1,2,0))[:,:,None,:]-muL,covL+noiseL),Model._loo_prior(weights))
        elif bLeaveOneOut and Model._loo_rank_one(m):
            q,logdet=Model._loo_quad(R,Rm,noiseCov,weights)
            lAll=Model._set_own(lAll,Model._lpdf_quad(q,logdet,R.shape[0],m) + Model._loo_prior(weights))
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
        if bLeaveOneOut and Model._loo_pooled_low_rank(m):
            q,logdet=Model._loo_quad_all(R,Rm,noiseCov,Model._noise_loo(noiseCov,RVar,weights) if m.bLooNoise else None,
                                         weights,m,Y)
            lAll=Model._add_own(Model._lpdf_quad(q,logdet,R.shape[0],m),Model._loo_prior(weights))
        elif bLeaveOneOut and Model._loo_needs_all(m):
            muL,covL=Model._ctg_stats_loo_all(Rm,weights,m,Y)
            noiseL=Model._noise_all(noiseCov,Model._noise_loo(noiseCov,RVar,weights,m))
            lAll=Model._add_own(lmvt0(jnp.transpose(R,(1,2,0))[:,:,None,:]-muL,(covL+noiseL)*(m.df-2)/m.df,m.df),
                                Model._loo_prior(weights))
        elif bLeaveOneOut and Model._loo_rank_one(m):
            q,logdet=Model._loo_quad(R,Rm,noiseCov,weights)
            lAll=Model._set_own(lAll,Model._lpdf_quad(q,logdet,R.shape[0],m) + Model._loo_prior(weights))
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
        if bLeaveOneOut and Model._loo_pooled_low_rank(m):
            q,logdet=Model._loo_quad_all(Rc,Rmc,Nc,blocks(Model._noise_loo(noiseCov,RVar,weights)) if m.bLooNoise else None,
                                         weights,m,Y,bCentered=not bZero)
            lAll=Model._add_own(Model._lpdf_quad(q,logdet,h,m),Model._loo_prior(weights))
        elif bLeaveOneOut and Model._loo_needs_all(m):
            muL,covL=Model._ctg_stats_loo_all(Rmc,weights,m,Y,bCentered=not bZero)
            NcL=blocks(Model._noise_all(noiseCov,Model._noise_loo(noiseCov,RVar,weights,m)))
            lAll=Model._add_own(lcn0(jnp.transpose(Rc,(1,2,0))[:,:,None,:]-muL,covL+NcL.astype(covL.dtype)),Model._loo_prior(weights))
        elif bLeaveOneOut and Model._loo_rank_one(m):
            q,logdet=Model._loo_quad(Rc,Rmc,Nc,weights,bCentered=not bZero)
            lAll=Model._set_own(lAll,Model._lpdf_quad(q,logdet,h,m) + Model._loo_prior(weights))
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
            lAll=(Model._mix_loo_pooled_low_rank if Model._mix_loo_low_rank_ok(m) else Model._mix_loo_pooled)(
                x,Rm,noiseCov,RVar,weights,m,mu,det)
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
    def _mix_loo_low_rank_ok(m):
        # whether pooled 'mix' leave-one-out can use _mix_loo_pooled_low_rank
        return m.covRank is None and not (m.covShrink>0 and m.covTarget=='diag')

    @staticmethod
    def _mix_loo_pooled_low_rank(x,Rm,noiseCov,RVar,weights,m,mu,det):
        """
        _mix_loo_pooled without covRank or diagonal shrinkage. For the other categories (i != k), leaving out stimulus
        (l,k) of weight 1 changes only the pooled statistics: P_i by K_ik dS and Q_i by K_ik (and the pooled target), with
        dS = c_k dc dc^T the downdate of category k's scatter about its mean (c_k = N_k/(N_k-1)). Component c of category
        i then has the covariance M = B_ikc - a_ikc dc dc^T with
            B_ikc = (1-s) (S_ic + pi_ic P_i)/D_ikc + s sum_j Scat_j/(D-1) + ridge_i + noise_i,  D_ikc = den_ic + pi_ic (Q_i - K_ik)
            a_ikc = c_k ((1-s) pi_ic K_ik/D_ikc + s/(D-1))
        (an empty component keeps cov0_i, a = 0): one Cholesky factor per (i, k, c) instead of one per stimulus, as in
        _loo_quad_all. The own category (i = k), whose responsibility-weighted component statistics change per stimulus, is
        computed as in _mix_loo_pooled. Stim weights are 0 (padding, whose value is ignored) or 1.
        """
        pool=det['pool']
        nF=x.shape[-1]
        Nc,S,n=det['Nc'],det['S'],det['n']                                        # [ i x c ], [ i x c x nF x nF ], [ i ]
        live=Nc>1e-8
        den=jnp.where(live,jnp.maximum(Nc-1,Nc/2),1.)
        pi=jnp.where(live,Nc,0.)/n[:,None]
        K=pool['K']                                                              # [ i x k ], zero diagonal
        s=m.covShrink                                                            # the pooled target (see _mix_loo_low_rank_ok)
        Ssum=jnp.sum(pool['Scat'],axis=0)
        DL=jnp.sum(pool['dof'])-1
        Dn=den[:,None,:]+pi[:,None,:]*(pool['Q'][:,None]-K)[...,None]           # [ i x k x c ]
        eye=jnp.eye(nF,dtype=x.dtype)
        ridge=det['ridge'][:,None,None,None,None]*eye
        B=(1-s)*(S[:,None]+pi[:,None,:,None,None]*pool['P'][:,None,None])/Dn[...,None,None] + s*Ssum/DL
        B=jnp.where(live[:,None,:,None,None],B,det['cov0'][:,None,None]) + ridge + noiseCov[:,None,None]   # [ i x k x c x nF x nF ]
        c=n/(n-1)
        a=jnp.where(live[:,None,:],c[None,:,None]*((1-s)*pi[:,None,:]*K[...,None]/Dn + s/DL),0.)         # [ i x k x c ]
        L=jnp.linalg.cholesky(B)
        Li=_tri_inv(L)
        o=(jnp.sum(Rm*weights,axis=1)/n).T                                        # [ k x nF ] category means
        Rt=jnp.transpose(Rm,(1,2,0))                                              # [ l x k x nF ]
        dc=Rt-o[None]
        u=jnp.einsum('ikcfg,lkg->ikcfl',Li,dc)
        z=jnp.einsum('ikcfg,lkg->ikcfl',Li,x)-jnp.einsum('ikcfg,icg->ikcf',Li,mu)[...,None]
        aw=a[...,None]*weights.T[None,:,None,:]                                   # [ i x k x c x l ]: 0 for padding
        dd=1-aw*jnp.sum(u**2,axis=3)
        q=jnp.sum(z**2,axis=3)+aw*jnp.sum(u*z,axis=3)**2/dd
        logdet=2*jnp.sum(jnp.log(jnp.diagonal(L,axis1=-2,axis2=-1)),axis=-1)[...,None]+jnp.log(dd)
        lc=-q/2-nF/2*np.log(2*np.pi)-logdet/2 + Model._logpi(pi)[:,None,:,None]
        lAll=jnp.transpose(logsumexp(lc,axis=2),(2,1,0))                          # [ l x k x i ]

        # the own category: its components without the stimulus's responsibility-weighted share (as _mix_loo_pooled)
        w=weights
        ao=jnp.transpose(det['r'],(1,0,2))                                        # [ l x k x c ]
        d=Rt[:,:,None,:]-mu[None]                                                 # [ l x k x c x nF ]
        NL=Nc[None]-ao
        liveL=NL>1e-8
        NLs=jnp.where(liveL,NL,1.)
        muL=mu[None]-(ao/NLs)[...,None]*d
        SL=S[None]-(ao*Nc[None]/NLs)[...,None,None]*d[...,:,None]*d[...,None,:]
        piL=jnp.where(liveL,NL,0.)/(n-w)[...,None]
        T=None
        if s>0:
            dS=(w*n/(n-w))[...,None,None]*dc[...,:,None]*dc[...,None,:]         # [ l x k x nF x nF ]
            T=((Ssum[None,None]-dS)/(jnp.sum(pool['dof'])-w)[...,None,None])[:,:,None]
        ridgeo=det['ridge'][None,:,None,None,None]*eye
        covL=Model._mix_cov(SL,jnp.where(liveL,jnp.maximum(NL-1,NL/2),1.),piL,pool,m,pool['P'][None],pool['Q'][None],T) + ridgeo
        covL=jnp.where(liveL[...,None,None],covL,det['cov0'][None,:,None]+ridgeo)
        noiseL=Model._noise_loo(noiseCov,RVar,weights,m)                          # [ l x k x nF x nF ]
        own=logsumexp(lmvn0(x[:,:,None,:]-muL,covL+noiseL[:,:,None]) + Model._logpi(piL),axis=-1)
        return Model._add_own(Model._set_own(lAll,own),Model._loo_prior(weights))

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


__all__=['Model']
