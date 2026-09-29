"""The response model (Nrn)."""
from ._base import *
from .stim import Filter


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
    rho correlates the noise of the response dimensions: a number correlates all of them (filters, sub-filters,
    real/imaginary components) equally; a correlation matrix [ nDim x nDim ] specifies each pair, over the flattened
    response dimensions in the order of _flatten_responses (real parts of all filters, sub-filters within each filter,
    then the imaginary parts), in noise sampling and in the likelihood alike. None: no noise.
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
                 whitenType='None',whitenMethod='zca',whitenEps=1e-5,readoutType='None',bBias=False,bSplitNegatives=False,
                 rsat=None,c50=None,nNR=2.,softmaxT=1.,normPool=None,bLearnNormPool=False,nReadout=None,readoutActivation='None',
                 respBudget=None):
        """
        activationType - response nonlinearity, on real and imaginary components separately (responses are in units of
                       rmax): 'relu', 'softplus', 'abs', 'logistic', 'swish', 'swish2', 'tanh', 'gauss', 'igauss', and
                       'ramp'        - min(max(r,0), rsat): rectified, saturating at rsat (default rmax/2)
                       'nakarushton' - rmax r^nNR / (r^nNR + c50^nNR) for r > 0 (Naka-Rushton / hyperbolic ratio; c50
                                       default rmax/4)
                       'softmax'     - rmax softmax(r/softmaxT) over the filters (and sub-filters): a population
                                       nonlinearity whose responses sum to rmax (real responses only)
        bBias        - learn an offset per response dimension, added to the linear response before the activation (it
                       matters with an activation, normalization, or through the response-dependent noise variance)
        bSplitNegatives - ON/OFF channels: each response dimension r becomes two rectified ones, max(r,0) and max(-r,0)
                       (on real and imaginary components separately), after the activation: firing rates are not
                       negative, and each channel has its own noise (fano*|r| + var0)
        normPool     - normalizeType 'gen': [ nChan x nChan ] weights of the normalization pool, D_i = eps + sum_j
                       normPool_ij |r_j| (summed over sub-filters and components), with nChan the response dimensions
                       entering normalization (filters, x2 with bSplitNegatives; flattened dimensions when whitened).
                       None: every channel normalizes every other (the plain 'gen' normalization)
        bLearnNormPool - learn the normalization pool weights (nonnegative, softplus of Unit parameters, starting at
                       normPool or at all ones). Only their relative sizes are learned: each row keeps the total weight
                       of the initial pool (with fixed additive noise, a smaller denominator would otherwise always help)
        readoutType 'linear' - a second layer: nReadout output neurons (default: the number of filters), each a learned
                       unit-norm linear combination of all the (normalized) responses, on real and imaginary components
                       separately, followed by readoutActivation (any activationType) and stage-2 noise. With bBias, the
                       output neurons have learned offsets too.
                       nReadout as a tuple of widths, e.g. (8, 4), stacks that many layers, each a unit-norm linear
                       combination of the previous layer followed by readoutActivation (with bBias, offsets before it);
                       stage-2 noise is on the last. Without an activation, the stack is one linear map.
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
        respBudget   - a response budget: each final response dimension (after the readout, before stage-2 noise) is
                       scaled so that its root mean square over the training stimuli is respBudget. Its signal-to-noise
                       ratio against the fixed additive noise (var0) then can not grow with the size of the responses,
                       only with their shape: learned normalization pools (bLearnNormPool) and second layers can not
                       shrink denominators or grow weights to beat the noise. Stage-1 noise carried through is scaled
                       likewise. Held-out stimuli are scaled by the training stimuli's gains (as whitening; see
                       Unit.evaluate); with Optimizer batchSize, by each batch's.
        readoutType  - pooling of the (normalized) responses across filters, with learned nonnegative weights p
                       (softplus of the parameters Unit.pool_p, normalized to sum 1), before stage-2 noise:
                       'None'
                       'resultant'      - the filter responses and their weighted resultant sum_j p_j R_j
                       'resultant_only' - only the weighted resultant (weighted phase congruency with 'phase')
        """
        # rmax=5.7, var0=0.23 as in burgelab/AMA (paramRSP of AMAdataDisparity.mat)
        # python floats: numpy scalars (e.g. read from a .mat file) would promote float32 learning to float64 under jax x64
        self.fano=float(fano)
        self.var0=float(var0)
        self.rmax=float(rmax)
        self.eps=float(eps)

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
        self.bBias=bool(bBias)
        self.bSplitNegatives=bool(bSplitNegatives)
        self.rsat=None if rsat is None else float(rsat)
        self.c50=None if c50 is None else float(c50)
        self.nNR=float(nNR)
        self.softmaxT=float(softmaxT)
        self.normPool=None if normPool is None else np.array(normPool,dtype=float)
        self.bLearnNormPool=bool(bLearnNormPool)
        self.nReadout=None if nReadout is None else (int(nReadout) if np.ndim(nReadout)==0 else tuple(int(n) for n in nReadout))
        self.readoutActivation=readoutActivation
        self.respBudget=None if respBudget is None else float(respBudget)
        if self.respBudget is not None and not self.respBudget>0:
            raise Exception('respBudget must be positive (or None)')
        if not hasattr(Nrn,'_activation__'+str(readoutActivation).lower()):
            raise Exception('readoutActivation must be an activationType (e.g. None, relu, ramp, softmax)')
        if self.normPool is not None and (self.normPool.ndim!=2 or self.normPool.shape[0]!=self.normPool.shape[1]
                                          or np.any(self.normPool<0)):
            raise Exception('normPool must be a square matrix of nonnegative weights')
        if (self.rsat is not None and not self.rsat>0) or (self.c50 is not None and not self.c50>0) or not self.nNR>0 \
                or not self.softmaxT>0:
            raise Exception('rsat, c50, nNR and softmaxT must be positive')
        if self.nReadout is not None and (np.min(self.nReadout)<1 or np.size(self.nReadout)<1):
            raise Exception('nReadout must be at least 1 (or a tuple of layer widths of at least 1)')

        self.rho=rho

        self.filter=Filter()
        self.bFinalized=False

    @property
    def bAnalytic(self):
        if hasattr(self,'filter') and hasattr(self.filter,'index') and hasattr(self.filter.index,'bAnalytic'):
            return self.filter.index.bAnalytic
        else:
            return None

    def _key(self):
        return (self.fano,self.var0,self.rmax,self.eps,self.nSamples,_freeze(self.rho),
                self.bNoise_1,self.bNoise_2,self.activationType,self.normalizeType,self.corrType,self.averageType,
                self.whitenType,self.whitenMethod,self.whitenEps,self.readoutType,getattr(self.filter,'pix_dims',None),
                getattr(self,'bFourier',None),getattr(self,'bSplit',None),self.bAnalytic,str(getattr(self,'dtype',None)),
                self.bBias,self.bSplitNegatives,self.rsat,self.c50,self.nNR,self.softmaxT,_freeze(self.normPool),
                self.bLearnNormPool,self.nReadout,str(self.readoutActivation).lower(),getattr(self.filter,'n',None),self.respBudget)

    def copy(self):
       return Nrn(**_get_copy_dict(self,_NRN_EXCL))

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

    @property
    def bPooledReadout(self):
        return str(self.readoutType).lower() in ('resultant','resultant_only')

    @property
    def bParams(self):
        # whether the response model has learned parameters (Unit._p)
        return self.bReadout or self.bBias or self.bLearnNormPool

    #- response dimensions (after finalize)
    @property
    def bComplexOut(self):
        # complex responses (quadrature pairs, not whitened)
        return bool(self.bAnalytic) and not self._bWhiten

    @property
    def nIn(self):
        # rows (axis 0) of the responses entering the activation: filters, or flattened dimensions when whitened
        n=self.filter.n
        if self._bWhiten:
            return n*(2 if self.bAnalytic else 1)*(self.filter.nSplit if self.bSplit else 1)
        return n

    @property
    def nChan(self):
        # rows entering normalization and the readout
        return self.nIn*(2 if self.bSplitNegatives else 1)

    @property
    def nSub(self):
        # sub-filters per row (split parts, not whitened)
        return self.filter.nSplit if (self.bSplit and not self._bWhiten) else 1

    @property
    def _readoutWidths(self):
        # the widths of the layers of the readout 'linear'
        if self.nReadout is None:
            return (self.nChan,)
        return self.nReadout if isinstance(self.nReadout,tuple) else (self.nReadout,)

    @property
    def nOut(self):
        # the number of output neurons of the readout 'linear'
        return self._readoutWidths[-1]

    @property
    def nDimOut(self):
        # flattened real response dimensions of the final responses (see _flatten_responses)
        comp=2 if self.bComplexOut else 1
        rt=str(self.readoutType).lower()
        if rt=='linear':
            return self.nOut*comp
        rows={'resultant':self.nChan+1,'resultant_only':1}.get(rt,self.nChan)
        return rows*comp*self.nSub

    def params0(self,old=None,seed=0):
        """
        initial learned response-model parameters (a dict; see Unit._p), keeping the values in old whose shapes match:
            'pool'  [ nChan ]            pooling weights (readoutType 'resultant', 'resultant_only')
            'bias'  [ nComp x nIn (x nSplit) ] offsets before the activation (bBias)
            'norm'  [ nChan x nChan ]    normalization pool (bLearnNormPool)
            'A'     [ n1 x nChan*nSub ] second-layer weights (readoutType 'linear', n1 the first of nReadout's widths),
                    'bias2' [ nComp x n1 ] its offsets; 'A_l' [ n(l+1) x n(l) ] and 'bias2_l' further layers (l = 1, ...)
        """
        old=old or {}
        rd=np.dtype(jnp.finfo(self.dtype).dtype)
        comp=2 if self.bComplexOut else 1
        out={}
        if self.bPooledReadout:
            out['pool']=np.zeros(self.nChan)
        if self.bBias:
            out['bias']=np.zeros((comp,self.nIn)+((self.filter.nSplit,) if (self.bSplit and not self._bWhiten) else ()))
        if self.bLearnNormPool:
            M=np.ones((self.nChan,self.nChan)) if self.normPool is None else np.asarray(self.normPool)
            out['norm']=np.log(np.expm1(np.maximum(M,1e-6)*np.log(2.)))       # softplus^-1(M log 2)
        if str(self.readoutType).lower()=='linear':
            nI=self.nChan*self.nSub
            rng=np.random.default_rng(seed)
            for l,nO in enumerate(self._readoutWidths):
                sfx='' if l==0 else '_'+str(l)
                out['A'+sfx]=np.eye(nO,nI)+0.1*rng.standard_normal((nO,nI))/np.sqrt(nI)
                if self.bBias:
                    out['bias2'+sfx]=np.zeros((comp,nO))
                nI=nO
        for k,v in list(out.items()):
            o=old.get(k)
            if o is not None and np.shape(o)==v.shape:
                v=np.asarray(o)
            elif o is not None and k=='pool':
                # new filters: pooling weights extended with zeros (equal weights)
                v=v.copy(); o=np.asarray(o).ravel()[:len(v)]; v[:len(o)]=o
            elif o is not None and k=='bias' and np.ndim(o)==v.ndim and np.shape(o)[::2]==v.shape[::2]:
                # new filters (train_append): previous offsets for the previous filters, 0 for the new ones
                v=v.copy(); m=min(v.shape[1],np.shape(o)[1]); v[:,:m]=np.asarray(o)[:,:m]
            out[k]=jnp.asarray(v,dtype=rd)
        return out

    @partial(jit, static_argnames=['self','bGain'])
    def main(self,rng,stim,f,weights=None,W=None,p=None,G=None,bGain=False):
        """
        returns r, rNs, R, RNs, RVar
            r    mean response before normalization
            rNs  noisy response before normalization
            R    mean response (after normalization)
            RNs  noisy response (after normalization)
            RVar noise variance of the final response given its mean, fano*|R| + var0
        with whitening, responses are flattened real dimensions [ nDim x nStim_Ctg x nCtg ] (see _flatten_responses).
        weights [ nStim_Ctg x nCtg ] mark the valid stimuli (for whitenType='response' and respBudget); W is a frozen
        whitening matrix; p the learned response-model parameters (a dict, see Nrn.params0; an array is the pooling
        weights); G the respBudget gains to use (default: those of these stimuli). bGain: return the gains instead
        """
        rng_key1,rng_key2 = jxrandom.split(rng)
        if p is not None and not isinstance(p,dict):
            p={'pool':p}                                                         # pooling weights alone (older callers)

        # respond
        # fourier-domain filters hold half of the frequencies (see _Index.pix). With the orthonormal transform in Stim,
        # the real part of the response is that of the real spatial filter with this spectrum, and the imaginary part is
        # that of its quadrature (hilbert) pair. sqrt(2) gives each of those real filters unit norm when ||f||=1.
        #   fourierType=1: real part only (a real filter)      fourierType=2: both parts (a quadrature pair)
        f=self._scale(f)
        r=self._respond(f,stim)
        # population whitening of the mean responses, before noise
        r=self._whiten_fun(r,f,weights,W)
        r=self._add_bias(r,None if p is None else p.get('bias'))
        r=self._activation_fun(r,self.bFourier,self)
        r=self._split_negatives(r)

        # noisey output 1
        rNs = self._average_fun(self._noise_1_fun(r,self.fano,self.var0,self.nSamples,rng_key1,self.rho))

        # normalize
        M=self._norm_pool(p)
        R   = self._normalize_fun(r,  f,stim,self.eps,self.bSplit,M)
        RN  = self._normalize_fun(rNs,f,stim,self.eps,self.bSplit,M)

        # readout across filters (pooled resultant, or a second layer)
        R   = self._readout_fun(R,p,self)
        RN  = self._readout_fun(RN,p,self)

        # response budget
        if self.respBudget is not None:
            G=self.budget_gain(R,weights) if G is None else G
            if bGain:
                return G
            R,RN=R*G,RN*G

        # noisey output 2
        RNs = self._average_fun(self._noise_2_fun(RN,self.fano,self.var0,self.nSamples,rng_key2,self.rho))

        return r,rNs,R,RNs,self._likelihood_variance(r,R,f,stim,M,G if self.respBudget is not None else None)

    def budget_gain(self,R,weights=None):
        # respBudget / the root mean square of each response dimension over the valid stimuli [ ... x 1 x 1 ]
        if weights is None:
            weights=jnp.ones(R.shape[-2:],dtype=jnp.real(R).dtype)
        ms=jnp.sum(jnp.abs(R)**2*weights,axis=(-2,-1),keepdims=True)/jnp.sum(weights)
        return self.respBudget*_safe_divide(1.,jnp.sqrt(ms))

    def gain(self,stim,f,weights=None,W=None,p=None):
        """the respBudget gains of these (training) stimuli, to decode other stimuli with (None without respBudget)"""
        if self.respBudget is None:
            return None
        if p is not None and not isinstance(p,dict):
            p={'pool':p}
        return self.main(jxrandom.key(0),stim,f,weights,W,p,bGain=True)

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self,rho):
        # the noise correlation type follows rho, also when rho is changed after construction
        if rho is not None and not isinstance(rho,str) and np.ndim(rho)>0:
            rho=np.array(rho,dtype=float)                                        # a correlation matrix
            if rho.ndim!=2 or rho.shape[0]!=rho.shape[1]:
                raise Exception('rho must be a number or a square correlation matrix')
            self._rho=rho
            self.corrType='corr'
            return
        self._rho=rho
        if rho is None or (isinstance(rho,str) and rho=='None'):
            self.corrType='None'
        elif rho==0:
            self.corrType='uncorr'
        else:
            self.corrType='corr'

    def _likelihood_variance(self,r,R,f,stim,M=None,G=None):
        # see the class docstring. r: responses before normalization, R: after (and after the respBudget gains G);
        # M: the 'gen' normalization pool
        if not self.bNoise_1:
            return self.variance(R,self.fano,self.var0)
        v1=self.variance(r,self.fano,self.var0)
        if str(self.normalizeType).lower()=='gen':
            # R_i = r_i/D_i, D_i = eps + sum_j M_ij |r_j| (j over channels, summed over sub-filters):
            # var(R_i) ~ sum_j (dR_i/dr_j)^2 v_j = v_i/D_i^2 - 2 M_ii |r_i| v_i/D_i^3 + r_i^2 sum_j M_ij^2 v_j/D_i^4  (real)
            D=Nrn._gen_denominator(r,self.eps,M)
            D=jnp.where(D>0,D,1)
            if M is None:
                axes=tuple(range(r.ndim-2))
                V=jnp.sum(v1,axis=axes,keepdims=True)
                Mii=1.
            else:
                V=Nrn._pool_rows(v1,M**2)
                Mii=jnp.diagonal(M).astype(jnp.real(r).dtype).reshape((-1,)+(1,)*(r.ndim-1))   # a float64 normPool stays out of float32 learning
            var=v1/D**2 - 2*Mii*jnp.abs(r)*v1/D**3 + r**2*V/D**4
        else:
            # linear normalizations ('broad', 'narrow') divide each response by a stimulus-dependent denominator: its gain
            # is the normalization of ones (also where an activation made the response exactly 0)
            g=self._normalize_fun(jnp.ones(r.shape,dtype=jnp.real(r).dtype),f,stim,self.eps,self.bSplit,M)
            var=v1*g**2
        if G is not None:
            var=var*G**2
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
    def _activation__relu(R,bComplex,nrn=None):
        return Nrn._componentwise(lambda x: jnp.maximum(x,0),R,bComplex)

    @staticmethod
    def _activation__softplus(R,bComplex,nrn=None):
        return Nrn._componentwise(lambda x: jnp.logaddexp(x,0),R,bComplex)

    @staticmethod
    def _activation__abs(R,bComplex,nrn=None):
        return Nrn._componentwise(jnp.abs,R,bComplex)

    @staticmethod
    def _activation__logistic(R,bComplex,nrn=None):
        return Nrn._componentwise(lambda x: 1/(1+jnp.exp(-x)),R,bComplex)

    @staticmethod
    def _activation__swish(R,bComplex,nrn=None):
        return Nrn._componentwise(lambda x: x/(1+jnp.exp(-x)),R,bComplex)

    @staticmethod
    def _activation__swish2(R,bComplex,nrn=None):
        return Nrn._componentwise(lambda x: x*((1+jnp.tanh(x))/2),R,bComplex)

    @staticmethod
    def _activation__tanh(R,bComplex,nrn=None):
        return Nrn._componentwise(jnp.tanh,R,bComplex)

    @staticmethod
    def _activation__gauss(R,bComplex,nrn=None):
        return Nrn._componentwise(lambda x: jnp.exp(-x**2),R,bComplex)

    @staticmethod
    def _activation__igauss(R,bComplex,nrn=None):
        return Nrn._componentwise(lambda x: 1-jnp.exp(-x**2),R,bComplex)

    @staticmethod
    def _activation__ramp(R,bComplex,nrn=None):
        rsat=nrn.rmax/2 if nrn.rsat is None else nrn.rsat
        return Nrn._componentwise(lambda x: jnp.clip(x,0,rsat),R,bComplex)

    @staticmethod
    def _activation__nakarushton(R,bComplex,nrn=None):
        c50=nrn.rmax/4 if nrn.c50 is None else nrn.c50
        def nr(x):
            # x**nNR only where x > 0: at 0 its derivative is infinite for nNR < 1, which times the 0 of the rectification is NaN
            pos=x>0
            xp=jnp.where(pos,jnp.where(pos,x,1)**nrn.nNR,0)
            return nrn.rmax*xp/(xp+c50**nrn.nNR)
        return Nrn._componentwise(nr,R,bComplex)

    @staticmethod
    def _activation__softmax(R,bComplex,nrn=None):
        # over all response dimensions of a stimulus (filters, sub-filters)
        if jnp.iscomplexobj(R):
            raise Exception("activation 'softmax' needs real responses")
        X=jnp.reshape(R,(-1,)+R.shape[-2:])
        return jnp.reshape(nrn.rmax*jax.nn.softmax(X/nrn.softmaxT,axis=0),R.shape)

    #- learned response-model parameters (see Unit._p): offsets, split, normalization pool
    @staticmethod
    def _add_bias(r,b):
        # b [ nComp x (response dims before the stimulus axes) ]: offsets of the real (and imaginary) components
        if b is None:
            return r
        b=b.astype(jnp.real(r).dtype).reshape(b.shape+(1,1))
        if jnp.iscomplexobj(r):
            return r + (b[0] + 1j*b[1])
        return r + b[0]

    def _split_negatives(self,r):
        if not self.bSplitNegatives:
            return r
        on=lambda x: jnp.maximum(x,0)
        if jnp.iscomplexobj(r):
            return jnp.concatenate((on(r.real)+1j*on(r.imag),on(-r.real)+1j*on(-r.imag)),axis=0)
        return jnp.concatenate((on(r),on(-r)),axis=0)

    def _norm_pool(self,p):
        # the 'gen' normalization pool [ nChan x nChan ], or None for all ones
        if str(self.normalizeType).lower()!='gen':
            return None
        if self.bLearnNormPool:
            # relative weights: each row keeps the total weight of the initial pool, or shrinking the denominator would
            # scale all responses up against the fixed additive noise (var0)
            W=jnp.logaddexp(p['norm'],0.)
            M0=np.ones((W.shape[0],W.shape[0])) if self.normPool is None else self.normPool
            return W*(jnp.asarray(M0.sum(1),dtype=W.dtype)/jnp.sum(W,axis=1))[:,None]
        return None if self.normPool is None else jnp.asarray(self.normPool)

    @staticmethod
    def _pool_rows(A,M):
        # sum_j M_ij sum_(other non-stimulus axes) A_j  ->  [ nChan x 1 ... x nStim_Ctg x nCtg ]
        a=jnp.sum(A.reshape(A.shape[:1]+(-1,)+A.shape[-2:]),axis=1)             # [ nChan x nStim_Ctg x nCtg ]
        out=jnp.einsum('ij,jsc->isc',M.astype(a.dtype),a)
        return out.reshape((A.shape[0],)+(1,)*(A.ndim-3)+A.shape[-2:])

    @staticmethod
    def _gen_denominator(R,eps,M):
        if M is None:
            axes=tuple(range(R.ndim-2))
            return eps + jnp.sum(jnp.abs(R),axis=axes,keepdims=True)
        return eps + Nrn._pool_rows(jnp.abs(R),M)

    #- normalize
    @staticmethod
    def _normalize__none(R,*_):
        return R

    @staticmethod
    def _normalize__gen(R,f,stim,eps,bSplit,M=None):
        # divisive normalization by the pooled population response (the pool M, or all channels)
        return _safe_divide(R,Nrn._gen_denominator(R,eps,M))

    @staticmethod
    def _normalize__broad(R,f,stim,eps,bSplit,M=None):
        # N_brd = ||s||_2 (stimulus contrast energy), one value per stimulus
        axes=tuple(range(stim.ndim-2))
        N=jnp.sqrt(jnp.sum(jnp.abs(stim)**2,axis=axes))       # [ nStim_Ctg x nCtg ]
        return _safe_divide(R,eps + N)

    @staticmethod
    def _normalize__narrow(R,f,stim,eps,bSplit,M=None):
        # N_nrw = A_s^T A_f (amplitude spectra), one value per filter and stimulus; stim and f are fourier-domain
        if bSplit:
            return _safe_divide(R,eps + jnp.einsum('psf,psnc->fsnc',jnp.abs(f),jnp.abs(stim)))
        else:
            return _safe_divide(R,eps + jnp.einsum('pf,pnc->fnc',jnp.abs(f),jnp.abs(stim)))

    @staticmethod
    def _normalize__phase(R,f,stim,eps,bSplit,M=None):
        # unit phasors (signs for real responses): contrast- and gain-invariant
        return R/(jnp.abs(R)+eps)

    #- pooled readout over filters (axis 0), weights p [ nF ] (unconstrained; softplus, normalized)
    @staticmethod
    def pool_weights(p):
        w=jnp.logaddexp(p,0.)
        return w/jnp.sum(w)

    @staticmethod
    def _readout__none(R,p,nrn=None):
        return R

    @staticmethod
    def _pool(R,p):
        # weighted sum over filters; real and imaginary parts separately, so the gradient of the real weights is real
        w=Nrn.pool_weights(p).astype(jnp.real(R).dtype)
        if jnp.iscomplexobj(R):
            return (jnp.tensordot(w,R.real,axes=(0,0)) + 1j*jnp.tensordot(w,R.imag,axes=(0,0)))[None]
        return jnp.tensordot(w,R,axes=(0,0))[None]

    @staticmethod
    def _readout__resultant(R,p,nrn=None):
        return jnp.concatenate((R,Nrn._pool(R,p['pool'])),axis=0)

    @staticmethod
    def _readout__resultant_only(R,p,nrn=None):
        return Nrn._pool(R,p['pool'])

    @staticmethod
    def readout_matrix(A):
        # unit-norm rows: the readout, like the filters, cannot raise its signal-to-noise ratio by its scale
        return A/jnp.sqrt(jnp.sum(A**2,axis=1,keepdims=True))

    @staticmethod
    def _readout__linear(R,p,nrn=None):
        # learned layers: [ nOut x nStim_Ctg x nCtg ] from all non-stimulus dimensions of R (see readoutType 'linear')
        out=jnp.reshape(R,(-1,)+R.shape[-2:])
        act=getattr(Nrn,'_activation__'+str(nrn.readoutActivation).lower())
        for l in range(len(nrn._readoutWidths)):
            sfx='' if l==0 else '_'+str(l)
            A=Nrn.readout_matrix(p['A'+sfx]).astype(jnp.real(out).dtype)
            mix=lambda x: jnp.einsum('oi,isc->osc',A,x)
            out=mix(out.real)+1j*mix(out.imag) if jnp.iscomplexobj(out) else mix(out)
            out=act(Nrn._add_bias(out,p.get('bias2'+sfx)),jnp.iscomplexobj(out),nrn)
        return out

    #- filters and responses
    def _scale(self,f):
        return f*_SQRT2 if self.bFourier else f

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
        if rho is not None and (np.ndim(rho)>0 or rho!=0):
            zf=jnp.reshape(z,(-1,)+z.shape[-3:])                                 # [ nDim' x nStim_Ctg x nCtg x nSamples ]
            L=jnp.linalg.cholesky(Nrn._corr_mat(rho,zf.shape[0],zf.dtype))
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
        return Nrn._corr_mat(self.rho,n,dtype)

    @staticmethod
    def _corr_mat(rho,n,dtype=float):
        # rho: one correlation for all pairs, or the correlation matrix itself
        if np.ndim(rho)>0:
            return jnp.asarray(rho,dtype=dtype)
        return rho + (1-rho)*jnp.eye(n,dtype=dtype)

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
        corrMat=Nrn._corr_mat(rho,RVar.shape[0],RVar.dtype)
        return sd[:,:,None]*sd[:,None,:]*corrMat[None]


__all__=['Nrn']
