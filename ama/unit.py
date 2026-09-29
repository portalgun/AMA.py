"""The user entry point (Unit), which wires Stim, Nrn, Model, Objective and Optimizer together, and Response."""
from ._base import *
from .stim import Stim, Filter
from .nrn import Nrn
from .model import Model
from .objective import Objective
from .optimizer import Optimizer


class Unit(_Static):
    def __init__(self,stim,nrn,model,objective,optimizer=None,seed=None,rng=None,rng_last=None,name=None):
        """name: a label for this unit, used as the title of its figures (e.g. 'mixed amplitude / natural / phase pooled')"""
        self.name=name
        self.nrn=nrn
        self.stim=stim
        self.stim_full=stim
        self.model=model
        self.objective=objective
        self.optimizer=optimizer
        self.opt_state=None
        self.restart_costs=None
        self.pool_p=None          # pooling-weight parameters for Nrn readoutType (see Nrn.pool_weights)
        self.nrn_p=None           # other learned response-model parameters (Nrn bBias, bLearnNormPool, readoutType 'linear')
        self.train_log=[]         # training calls, for config() / from_config()
        self._set_geometry()

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

    def _set_geometry(self):
        # the latent geometry (circular dimensions) of the stimuli, used by the likelihood pooling and the objective. A Model
        # or Objective shared with a unit of another geometry is copied, so that unit keeps its own
        per=getattr(self.stim_full,'Yperiod',None)
        grid=_latent_grid(self.stim_full.Y) if hasattr(self.stim_full,'Y') else None
        bCont=bool(getattr(self.stim_full,'bContinuous',False))
        for name in ('model','objective'):
            obj=getattr(self,name)
            if getattr(obj,'_bGeometrySet',False) and (obj._Yperiod!=per or (name=='objective' and (obj._Ygrid!=grid or
                                                                                                   obj._bContinuous!=bCont))):
                obj=obj.copy()
                setattr(self,name,obj)
            obj._Yperiod=per
            obj._bGeometrySet=True
        self.objective._Ygrid=grid
        self.objective._bContinuous=bCont

    def split(self,stimInd=None):

        nrn=self.nrn.copy()
        model=self.model.copy()
        objective=self.objective.copy()
        optimizer=None if self.optimizer is None else self.optimizer.copy()

        stim=self.stim if stimInd is None else self.stim._subset(stimInd)

        # the same key as this unit (not a new split of it), so splitting does not change what this unit does next
        unit=Unit(stim,nrn,model,objective,optimizer=optimizer,seed=self.seed,rng=self.rng,rng_last=self.rng_last)
        unit.pool_p=self.pool_p
        unit.nrn_p=self.nrn_p
        unit.multiscale_out=getattr(self,'multiscale_out',None)
        unit.param_out=getattr(self,'param_out',None)

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

        # a copy: the caller's Stim (which other units may share) keeps its domain and precision
        stim=copy.copy(self.stim_full)._finalize(dtype,None,bFourier,bSplit)
        self.stim=stim if stimInd is None else stim._subset(stimInd)
        self._stimInd=None if stimInd is None else np.atleast_1d(np.asarray(stimInd,dtype=int))
        # results of generated filters describe only the training call that made them
        self.multiscale_out=self.param_out=None
        self._set_geometry()

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
        self._check_budget()

    def _check_budget(self):
        # learned response scales against fixed additive noise: see Nrn respBudget
        nrn=self.nrn
        if nrn.respBudget is not None or nrn.corrType=='None':
            return
        learned=[name for name,b in (('bLearnNormPool',nrn.bLearnNormPool),
                                     ("readoutType='linear'",str(nrn.readoutType).lower()=='linear')) if b]
        if learned:
            warnings.warn(' and '.join(learned) + ' without Nrn respBudget: against the fixed additive noise (var0), the '
                          'learned weights can lower the cost by growing the responses rather than by coding better, so '
                          'costs are not comparable to fixed response models. Set respBudget (e.g. 1.) to fix each '
                          "response dimension's root mean square",stacklevel=5)

    @property
    def _nDim(self):
        # flattened real response dimensions
        return self.nrn.nDimOut

    def _loo_min_count(self):
        # stimuli per category for bLeaveOneOut: the left-out covariance needs a degree of freedom (a centered one two),
        # unless it is pooled over categories
        m=self.model
        return 2 if (m.modelType=='full' or m.ctgPoolWidth is not None or (m.modelType=='circ' and m.circMean=='zero')) else 3

    def _check_batches(self):
        if self.optimizer.batchSize is None or self.model.modelType not in ('gss','student','circ','mix'):
            return
        _,mask=self.optimizer._batch_plan(self.stim.weights)
        mMin=int(np.asarray(mask).sum(0).min())
        if self.model.modelType=='mix' and mMin<2*self.model.nMix:
            raise Exception("modelType='mix' needs at least 2*nMix stimuli per category in every batch; increase Optimizer nBatchMinCtg")
        if self.model.bLeaveOneOut and mMin<self._loo_min_count():
            raise Exception('bLeaveOneOut needs at least ' + str(self._loo_min_count()) + ' stimuli per category in every batch; '
                            'increase Optimizer nBatchMinCtg')
        if mMin < self._nDim+1 and self.model.covRank is None:
            warnings.warn('batches have as few as ' + str(mMin) + ' stimuli in a category, fewer than the ' + str(self._nDim+1)
                          + ' needed for a full-rank AMA-Gauss covariance of ' + str(self._nDim) + ' response dimensions; '
                          'increase batchSize or Optimizer nBatchMinCtg',stacklevel=3)

    def _check(self):
        counts=np.asarray(self.stim.weights).sum(0)
        if self.model.modelType in ('gss','student','circ') and np.any(counts<2) and self.model.ctgPoolWidth is None:
            raise Exception("modelType='" + self.model.modelType + "' needs at least 2 stimuli in every category (fewest: " + str(int(counts.min())) + ')')
        if self.model.nNoiseSamples>1 or self.model.bFixedNoise:
            if str(self.model.responseType).lower()!='basic':
                raise Exception("nNoiseSamples and bFixedNoise average over noisy observations: they need responseType='basic'")
            if not (self.nrn.bNoise_1 or self.nrn.bNoise_2):
                raise Exception('nNoiseSamples and bFixedNoise need sampled response noise (Nrn bNoise_1 or bNoise_2)')
        if self.model.modelType=='student' and not self.model.df>2:
            raise Exception("modelType='student' needs df > 2 (its covariance is matched to the category covariance)")
        if self.model.modelType=='circ' and self.nrn.bFinalized:
            if not self.nrn.bAnalytic:
                raise Exception("modelType='circ' needs complex quadrature-pair responses (fourierType=2)")
            if self.nrn._bWhiten:
                raise Exception("modelType='circ' can not be combined with whitening, which mixes real and imaginary components")
        if self.model.bLeaveOneOut is False and self.model.modelType not in ('gss','full','student','circ','mix'):
            raise Exception("modelType must be 'gss', 'full', 'student', 'circ', or 'mix'")
        if self.model.covRank is not None:
            if self.model.modelType not in ('gss','student','mix','circ'):
                raise Exception("covRank is used by modelType 'gss', 'student', 'mix' and 'circ'")
            nCov=self._nDim//2 if self.model.modelType=='circ' else self._nDim
            if self.nrn.bFinalized and self.model.covRank>=nCov:
                raise Exception('covRank (' + str(self.model.covRank) + ') must be below the number of ' +
                                ('complex ' if self.model.modelType=='circ' else '') + 'response dimensions (' + str(nCov) + ')')
        if self.model.nRef is not None and self.model.modelType!='full':
            raise Exception("nRef is only used by modelType='full'")
        if self.model.nNeighbors is not None:
            if self.model.modelType!='full':
                raise Exception("nNeighbors is only used by modelType='full'")
            if self.model.nRef is not None:
                raise Exception('nNeighbors and nRef are alternatives; set one of them')
        if self.model.bWithin and not getattr(self.stim,'bContinuous',False):
            raise Exception('bWithin estimates within categories from the stimuli\'s own latent values: give Stim y (e.g. Stim.binned)')
        if self.model.modelType=='full' and self.model.bLeaveOneOut and self.model.nRef is not None and self.model.nRef<2:
            raise Exception('bLeaveOneOut with nRef needs nRef of at least 2')
        if self.model.modelType=='mix':
            if self.model.bPoolMeans:
                raise Exception("modelType='mix' pools covariances over categories (ctgPoolWidth), not means (bPoolMeans)")
            if np.any(counts<2*self.model.nMix):
                raise Exception("modelType='mix' needs at least 2*nMix stimuli in every category (fewest: " + str(int(counts.min())) + ')')
        if self.model.modelType=='full' and self.nrn.corrType=='None':
            raise Exception("modelType='full' needs response noise; rho=None (no noise) is only supported with modelType='gss'")
        if self.nrn.corrType=='corr' and self.nrn.bFinalized:
            nDim=self._nDim
            rho=self.nrn.rho
            if np.ndim(rho)>0:
                if rho.shape!=(nDim,nDim):
                    raise Exception('the rho matrix must be ' + str(nDim) + ' x ' + str(nDim) + ' (the flattened response dimensions)')
                if not (np.allclose(rho,rho.T) and np.allclose(np.diag(rho),1) and np.linalg.eigvalsh(rho).min()>0):
                    raise Exception('the rho matrix must be a correlation matrix: symmetric, unit diagonal, positive definite')
            else:
                lo=-1/(nDim-1) if nDim>1 else -np.inf
                if not (lo < rho < 1):
                    raise Exception('rho must be in (' + str(lo) + ', 1) for ' + str(nDim) + ' response dimensions')
        if self.nrn._bWhiten:
            if str(self.nrn.whitenType).lower() not in ('gram','response'):
                raise Exception("whitenType must be 'None', 'gram', or 'response'")
            if self.nrn.whitenMethod not in ('zca','chol'):
                raise Exception("whitenMethod must be 'zca' or 'chol'")
            if self.nrn.normalizeType=='narrow':
                raise Exception("whitening can not be combined with normalizeType='narrow'")
            if str(self.nrn.whitenType).lower()=='response' and self.nrn.bFinalized:
                nDim=self._nDim
                if nDim >= counts.sum():
                    raise Exception("whitenType='response' needs more stimuli (" + str(int(counts.sum())) + ') than response dimensions (' + str(nDim) + ')')
        if str(self.nrn.readoutType).lower() not in ('none','resultant','resultant_only','linear'):
            raise Exception("readoutType must be 'None', 'resultant', 'resultant_only', or 'linear'")
        nrn=self.nrn
        if nrn.bSplitNegatives and str(nrn.normalizeType).lower()=='narrow':
            raise Exception("bSplitNegatives can not be combined with normalizeType='narrow' (one denominator per filter)")
        if nrn.bSplitNegatives and nrn.bPooledReadout:
            raise Exception("bSplitNegatives can not be combined with the pooled readouts ('resultant', 'resultant_only')")
        if (nrn.normPool is not None or nrn.bLearnNormPool) and str(nrn.normalizeType).lower()!='gen':
            raise Exception("normPool and bLearnNormPool apply to normalizeType='gen'")
        if nrn.bFinalized:
            if nrn.normPool is not None and str(nrn.normalizeType).lower()=='gen' and nrn.normPool.shape!=(nrn.nChan,nrn.nChan):
                raise Exception('normPool must be ' + str(nrn.nChan) + ' x ' + str(nrn.nChan) + ' (the response channels entering normalization)')
            if str(nrn.activationType).lower()=='softmax' and nrn.bComplexOut:
                raise Exception("activationType='softmax' needs real responses (not fourierType=2 without whitening)")
            if str(nrn.readoutActivation).lower()=='softmax' and nrn.bComplexOut:
                raise Exception("readoutActivation='softmax' needs real responses (not fourierType=2 without whitening)")
        if str(nrn.readoutType).lower()!='linear' and (nrn.nReadout is not None or str(nrn.readoutActivation).lower()!='none'):
            raise Exception("nReadout and readoutActivation are used by readoutType='linear'")
        if self.nrn.bReadout and self.nrn.bNoise_1:
            raise Exception('stage-1 noise (bNoise_1) is not modeled through a pooled readout')
        if self.nrn.bReadout and self.nrn._bWhiten:
            raise Exception('a pooled readout can not be combined with whitening')
        nDim,per=getattr(self.stim,'nDim',1),getattr(self.stim,'Yperiod',None)
        if self.objective.errType=='fisher' and nDim>1 and self.objective._Ygrid is None:
            raise Exception("errType='fisher' with several latent dimensions needs the levels (Y) on a cartesian grid")
        if per is not None:
            Yv=np.asarray(self.stim.Y).reshape(len(self.stim.Y),-1)
            for d,P in enumerate(per):
                if P is not None and np.ptp(Yv[:,d])>=P:
                    raise Exception('latent values of circular dimension ' + str(d) + ' must span less than its period (' + str(P) + ')')
        if self.objective.estType=='cmean' and nDim>1:
            raise Exception("estType='cmean' needs a single latent dimension; set Stim Yperiod and use 'mean'")
        if self.model.bLeaveOneOut:
            if self.objective.errType=='mle':
                raise Exception("bLeaveOneOut defines a leave-one-out posterior and can not be used with errType='mle'")
            nMin=self._loo_min_count()
            if np.any(counts<nMin):
                raise Exception('bLeaveOneOut with modelType=' + repr(self.model.modelType) + ' needs at least ' + str(nMin) + ' stimuli in every category')
        if self.nrn.bNoise_1 and str(self.nrn.normalizeType).lower()=='phase':
            raise Exception("stage-1 noise (bNoise_1) is not modeled through normalizeType='phase', which is not linear")
        if self.nrn.corrType!='None' and not self.nrn.var0>0:
            raise Exception('var0 must be positive: the noise variance fano*|r| + var0 is otherwise 0 for zero responses')
        if (self.nrn.bNoise_1 and str(self.nrn.normalizeType).lower()=='gen' and self.nrn.bFinalized
                and self.nrn.bAnalytic and not self.nrn._bWhiten):
            raise Exception("stage-1 noise (bNoise_1) with normalizeType='gen' is only modeled for real responses (not fourierType=2)")

#- LEARN MODES
    def _p_old(self):
        old=dict(getattr(self,'nrn_p',None) or {})
        if self.pool_p is not None:
            old['pool']=self.pool_p
        return old

    def _p0(self):
        # initial learned response-model parameters (Nrn.params0): previous values where they fit; pooling weights extended
        # with zeros (equal weights) for new filters
        return self.nrn.params0(self._p_old(),seed=self.seed)

    def _set_p(self,p):
        # store learned response-model parameters (a dict) in pool_p and nrn_p
        if p is None:
            return
        p={k:np.asarray(v) for k,v in p.items()}
        self.pool_p=p.pop('pool',None)
        self.nrn_p=p or None

    def _val_fun(self,stimVal):
        test=self._prepare_stim(stimVal)
        def fun(params):
            full={'f':self.filter.insert(params['f'])}
            if 'p' in params:
                full['p']=params['p']
            return self._loss_fun_heldout(full,self.rng,test.val,test.weights,test.yCtg,test.Y,self.stim.val,self.stim.weights,
                                          self.stim.yCtg)
        return fun

    def _opt_key(self,shape):
        # what an optimizer state belongs to: the learned filter columns, their parameter shape, the optimizer, the precision
        ind=self.filter.index
        cols=tuple(int(c) for c in np.sort(np.concatenate((ind.ind_lrn,ind.ind_rec))))
        return (tuple(shape),cols,str(self.optimizer.optimizerType),np.dtype(self.nrn.dtype).name)

    def _run(self,f0,rng,opt_state=None,stimVal=None):
        self._check_batches()
        extra={'p':self._p0()} if self.nrn.bParams else None
        val_fun=None if stimVal is None else self._val_fun(stimVal)
        nbr=self._bNeighbors()
        self.out_params,self.opt_state,self.rng_last=self.optimizer.minimize(f0,rng,self.stim,self.filter,_BoundLoss(self,'_loss_fun_lrn'),
                                                                             opt_state=opt_state,extra_params=extra,val_fun=val_fun,
                                                                             mstate0=None if nbr else self._mix_state0(f0),
                                                                             state_fun=_BoundLoss(self,'_nbr_state') if nbr else None)
        self._opt_param_shape=self.out_params['f'].shape
        self._opt_state_key=self._opt_key(self._opt_param_shape)
        self.filter.extract(self.out_params['f'])
        if 'p' in self.out_params:
            self._set_p(self.out_params['p'])

    def _train_random(self,rng,nRestarts,stimVal=None):
        # learn from random initial filters; with restarts, keep the run with the lowest cost on the validation stimuli
        # (stimVal) if given, else on the training stimuli. restart_costs holds those costs
        if nRestarts<1:
            raise Exception('nRestarts must be at least 1')
        self.restart_costs=[]
        best=None
        for i in range(nRestarts):
            if i>0:
                rng=jxrandom.fold_in(rng,i)
            rng,rng_key = jxrandom.split(rng)
            f0=self.filter.get_f0(rng_key,self.optimizer._f0_jxrand_fun)
            self._run(f0,rng,stimVal=stimVal)
            if nRestarts==1:
                self.restart_costs=None
                return
            cost=float(self.loss) if stimVal is None else self.evaluate(stimVal)
            self.restart_costs.append(cost)
            if best is None or cost<best[0]:
                best=(cost,jnp.asarray(self.filter.out),self.out_params,self.opt_state,self.rng_last,
                      list(self.optimizer.loss_hist),list(self.optimizer.val_hist),self.optimizer.best_step,
                      self._opt_param_shape,self._opt_state_key,self.pool_p,self.nrn_p)
        (_,self.filter.out,self.out_params,self.opt_state,self.rng_last,self.optimizer.loss_hist,self.optimizer.val_hist,
         self.optimizer.best_step,self._opt_param_shape,self._opt_state_key,self.pool_p,self.nrn_p)=best

    @_logged_training
    def train_new(self,n,fourierType=None,bSplit=None,stimInd=None,dtype=None,optimizer=None,nRestarts=1,stimVal=None):
        """learn n new filters. stimVal: validation stimuli for early stopping (Optimizer patience)"""
        if optimizer is not None:
            self.optimizer=optimizer
        self.pool_p=self.nrn_p=None

        self._finalize(n,
                      np.arange(n),
                      (),
                      (),
                      fourierType=fourierType,
                      stimInd=stimInd,
                      dtype=dtype,
                      bSplit=bSplit
        )

        self._train_random(self.rng,nRestarts,stimVal)

    @_logged_training
    def train_recurse(self,ind_rec=None,fourierType=None,bSplit=None,stimInd=None,dtype=None,optimizer=None,stimVal=None):
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
        opt_state=self.opt_state if getattr(self,'_opt_state_key',None)==self._opt_key(f0.shape) else None

        rng,_ = jxrandom.split(self.rng if self.rng_last is None else self.rng_last)
        self._run(f0,rng,opt_state,stimVal=stimVal)

    @_logged_training
    def train_append(self,n_append,fourierType=None,bSplit=None,stimInd=None,dtype=None,optimizer=None,nRestarts=1,stimVal=None):
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

        self._train_random(self.rng if self.rng_last is None else self.rng_last,nRestarts,stimVal)

    #- parametric filter banks
    def _parametric_init(self,family,n,bTied,init,b2D):
        init=dict(init or {})
        kTop=float(init.get('kTop',0.25))                   # peak frequency of the first filter, cycles per sample
        ratio=float(init.get('ratio',2.0))                  # scale ratio between adjacent filters (tied banks)
        rd=jnp.finfo(self.nrn.dtype).dtype
        full=lambda v: jnp.full((n,),v,dtype=rd) if not bTied else jnp.asarray(v,dtype=rd)
        params={}
        if bTied:
            params['log_kTop']=jnp.asarray(np.log(kTop),dtype=rd)
            params['log_ratio_m1']=jnp.asarray(np.log(ratio-1),dtype=rd)
        else:
            params['log_kPeak']=jnp.asarray(np.log(kTop/ratio**np.arange(n)),dtype=rd)
        if family=='morse':
            params['log_gamma']=full(np.log(float(init.get('gamma',2.0))))
            params['log_b']=full(np.log(float(init.get('b',2.0))))
        elif family=='loggabor':
            params['log_sigma_u']=full(np.log(float(init.get('sigma_u',0.5))))
        else:
            raise Exception("family must be 'morse' or 'loggabor'")
        if b2D:
            params['log_sigma_theta']=full(np.log(float(init.get('sigma_theta',np.deg2rad(20)))))
        return params

    def _parametric_filters(self,params,family,n,bTied,orientations):
        """unit-norm fourier-domain filters [ nPix x nF ] on the learned half-space coefficients"""
        flt=self.filter
        pix_dims=flt.pix_dims
        grids=np.meshgrid(*[_centered_freqs(d) for d in pix_dims],indexing='ij')
        kr=np.sqrt(sum(g**2 for g in grids)).ravel()[flt.index.pix]             # [ nPix_learned ]
        e=jnp.exp
        if bTied:
            kPeak=e(params['log_kTop'])/(1+e(params['log_ratio_m1']))**jnp.arange(n)
        else:
            kPeak=e(params['log_kPeak'])
        x=kr[:,None]/kPeak[None,:]                                               # [ nPix_learned x nF ]
        lx=jnp.log(jnp.maximum(x,1e-30))
        if family=='morse':
            gam,b=e(params['log_gamma']),e(params['log_b'])
            lprof=b*lx + (b/gam)*(1-jnp.exp(gam*lx))                             # peak 1 at x = 1
        else:
            su=e(params['log_sigma_u'])
            lprof=-lx**2/(2*su**2)
        if len(pix_dims)==2:
            ang=np.arctan2(grids[1],grids[0]).ravel()[flt.index.pix]
            d=(ang[:,None]-np.asarray(orientations)[None,:]+np.pi)%(2*np.pi)-np.pi
            st=e(params['log_sigma_theta'])
            lprof=lprof - d**2/(2*st**2) + jnp.where(np.cos(d)>0,0.,-jnp.inf)
        prof=jnp.exp(jnp.maximum(lprof,-700.))
        prof=prof/jnp.linalg.norm(prof,axis=0,keepdims=True)
        f=jnp.zeros((int(np.prod(pix_dims)),n),dtype=self.nrn.dtype)
        return f.at[flt.index.pix].set(prof.astype(self.nrn.dtype))

    def _train_generated(self,params,kind,cfg,stimVal=None):
        """
        optimize params (a dict of arrays; 'p' holds readout pooling weights) of the filters generated by kind ('parametric'
        or 'multiscale', see _generated_filters) with configuration cfg, on this unit's cost plus the generator's penalty,
        with this unit's Optimizer (optimizerType, lRate0, nIterMax, nStepsPerChunk, batchSize, bVerbose, patience with
        stimVal). Returns the final (or best validation) params.
        """
        loss=_GeneratedLoss(self,kind,cfg)
        val_fun=None
        if stimVal is not None:
            test=self._prepare_stim(stimVal)
            val_fun=lambda prm: _generated_heldout(loss,prm,self.rng,test.val,test.weights,test.yCtg,test.Y,
                                                   self.stim.val,self.stim.weights,self.stim.yCtg)
        rng,_=jxrandom.split(self.rng)
        params,self.rng_last=self.optimizer.minimize_generated(params,rng,self.stim,loss,val_fun)
        return params

    def _generated_filters(self,kind,params,cfg):
        # filters [ nPix x nF ] generated from params; with a learned phase (bPhase) each filter's spectrum is rotated by it
        if kind=='parametric':
            f=self._parametric_filters(params,cfg['family'],cfg['n'],cfg['bTied'],cfg['orientations'])
        else:
            f=self._multiscale_filters(params,cfg['nTot'],cfg)
        if 'phase' in params:
            f=f*jnp.exp(1j*params['phase']).astype(f.dtype)[None,:]
        return f

    def _generated_penalty(self,kind,params,cfg):
        # multiscale knotSmooth: squared second differences of each mother's tapered knots, relative to its energy
        if kind!='multiscale' or not cfg['knotSmooth']>0:
            return None
        nMothers=cfg['nMothers']
        tm=self._tapered_mothers(params,cfg)
        energy=jnp.sum((jnp.abs(tm)**2).reshape(nMothers,-1),axis=1)+1e-12
        tot=0.
        for v in (jnp.real(tm),jnp.imag(tm)):
            d=jnp.sum((jnp.diff(v,n=2,axis=1)**2).reshape(nMothers,-1),axis=1)
            if tm.ndim==3:                                                   # 2D: also along angle
                d=d+jnp.sum((jnp.diff(v,n=2,axis=2)**2).reshape(nMothers,-1),axis=1)
            tot=tot+jnp.sum(d/energy)
        return cfg['knotSmooth']*tot

    def _set_generated(self,f,params):
        # generated filters [ nPix x nF ] become this unit's ordinary filters
        self.filter.out=jnp.reshape(f,self.filter._shape_exp)
        self.out_params={'f':f[self.filter._insert_index_jx]}
        self._opt_param_shape=self.out_params['f'].shape
        self._opt_state_key=None
        self.opt_state=None
        if 'p' in params:
            self._set_p(params['p'])

    #- non-parametric multiscale filter banks
    @staticmethod
    def _interp_uniform(x,x0,dx,v):
        """
        linear interpolation of knot values v [ nKnot x ... ] on the uniform grid x0 + dx*arange(nKnot) at points x
        [ nPts x ... ] (broadcast), zero outside the grid
        """
        n=v.shape[0]
        t=(x-x0)/dx
        i=jnp.floor(t)
        w=t-i
        i=i.astype(int)
        inside=(t>=0)&(t<=n-1)
        i0=jnp.clip(i,0,n-1)
        i1=jnp.clip(i+1,0,n-1)
        val=v[i0]*(1-w)+v[i1]*w
        return jnp.where(inside,val,0.)

    @staticmethod
    def _knot_taper(nKnot,edgeTaper):
        """[ nKnot ] window: cosine from 0 at the outermost knots to 1 over a fraction edgeTaper of the knot range at each end"""
        t=np.linspace(0.,1.,nKnot)
        d=np.minimum(t,1-t)
        if edgeTaper<=0:
            return np.ones(nKnot)
        return np.where(d<edgeTaper,0.5*(1-np.cos(np.pi*d/edgeTaper)),1.)

    @staticmethod
    def _tapered_mothers(params,cfg):
        """complex mother knots [ nMothers x nKnot (x nKnotTheta) ] after the log-frequency edge taper"""
        w=jnp.asarray(cfg.get('knotTaper',np.ones(params['mother_re'].shape[1])),dtype=params['mother_re'].dtype)
        w=w.reshape((1,-1)+(1,)*(params['mother_re'].ndim-2))
        return (params['mother_re'] + 1j*params['mother_im'])*w

    def _multiscale_filters(self,params,n,cfg):
        """
        dilated (and, in 2D, rotated) copies of non-parametric mother filters, as unit-norm fourier-domain filters
        [ nPix x nF ]. Each mother is complex knot values over log(k/k_peak) (1D: [ nKnot ]; 2D: [ nKnot x nKnotTheta ]
        over log radius and angle), linearly interpolated; params hold [ nMothers x ... ]. Filter order: mother first,
        then the scale (and orientation) filters of cfg (every mother shares the same scales and orientations).
        """
        flt=self.filter
        pix_dims=flt.pix_dims
        grids=np.meshgrid(*[_centered_freqs(d) for d in pix_dims],indexing='ij')
        kr=np.sqrt(sum(g**2 for g in grids)).ravel()[flt.index.pix]
        lk=np.log(np.maximum(kr,1e-12))                                           # [ nPts ]
        e=jnp.exp
        if cfg['scaleType']=='tied':
            # scaleExp: the scale step of each filter (0, 1, 2, ... ; +1/2 for interleaved filters)
            logPeak=params['log_kTop'] - jnp.log1p(e(params['log_ratio_m1']))*jnp.asarray(cfg['scaleExp'])
        elif cfg['scaleType']=='fixed':
            logPeak=jnp.asarray(np.log(cfg['kPeak']))
        else:
            logPeak=params['log_kPeak']
        mothers=self._tapered_mothers(params,cfg)                                 # [ nMothers x nKnot (x nKnotTheta) ]
        u=lk[:,None]-logPeak[None,:]                                              # [ nPts x nF_mother ] log(k/kPeak_j)
        uw=cfg['uWidth']
        du=2*uw/(cfg['nKnot']-1)
        allvals=[]
        for im in range(cfg.get('nMothers',1)):
          mother=mothers[im]
          if len(pix_dims)==1:
            vals=self._interp_uniform(u,-uw,du,mother)                            # mother [ nKnot ]
          else:
            ang=np.arctan2(grids[1],grids[0]).ravel()[flt.index.pix]
            d=(ang[:,None]-np.asarray(cfg['orientations'])[None,:]+np.pi)%(2*np.pi)-np.pi   # [ nPts x nF ]
            nT=cfg['nKnotTheta']
            dth=np.pi/(nT-1)
            # bilinear on [ log radius x angle in [-pi/2, pi/2] ]
            tu=(u+uw)/du
            tt=(d+np.pi/2)/dth
            iu=jnp.floor(tu); wu=tu-iu; iu=iu.astype(int)
            it=np.floor(tt); wt=tt-it; it=it.astype(int)
            inside=(tu>=0)&(tu<=cfg['nKnot']-1)&(tt>=0)&(tt<=nT-1)
            c=lambda a,m: jnp.clip(a,0,m-1)
            v00=mother[c(iu,cfg['nKnot']),c(it,nT)]
            v01=mother[c(iu,cfg['nKnot']),c(it+1,nT)]
            v10=mother[c(iu+1,cfg['nKnot']),c(it,nT)]
            v11=mother[c(iu+1,cfg['nKnot']),c(it+1,nT)]
            vals=((1-wu)*((1-wt)*v00+wt*v01)+wu*((1-wt)*v10+wt*v11))
            vals=jnp.where(inside,vals,0.)
          allvals.append(vals)
        vals=jnp.concatenate(allvals,axis=1)                                      # [ nPts x nMothers*nF_mother ]
        norm=jnp.sqrt(jnp.sum(jnp.abs(vals)**2,axis=0,keepdims=True))
        vals=vals/jnp.where(norm>0,norm,1)
        f=jnp.zeros((int(np.prod(pix_dims)),n),dtype=self.nrn.dtype)
        return f.at[flt.index.pix].set(vals.astype(self.nrn.dtype))

    @_logged_training
    def train_multiscale(self,n=None,fourierType=2,nKnot=25,uWidth=2.5,scaleType='tied',kTop=0.25,ratio=2.0,kPeak=None,
                         orientations=None,nKnotTheta=13,knotSmooth=0.,init='loggabor',dtype=None,stimVal=None,
                         nScales=None,nOrientations=None,bInterleave=False,nMothers=1,motherInitNoise=0.1,edgeTaper=0.,
                         bPhase=False):
        """
        learn a multiscale filter bank whose filters are copies of one (or nMothers) non-parametric mother filter: dilated (shifted in
        log frequency) and, for 2D stimuli, rotated (shifted in angle). The mother filter is free: complex spectrum values
        at nKnot points uniform over log(k/k_peak) in [-uWidth, uWidth] (2D: times nKnotTheta angles over [-pi/2, pi/2],
        one-sided), linearly interpolated, zero outside. Filters are unit norm, in the fourier domain (fourierType 1 or 2).

        scaleType - 'tied'  : peak frequencies kTop/ratio^j with kTop and ratio learned (initialized from kTop, ratio)
                    'fixed' : peak frequencies kPeak [ nF ] (default kTop/ratio^j), not learned
                    'free'  : a learned peak frequency per filter (initialized from kTop/ratio^j)
        orientations - 2D: [ nF ] radians from the first stimulus axis (default 0)
        knotSmooth   - weight of a penalty on squared second differences of the mother's knot values (smooth spectra)
        init         - initial mother: 'loggabor' (a real log-gaussian bump, sigma_u = uWidth/3) or 'random'
        nScales, nOrientations - 2D: build the filters as a grid instead of per-filter lists: peaks kTop/ratio^j
                       (j = 0..nScales-1) at orientations o*pi/nOrientations (o = 0..nOrientations-1, wrapped to
                       (-pi/2, pi/2]), n = nScales*nOrientations filters (n need not be given)
        bInterleave  - with the grid: add an interleaved set of the same size at the tritones (half a scale step,
                       kTop/ratio^(j+1/2): half an octave for ratio 2) in the half-rotations (orientations offset by
                       half the orientation step, pi/(2 nOrientations)). All filters are copies of the same mother;
                       with scaleType='tied' the interleaved scales stay halfway (in log frequency) as ratio is learned.
                       Filter order: the grid (scales x orientations, scale first), then the interleaved grid.
        nMothers     - learn this many mother filters simultaneously. Each gets the same set of scale (and orientation)
                       filters, so the bank has nMothers * n filters, ordered mother first; the scales (kTop and ratio,
                       kPeak, or the free peaks) and orientations are shared by all mothers. n (or the grid) counts the
                       filters per mother.
        edgeTaper    - force each mother to vanish at the ends of its log-frequency support: its knots are multiplied by a
                       fixed window that rises (cosine) from 0 at the outermost knots to 1 over this fraction of the knot
                       range at each end (0 = no taper; e.g. 0.2 tapers 5 of 25 knots per end). Without it a mother can
                       move its energy to the edge of the support, making its dilated copies narrowband filters cut off at
                       the boundary. The reported mother, the filters, and knotSmooth use the tapered knots. 2D: the taper
                       is along log radius only.
        motherInitNoise - with nMothers > 1, every mother's initial knots get independent complex gaussian noise of this
                       size relative to the initial bump (seeded from the unit's rng): identical mothers would receive
                       identical gradients and never diverge.
        bPhase       - also learn a phase per filter (exp(i phase) on its spectrum), so the dilated copies of a mother need
                       not share its phase (e.g. phase alignment across scales for a pooled readout)
        Uses this unit's Optimizer (and Objective regType on the generated filters is not applied; use knotSmooth).
        The learned filters become ordinary filters (unit.out); multiscale_values() returns the mother filter and scales.
        Not implemented for split stimuli.
        """
        if fourierType not in (1,2):
            raise Exception('train_multiscale learns in the fourier domain: fourierType must be 1 or 2')
        if scaleType not in ('tied','fixed','free'):
            raise Exception("scaleType must be 'tied', 'fixed', or 'free'")
        bGrid=nScales is not None or nOrientations is not None
        if bInterleave and not bGrid:
            raise Exception('bInterleave needs the filter grid: give nScales and nOrientations')
        scaleExp=None
        if bGrid:
            if nScales is None or nOrientations is None or nScales<1 or nOrientations<1:
                raise Exception('give both nScales and nOrientations (at least 1)')
            if kPeak is not None or orientations is not None:
                raise Exception('with nScales and nOrientations, kPeak and orientations are set by the grid')
            jj,oo=np.meshgrid(np.arange(nScales),np.arange(nOrientations),indexing='ij')
            scaleExp=jj.ravel().astype(float)
            orientations=oo.ravel()*np.pi/nOrientations
            if bInterleave:
                scaleExp=np.concatenate((scaleExp,scaleExp+0.5))
                orientations=np.concatenate((orientations,orientations+np.pi/(2*nOrientations)))
            orientations=(orientations+np.pi/2)%np.pi-np.pi/2                     # wrap to [-pi/2, pi/2)
            orientations=np.where(np.isclose(orientations,-np.pi/2),np.pi/2,orientations)
            nGrid=len(scaleExp)
            if n is not None and n!=nGrid:
                raise Exception('n (' + str(n) + ') does not match the grid (' + str(nGrid) + ' filters)')
            n=nGrid
        elif n is None:
            raise Exception('give n, or nScales and nOrientations')
        if nKnot<3 or (nKnotTheta is not None and nKnotTheta<2):
            raise Exception('nKnot must be at least 3 and nKnotTheta at least 2')
        nMothers=int(nMothers)
        if nMothers<1:
            raise Exception('nMothers must be at least 1')
        if not 0<=edgeTaper<=0.5:
            raise Exception('edgeTaper must be in [0, 0.5]')
        nTot=nMothers*n
        self.pool_p=self.nrn_p=None
        self._finalize(nTot,np.arange(nTot),fourierType=fourierType,dtype=dtype,bSplit=False)
        if len(self.filter.pix_dims) not in (1,2):
            raise Exception('train_multiscale supports 1D and 2D stimuli')
        b2D=len(self.filter.pix_dims)==2
        if bGrid and not b2D:
            raise Exception('nScales, nOrientations, and bInterleave need 2D stimuli (rotation)')
        orientations=np.zeros(n) if orientations is None else np.asarray(orientations,dtype=float)
        if b2D and orientations.shape!=(n,):
            raise Exception('orientations must have one value per filter')
        if scaleExp is None:
            scaleExp=np.arange(n,dtype=float)
        kPeak0=kTop/ratio**scaleExp if kPeak is None else np.asarray(kPeak,dtype=float)
        if kPeak0.shape!=(n,):
            raise Exception('kPeak must have one value per filter')
        rd=jnp.finfo(self.nrn.dtype).dtype
        cfg=dict(scaleType=scaleType,uWidth=float(uWidth),nKnot=int(nKnot),nKnotTheta=int(nKnotTheta),
                 orientations=orientations,kPeak=kPeak0,scaleExp=scaleExp,nMothers=nMothers,
                 knotTaper=self._knot_taper(int(nKnot),float(edgeTaper)),nTot=nTot,knotSmooth=float(knotSmooth))

        ug=np.linspace(-uWidth,uWidth,nKnot)
        if init=='loggabor':
            m=np.exp(-ug**2/(2*(uWidth/3)**2))
            if b2D:
                tg=np.linspace(-np.pi/2,np.pi/2,nKnotTheta)
                m=m[:,None]*np.exp(-tg[None,:]**2/(2*np.deg2rad(20)**2))
        elif init=='random':
            rng=np.random.default_rng(int(np.asarray(jxrandom.key_data(self.rng)).ravel()[-1]))
            m=rng.standard_normal((nKnot,nKnotTheta) if b2D else (nKnot,))
        else:
            raise Exception("init must be 'loggabor' or 'random'")
        m=np.broadcast_to(m,(nMothers,)+np.shape(m)).copy()
        mi=np.zeros(np.shape(m))
        if nMothers>1 and motherInitNoise>0:
            nrng=np.random.default_rng(int(np.asarray(jxrandom.key_data(self.rng)).ravel()[-1])+7919)
            scale=motherInitNoise*np.max(np.abs(m))
            m=m+scale*nrng.standard_normal(m.shape)
            mi=scale*nrng.standard_normal(m.shape)
        params={'mother_re':jnp.asarray(m,dtype=rd),'mother_im':jnp.asarray(mi,dtype=rd)}
        if scaleType=='tied':
            params['log_kTop']=jnp.asarray(np.log(kTop),dtype=rd)
            params['log_ratio_m1']=jnp.asarray(np.log(ratio-1),dtype=rd)
        elif scaleType=='free':
            params['log_kPeak']=jnp.asarray(np.log(kPeak0),dtype=rd)
        if bPhase:
            params['phase']=jnp.zeros(nTot,dtype=rd)
        if self.nrn.bParams:
            params['p']=self._p0()

        params=self._train_generated(params,'multiscale',cfg,stimVal=stimVal)
        self._set_generated(self._generated_filters('multiscale',params,cfg),params)

        mother=np.asarray(self._tapered_mothers(params,cfg))
        if nMothers==1:
            mother=mother[0]                                                      # single mother: [ nKnot (x nKnotTheta) ]
        if scaleType=='tied':
            r=float(1+np.exp(np.asarray(params['log_ratio_m1'])))
            kp=float(np.exp(np.asarray(params['log_kTop'])))/r**scaleExp
        elif scaleType=='fixed':
            r,kp=None,kPeak0
        else:
            r,kp=None,np.exp(np.asarray(params['log_kPeak']))
        tile=lambda v: np.tile(np.asarray(v),nMothers)                           # per mother -> per filter (mother first)
        out={'mother':mother,'log_freq_knots':ug,'k_peak':tile(kp),'ratio':r,'scaleType':scaleType,'fourierType':fourierType}
        out['scale_step']=tile(scaleExp)
        out['nMothers']=nMothers
        out['edgeTaper']=float(edgeTaper)
        out['mother_index']=np.repeat(np.arange(nMothers),n)
        if bPhase:
            out['phase']=np.asarray(params['phase'])
        if b2D:
            out['angle_knots']=np.linspace(-np.pi/2,np.pi/2,nKnotTheta)
            out['orientations']=tile(orientations)
        if bGrid:
            out['nScales'],out['nOrientations'],out['bInterleave']=nScales,nOrientations,bool(bInterleave)
            out['interleaved']=tile(scaleExp%1!=0)
        self.multiscale_out=out

    def multiscale_values(self):
        """the learned mother filter (complex knot values and their log-frequency, and angle, positions) and the peak
        frequencies (cycles/sample) and ratio of the last train_multiscale"""
        if getattr(self,'multiscale_out',None) is None:
            raise Exception('train_multiscale has not been run')
        return self.multiscale_out

    def parametric_values(self):
        """interpretable parameters of the last train_parametric: peak frequencies (cycles/sample), shapes, orientations"""
        if getattr(self,'param_out',None) is None:
            raise Exception('train_parametric has not been run')
        return self.param_out

    @_logged_training
    def train_parametric(self,n,family='morse',fourierType=2,bTied=True,orientations=None,init=None,dtype=None,stimVal=None,
                         bPhase=False):
        """
        learn a parametric filter bank instead of free filters: generalized Morse, |H(k)| = (k/kp)^b exp((b/gamma)(1-(k/kp)^gamma)),
        or log-Gabor, exp(-log(k/kp)^2/(2 sigma_u^2)), with a gaussian angular profile (width sigma_theta, one-sided) around
        each filter's orientation for 2D stimuli. With bTied, the filters are dilations of one mother filter with peak
        frequencies kTop/ratio^j (parameters kTop, ratio, and one shape); otherwise every filter has its own peak and shape.
        orientations [ nF ] radians from the first stimulus axis (2D; default 0). init overrides starting values (kTop, ratio,
        gamma, b, sigma_u, sigma_theta). Uses this unit's Optimizer (optimizerType, lRate0, nIterMax, nStepsPerChunk,
        patience with stimVal). The learned filters become ordinary filters (unit.out), so evaluation, saving, and
        train_recurse (free refinement from the parametric solution) work as usual; parametric_values() returns the
        parameters. Not implemented for split stimuli.
        bPhase - also learn a phase per filter, which multiplies its spectrum by exp(i phase): the profiles are otherwise
                 zero-phase (even filters with fourierType=1; cosine/sine pairs with fourierType=2), and a learned phase
                 gives odd or intermediate filters, or rotates each quadrature pair (which matters with 'phase'
                 normalization and pooled readouts, and through the response-dependent noise)
        """
        if fourierType not in (1,2):
            raise Exception('train_parametric learns in the fourier domain: fourierType must be 1 or 2')
        self.pool_p=self.nrn_p=None
        self._finalize(n,np.arange(n),fourierType=fourierType,dtype=dtype,bSplit=False)
        if len(self.filter.pix_dims) not in (1,2):
            raise Exception('train_parametric supports 1D and 2D stimuli')
        b2D=len(self.filter.pix_dims)==2
        orientations=np.zeros(n) if orientations is None else np.asarray(orientations,dtype=float)
        if b2D and orientations.shape!=(n,):
            raise Exception('orientations must have one value per filter')
        params=self._parametric_init(family,n,bTied,init,b2D)
        if bPhase:
            params['phase']=jnp.zeros(n,dtype=jnp.finfo(self.nrn.dtype).dtype)
        if self.nrn.bParams:
            params['p']=self._p0()
        cfg=dict(family=family,n=n,bTied=bTied,orientations=orientations)
        params=self._train_generated(params,'parametric',cfg,stimVal=stimVal)
        self._set_generated(self._generated_filters('parametric',params,cfg),params)
        e=lambda k: np.exp(np.asarray(params[k]))
        out={'family':family,'bTied':bTied,'fourierType':fourierType}
        if bTied:
            out['ratio']=float(1+e('log_ratio_m1'))
            out['k_peak']=float(e('log_kTop'))/out['ratio']**np.arange(n)
        else:
            out['k_peak']=e('log_kPeak')
        if family=='morse':
            out['gamma'],out['b']=e('log_gamma'),e('log_b')
        else:
            out['sigma_u']=e('log_sigma_u')
        if b2D:
            out['sigma_theta']=e('log_sigma_theta')
            out['orientations']=orientations
        if bPhase:
            out['phase']=np.asarray(params['phase'])
        self.param_out=out

    #- evaluation
    def _prepare_stim(self,stim):
        # other stimuli in this unit's learning domain and precision
        if not self.nrn.bFinalized:
            raise Exception('train (or finalize) the unit first')
        if (stim.nCtg!=self.stim.nCtg or np.shape(stim.Y)!=np.shape(self.stim.Y)
                or not np.allclose(np.asarray(stim.Y),np.asarray(self.stim.Y))):
            raise Exception('stimuli must have the same categories (Y) as the training stimuli')
        if getattr(stim,'Yperiod',None)!=getattr(self.stim,'Yperiod',None):
            raise Exception('stimuli must have the same Yperiod as the training stimuli')
        return copy.copy(stim)._finalize(self.nrn.dtype,None,self.nrn.bFourier,self.nrn.bSplit)

    @partial(jit, static_argnames=['self'])
    def _loss_fun_heldout(self,params,rng_key,stimval,stimweights,yCtg,Y,refval,refweights,refy):
        f=params['f']
        p=params.get('p')
        W=self.nrn.whitening(refval,f,refweights) if self.nrn._bWhiten else None
        G=self.nrn.gain(refval,f,refweights,W,p)

        def one(key):
            obs=self.nrn.main(key,stimval,f,stimweights,W,p,G)
            ref=self.nrn.main(key,refval,f,refweights,W,p,G)
            lAll,Yc=self._lik_parts(self._likelihoods_heldout(obs,ref,refweights,Y,refy))
            return self.objective.lrn_main(lAll,stimweights,yCtg,Y,refweights,Yc),None

        return self._noise_average(one,rng_key)[0]

    def evaluate(self,stim):
        """
        cost of decoding other stimuli (e.g. a held-out test set) with the current filters. This unit's stimuli are the
        training set: the category response distributions (AMA-Gauss), the reference stimuli (full AMA), the prior,
        and the whitening all come from them.
        """
        test=self._prepare_stim(stim)
        return float(self._loss_fun_heldout(self._params_out(),self.rng,test.val,test.weights,test.yCtg,test.Y,
                                            self.stim.val,self.stim.weights,self.stim.yCtg))

    def _log_posterior(self,stim=None):
        # (log posterior, the decoded Stim, the within-category values Yc or None)
        f=self.filter.out_flat
        if stim is None:
            lAll,Yc=self._lik_parts(self._likelihoods(self._nrn_out(),self.stim.weights,self.stim.Y,yRef=self._yRef(self.stim.yCtg)))
            return Objective._posterior__true(lAll,self.stim.weights),self.stim,Yc
        test=self._prepare_stim(stim)
        W=self.nrn.whitening(self.stim.val,f,self.stim.weights) if self.nrn._bWhiten else None
        G=self.nrn.gain(self.stim.val,f,self.stim.weights,W,self._p())
        obs=self.nrn.main(self.rng,test.val,f,test.weights,W,self._p(),G)
        ref=self.nrn.main(self.rng,self.stim.val,f,self.stim.weights,W,self._p(),G)
        lAll,Yc=self._lik_parts(self._likelihoods_heldout(obs,ref,self.stim.weights,self.stim.Y,self._yRef(self.stim.yCtg)))
        return Objective._posterior__true(lAll,self.stim.weights),test,Yc

    def estimates(self,estType='mode',stim=None):
        """
        estimates of the latent variable [ nStim_Ctg x nCtg (x nDim) ] (grouped like Stim.val; see Stim.weights) for the
        training stimuli, or for other stimuli decoded with the training set: 'mode' (MAP), 'mean' (MMSE; circular on
        circular dimensions), 'median', or 'cmean' (circular mean, Y in radians). With Model bWithin, continuous
        estimates within the categories
        """
        lpost,st,Yc=self._log_posterior(stim)
        return np.asarray(getattr(Objective,'_est__'+estType)(lpost,st.Y,st.Yperiod,Yc))

    def performance(self,estType='mode',stim=None):
        """
        estimation performance per latent level for the training stimuli, or for other stimuli decoded with the training
        set: bias, sd, and rmse of the estimates ([ nCtg (x nDim) ]; errors wrap on circular dimensions) and over all
        stimuli (rmseAll); pCorrect and confusion [ true x MAP category ] of the MAP category; and cost, the mean -log
        posterior at the correct level. Errors are measured from each stimulus's own latent value (Stim y), which for
        stimuli without their own values is their category's level
        """
        lpost,st,Yc=self._log_posterior(stim)
        lpost=np.asarray(lpost)
        est=np.asarray(getattr(Objective,'_est__'+estType)(jnp.asarray(lpost),st.Y,st.Yperiod,Yc))
        w=np.asarray(st.weights)>0
        Y=np.asarray(st.Y)
        y=np.asarray(st.yCtg)
        nCtg=len(Y)
        best=np.argmax(lpost,axis=-1)
        confusion=np.stack([np.bincount(best[w[:,c],c],minlength=nCtg) for c in range(nCtg)])
        err=[np.asarray(_wrap(jnp.asarray(est[w[:,c],c]-y[w[:,c],c]),st.Yperiod)) for c in range(nCtg)]
        correct=np.diagonal(lpost,axis1=-2,axis2=-1)
        return {'Y':Y,
                'estimates':est,
                'bias':np.array([e.mean(0) for e in err]),
                'sd':np.array([e.std(0) for e in err]),
                'rmse':np.array([np.sqrt(np.mean(e**2,0)) for e in err]),
                'rmseAll':np.sqrt(np.mean(np.concatenate(err)**2,0)),
                'pCorrect':np.diagonal(confusion)/confusion.sum(1),
                'confusion':confusion,
                'cost':float(-correct[w].mean())}

    def cross_validate(self,n,k=5,seed=0,**train_kw):
        """
        k-fold cross-validation, stratified by category: train n new filters on each training fold with copies of this
        unit's settings, and evaluate them on the held-out fold. Returns train and test costs [ k ] and filters [ k x ... ].
        """
        res={'train':[],'test':[],'filters':[]}
        for i,(train,test) in enumerate(self.stim_full.folds(k,seed)):
            unit=Unit(train,self.nrn.copy(),self.model.copy(),self.objective.copy(),
                      optimizer=None if self.optimizer is None else self.optimizer.copy(),seed=self.seed+i)
            unit.train_new(n,**train_kw)
            res['train'].append(float(unit.loss))
            res['test'].append(unit.evaluate(test))
            res['filters'].append(np.asarray(unit.out))
        return {key:np.array(val) for key,val in res.items()}

    #- save/load
    def save(self,fname):
        """save settings, filters, optimizer state, random keys, and frozen whitening (not the stimuli) with pickle"""
        if not self.nrn.bFinalized:
            raise Exception('train (or finalize) the unit first')
        asnp=lambda tree: None if tree is None else tree_util.tree_map(np.asarray,tree)
        state={'version':1,
               'nrn':_get_copy_dict(self.nrn,_NRN_EXCL),
               'model':_get_copy_dict(self.model),
               'objective':_get_copy_dict(self.objective),
               'optimizer':None if self.optimizer is None else _get_copy_dict(self.optimizer,_OPT_EXCL),
               'finalize':dict(n=self.filter.n,dtype=np.dtype(self.nrn.dtype).name,bFourier=self.nrn.bFourier,
                               bAnalytic=bool(self.nrn.bAnalytic),bSplit=self.nrn.bSplit,
                               stimInd=getattr(self,'_stimInd',None)),
               'out':np.asarray(self.filter.out),
               'last':asnp(self.filter.last),
               'opt_state':asnp(self.opt_state),
               'opt_param_shape':getattr(self,'_opt_param_shape',None),
               'opt_state_key':getattr(self,'_opt_state_key',None),
               'loss_hist':list(getattr(self.optimizer,'loss_hist',[])),
               'restart_costs':self.restart_costs,
               'seed':self.seed,
               'rng':np.asarray(jxrandom.key_data(self.rng)),
               'rng_last':None if self.rng_last is None else np.asarray(jxrandom.key_data(self.rng_last)),
               'W':asnp(self.nrn._W),
               'pool_p':asnp(self.pool_p),
               'nrn_p':asnp(getattr(self,'nrn_p',None)),
               'train_log':list(getattr(self,'train_log',[]) or []),
               'ama_source_sha256':source_sha256(),
               'name':getattr(self,'name',None),
               'multiscale_out':getattr(self,'multiscale_out',None),
               'param_out':getattr(self,'param_out',None),
               'stim_summary':self._stim_summary()}
        with open(fname,'wb') as fh:
            pickle.dump(state,fh)

    @classmethod
    def load(cls,fname,stim):
        """load a unit saved with Unit.save, with its training stimuli"""
        with open(fname,'rb') as fh:
            state=pickle.load(fh)
        asjnp=lambda tree: None if tree is None else tree_util.tree_map(jnp.asarray,tree)
        unit=cls(stim,Nrn(**state['nrn']),Model(**state['model']),Objective(**state['objective'],_bCopy=True),
                 optimizer=None if state['optimizer'] is None else Optimizer(**state['optimizer']),
                 seed=state['seed'],rng=jxrandom.wrap_key_data(state['rng']),
                 rng_last=None if state['rng_last'] is None else jxrandom.wrap_key_data(state['rng_last']),name=state.get('name'))
        fin=state['finalize']
        fourierType=(2 if fin['bAnalytic'] else 1) if fin['bFourier'] else 0
        if unit.optimizer is None:
            unit.optimizer=Optimizer()
        unit._finalize(fin['n'],np.arange(fin['n']),fourierType=fourierType,bSplit=fin['bSplit'],dtype=jnp.dtype(fin['dtype']),
                       stimInd=fin.get('stimInd'))
        if state['optimizer'] is None:
            unit.optimizer=None
        unit.filter.out=jnp.asarray(state['out'])
        unit.filter.last=asjnp(state['last'])
        unit.opt_state=asjnp(state['opt_state'])
        unit._opt_param_shape=state['opt_param_shape']
        unit._opt_state_key=state.get('opt_state_key')                   # older saves: the state is not reused
        if unit.optimizer is not None:
            unit.optimizer.loss_hist=state['loss_hist']
        unit.restart_costs=state['restart_costs']
        unit.nrn._W=asjnp(state['W'])
        unit.pool_p=None if state.get('pool_p') is None else np.asarray(state['pool_p'])
        unit.nrn_p=None if state.get('nrn_p') is None else {k:np.asarray(v) for k,v in state['nrn_p'].items()}
        unit.train_log=list(state.get('train_log') or [])
        unit.name=state.get('name')
        if state.get('multiscale_out') is not None:
            unit.multiscale_out=state['multiscale_out']
        if state.get('param_out') is not None:
            unit.param_out=state['param_out']
        return unit

    #- yaml configuration
    def _stim_summary(self):
        st=self.stim_full
        return {'dims':[int(d) for d in st.dims],'nCtg':int(st.nCtg),'nStim':int(np.sum(np.asarray(st.weights)>0)),
                'Y':_yaml_safe(np.asarray(st.Y)),'Yperiod':_yaml_safe(getattr(st,'Yperiod',None)),
                'bContinuous':bool(getattr(st,'bContinuous',False)),
                'bIsFourier':bool(st.bIsFourier),'nSplit':int(st.nSplit or 0)}

    def config(self):
        """
        this unit's options and settings as a plain dict (yaml-safe): Nrn, Model, Objective and Optimizer settings, the
        seed, the filter layout, a summary of the training stimuli, and the training calls made so far (method and
        arguments, from train_new / train_recurse / train_append / train_parametric / train_multiscale). Rebuild with
        Unit.from_config. Learned filters are not included (use save / load).
        """
        cfg={'ama_config_version':CONFIG_VERSION,'ama_source_sha256':source_sha256(),'name':getattr(self,'name',None),
             'seed':_yaml_safe(self.seed)}
        cfg.update(_components_config(self.nrn,self.model,self.objective,self.optimizer))
        if self.nrn.bFinalized:
            cfg['filters']={'n':int(self.filter.n),'fourierType':(2 if self.nrn.bAnalytic else 1) if self.nrn.bFourier else 0,
                            'dtype':np.dtype(self.nrn.dtype).name,'bSplit':bool(self.nrn.bSplit),
                            'shape':list(np.shape(self.filter.out)),'bPoolWeights':self.pool_p is not None}
        cfg['train']=_yaml_safe(list(getattr(self,'train_log',[]) or []))
        cfg['stim']=self._stim_summary()
        return cfg

    def save_config(self,fname):
        """write config() to a yaml file"""
        import yaml
        with open(fname,'w') as fh:
            yaml.safe_dump(self.config(),fh,sort_keys=False,default_flow_style=None,width=120)

    @classmethod
    def from_config(cls,cfg,stim,bTrain=False,stimVal=None):
        """
        build a unit from a configuration (dict, or the path of a yaml file written by save_config / config_from_saved)
        with these training stimuli. bTrain replays the recorded training calls in order; stimVal is passed to calls that
        were made with validation stimuli. Unknown top-level keys (e.g. project metadata) are ignored.
        """
        if isinstance(cfg,str) or hasattr(cfg,'__fspath__'):
            cfg=load_config(cfg)
        if int(cfg.get('ama_config_version',1))>CONFIG_VERSION:
            raise Exception('config version ' + str(cfg.get('ama_config_version')) + ' is newer than this ama (' + str(CONFIG_VERSION) + ')')
        opt=cfg.get('optimizer')
        unit=cls(stim,Nrn(**cfg['nrn']),Model(**cfg['model']),Objective(**cfg['objective'],_bCopy=True),
                 optimizer=None if opt is None else Optimizer(**opt),seed=cfg.get('seed'),name=cfg.get('name'))
        if bTrain:
            for step in cfg.get('train') or []:
                args=dict(step.get('args') or {})
                if 'stimVal' in args:
                    args['stimVal']=stimVal if args['stimVal'] else None
                if args.get('optimizer') is not None:
                    args['optimizer']=Optimizer(**args['optimizer'])
                if args.get('dtype') is not None:
                    args['dtype']=jnp.dtype(args['dtype'])
                if step['method'] not in ('train_new','train_recurse','train_append','train_parametric','train_multiscale'):
                    raise Exception('unknown training method in config: ' + str(step['method']))
                getattr(unit,step['method'])(**args)
        return unit

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
        return self._loss_fun(self._params_out(),self.rng,self.stim.val,self.stim.weights,self.stim.yCtg,self.stim.Y,self.nrn._W)

    @property
    def penalty(self):
        """filter penalty of Objective regType (without regWeight), which training adds to the cost"""
        if str(self.objective.regType).lower()=='none':
            return 0.
        return float(self._penalty(self.filter.out_flat))

    def _p(self):
        if not self.nrn.bParams:
            return None
        p=self._p_old()
        if not p:
            return None
        return {k:jnp.asarray(v) for k,v in p.items()}

    def _params_out(self):
        out={'f':self.filter.out_flat}
        if self._p() is not None:
            out['p']=self._p()
        return out

    def _penalty(self,f):
        """mean over filters of the regType penalty of the learned coefficients; f [ nPix x (nSplit) x nF ]"""
        flt=self.filter
        F=jnp.reshape(f,flt._shape_exp)
        nLead=len(flt.pix_dims)
        mask=np.zeros(int(np.prod(flt.pix_dims)),dtype=bool)
        mask[flt.index.pix]=True
        mask=np.reshape(mask,flt.pix_dims+(1,)*(F.ndim-nLead))
        reg=str(self.objective.regType).lower()
        if reg=='l1':
            tot=jnp.sum(jnp.abs(F)*mask)
        else:
            tot=0.
            for a in range(nLead):
                valid=mask & np.roll(mask,-1,axis=a)
                edge=np.ones(flt.pix_dims[a],dtype=bool)
                edge[-1]=False
                valid=valid & np.reshape(edge,tuple(-1 if i==a else 1 for i in range(F.ndim)))
                tot=tot+jnp.sum(jnp.abs(F-jnp.roll(F,-1,axis=a))**2*valid)
        return tot/flt.n

    @property
    def responses(self):
        return Response(*self.nrn.main(self.rng,self.stim.val,self.filter.out_flat,self.stim.weights,self.nrn._W,self._p()),self.stim)

    def _nrn_out(self):
        return self.nrn.main(self.rng,self.stim.val,self.filter.out_flat,self.stim.weights,self.nrn._W,self._p())

    @property
    def likelihoods(self):
        return self._likelihoods(self._nrn_out(),self.stim.weights,self.stim.Y)

    @property
    def posterior(self):
        return self.objective._posterior_fun(self.likelihoods,self.stim.weights)

    @property
    def error(self):
        Y,per=self.stim.Y,self.objective._Yperiod
        lAll,Yc=self._lik_parts(self._likelihoods(self._nrn_out(),self.stim.weights,Y,yRef=self._yRef(self.stim.yCtg)))
        lpost=self.objective._posterior_fun(lAll,self.stim.weights)
        return self.objective._err_fun(self.objective._est_fun(lpost,Y,per,Yc),self.stim.yCtg,self.objective.log_target(Y,self.stim.yCtg),
                                       Y,per,self.objective._err_opts())

    #- loss functions
    def _yRef(self,yCtg):
        # the reference stimuli's own latent values, which Model bWithin uses
        return yCtg if self.model.bWithin else None

    @staticmethod
    def _lik_parts(lik):
        # (lAll, Yc) from a likelihood result: lAll, or (lAll, Yc) with Model bWithin
        return lik if isinstance(lik,tuple) else (lik,None)

    def _likelihoods(self,nrn_out,stimweights,Y,rng_ref=None,mstate=None,yRef=None,nbr=None,obs=None):
        """
        log-likelihoods lAll, or (lAll, Yc) with yRef (Model bWithin). rng_ref: the key of the random references (nRef, or
        the tail of nNeighbors); mstate: a mixture fit to warm start from, then (result, the new fit); nbr: the nearest
        references (nNeighbors); obs: (indices [ mMax x nCtg ], weights) of a batch of stimuli to decode against all of
        them (nNeighbors with batchSize), lAll [ mMax x nCtg x nCtg ]
        """
        R,Rm,RVar=[_flatten_responses(x) for x in self.model._response_fun(*nrn_out)]
        noiseCov=self.nrn._corr_fun(RVar,stimweights,self.nrn.rho)
        if mstate is not None:
            # a mixture warm started from mstate: (likelihoods, the new fit)
            lAll,mnew=Model._mix_likelihoods(R,Rm,noiseCov,stimweights,self.model.bLeaveOneOut,self.model,mstate,RVar,Y)
            return Model._with_within(lAll,R,Rm,noiseCov,stimweights,yRef,Y,self.model,self.model.bLeaveOneOut,RVar),mnew
        noiseCorr=self.nrn.corr_matrix(R.shape[0],R.dtype) if self.nrn.corrType=='corr' else None
        if nbr is not None:
            # full AMA from the nearest references and a random tail of each category
            tail=Model._ref_subset(rng_ref,stimweights,min(self.model.nTail,stimweights.shape[0]))
            obsIdx=None
            if obs is not None:
                obsIdx=obs[0]
                R=jnp.take_along_axis(R,jnp.broadcast_to(obsIdx,R.shape[:-2]+obsIdx.shape),axis=-2)
                nbr=nbr[obsIdx,jnp.arange(nbr.shape[1])[None,:]]                 # [ mMax x nCtg x nCtg x K ]
            return Model._full_neighbors(R,Rm,RVar,noiseCorr,stimweights,nbr,tail,self.model.bLeaveOneOut,yRef,Y,self.model._Yperiod,
                                         obsIdx)
        if rng_ref is not None:
            # full AMA with nRef: decode against a random subset of the reference stimuli
            idx=Model._ref_subset(rng_ref,stimweights,self.model.nRef)
            take=lambda A: jnp.take_along_axis(A,jnp.broadcast_to(idx,A.shape[:-2]+idx.shape),axis=-2)
            return self.model.lrn_main(R,take(Rm),take(RVar),noiseCov,noiseCorr,take(stimweights),Y,idx,
                                       None if yRef is None else _take_stim(yRef,idx))
        return self.model.lrn_main(R,Rm,RVar,noiseCov,noiseCorr,stimweights,Y,None,yRef)

    def _bRefSubset(self):
        return self.model.modelType=='full' and self.model.nRef is not None

    def _bNeighbors(self):
        return self.model.modelType=='full' and self.model.nNeighbors is not None

    def _likelihoods_heldout(self,obs_out,ref_out,refweights,Y,refy=None):
        # observed responses of other stimuli, decoded against the reference (training) stimuli; no leave-one-out
        R=_flatten_responses(self.model._response_fun(*obs_out)[0])
        _,Rm,RVar=[_flatten_responses(x) for x in self.model._response_fun(*ref_out)]
        noiseCov=self.nrn._corr_fun(RVar,refweights,self.nrn.rho)
        noiseCorr=self.nrn.corr_matrix(R.shape[0],R.dtype) if self.nrn.corrType=='corr' else None
        return self.model._model_fun(R,Rm,RVar,noiseCov,noiseCorr,refweights,False,self.model,Y,None,
                                     refy if self.model.bWithin else None)

    @partial(jit, static_argnames=['self'])
    def _loss_fun_lrn(self,params,rng_key,prepped,index,stimval,stimweights,yCtg,Y,mstate=None):
        # the reference subset has its own key, so the response noise draws do not change with nRef. mstate: the mixture
        # fit (Model bWarmEM), or the nearest references (nNeighbors), which come back unchanged
        # with batches, nNeighbors' mstate is (nearest references, batch indices, batch weights): the batch is decoded
        # against all stimuli, which set the prior
        rng_ref=jxrandom.fold_in(rng_key,7) if (self._bRefSubset() or self._bNeighbors()) else None
        bNbr=self._bNeighbors()
        obs=mstate[1:] if bNbr and isinstance(mstate,tuple) else None
        nbr=(mstate[0] if obs is not None else mstate) if bNbr else None

        def one(noise_key):
            lik=self._likelihoods(self.nrn.lrn_main(noise_key,stimval,params['f'],prepped,index,stimweights,params.get('p')),
                                  stimweights,Y,rng_ref,None if bNbr else mstate,self._yRef(yCtg),nbr,obs)
            if bNbr:
                lik,mnew=lik,mstate
            else:
                lik,mnew=lik if mstate is not None else (lik,None)
            lAll,Yc=self._lik_parts(lik)
            if obs is None:
                return self.objective.lrn_main(lAll,stimweights,yCtg,Y,None,Yc),mnew
            return self.objective.lrn_main(lAll,obs[1],_take_stim(yCtg,obs[0]),Y,stimweights,Yc),mnew

        noise_key=jxrandom.key(self.model.noiseSeed) if self.model.bFixedNoise else rng_key
        cost,mnew=self._noise_average(one,noise_key)
        if self.objective.regWeight>0 and str(self.objective.regType).lower()!='none':
            cost=cost+self.objective.regWeight*self._penalty(self.nrn.insert(params['f'],prepped,index))
        return cost if mstate is None else (cost,mnew)

    def _noise_average(self,fun,key):
        """
        fun(key) -> (cost, aux), averaged over Model nNoiseSamples noisy observations (independent keys); aux of the first.
        One observation when nNoiseSamples is 1
        """
        S=self.model.nNoiseSamples
        if S<=1:
            return fun(key)
        keys=jxrandom.split(key,S)
        # full AMA's memory is bounded per chunk: its samples run one after the other, the others' in one batch
        cost,aux=lax.map(fun,keys) if self.model.modelType=='full' else jax.vmap(fun)(keys)
        return jnp.mean(cost),tree_util.tree_map(lambda a: a[0],aux)

    @partial(jit, static_argnames=['self'])
    def _nbr_state(self,params,prepped,index,stimval,stimweights):
        # the nearest references of every stimulus (nNeighbors) under the current filters, from the mean responses
        out=self.nrn.lrn_main(jxrandom.key(0),stimval,params['f'],prepped,index,stimweights,params.get('p'))
        _,Rm,RVar=[_flatten_responses(x) for x in self.model._response_fun(*out)]
        noiseCorr=self.nrn.corr_matrix(Rm.shape[0],Rm.dtype) if self.nrn.corrType=='corr' else None
        return Model._full_search(Rm,Rm,RVar,noiseCorr,stimweights,self.model.nNeighbors,self.model.bLeaveOneOut)

    def _mix_state0(self,f0):
        # the cold mixture fit at the initial filters, where a warm-started EM (Model bWarmEM) begins
        if self.model.modelType!='mix' or not self.model.bWarmEM:
            return None
        out=self.nrn.lrn_main(self.rng,self.stim.val,f0,self.filter.prepped_jx,self.filter._insert_index_jx,self.stim.weights,
                              self._p0() if self.nrn.bParams else None)
        Rm=_flatten_responses(self.model._response_fun(*out)[1])
        return tuple(lax.stop_gradient(v) for v in Model._mix_fit(Rm,self.stim.weights,self.model,bDetail=True,Y=self.stim.Y)[1]['state'])

    @partial(jit, static_argnames=['self'])
    def _loss_fun(self,params,rng_key,stimval,stimweights,yCtg,Y,W=None):
        def one(key):
            lAll,Yc=self._lik_parts(self._likelihoods(self.nrn.main(key,stimval,params['f'],stimweights,W,params.get('p')),stimweights,Y,
                                                      yRef=self._yRef(yCtg)))
            return self.objective.lrn_main(lAll,stimweights,yCtg,Y,None,Yc),None
        return self._noise_average(one,rng_key)[0]

    #- plot
    def plot_out(self,bFourier=None,name='f_out'):
        self.filter.plot_out(bFourier=bFourier,name=name)

    def plot_last(self,bFourier=None,name='f_last'):
        self.filter.plot_last(bFourier=bFourier,name=name)

    def _bank_info(self,bankInfo=None):
        """filter-bank metadata for the figures: multiscale_values() or parametric_values(), else bankInfo (a dict with any of
        k_peak, orientations, interleaved, mother, log_freq_knots, nMothers, mother_index), else None"""
        info=None
        for attr in ('multiscale_out','param_out'):
            if getattr(self,attr,None) is not None:
                info=dict(getattr(self,attr))
                break
        if info is None and bankInfo is not None:
            info=dict(bankInfo)
        if info is None:
            return None
        n=int(self.filter.n)
        if 'k_peak' in info and np.size(info['k_peak'])!=n:
            info.pop('k_peak')
        M=info.get('mother')
        info.setdefault('nMothers',1 if M is None or np.ndim(M)==1 else np.shape(M)[0])
        info.setdefault('mother_index',np.repeat(np.arange(info['nMothers']),n//info['nMothers']))
        return info

    def _figure_title(self,name,what):
        name=self.name if name is None else name
        return what if not name else str(name) + '\n' + what

    def plot_filter_bank(self,fname=None,name=None,bankInfo=None,perRow=8,dpi=110):
        """
        figure of the learned filter bank. 1D: one row of implied spatial filters per mother filter (real part solid,
        quadrature pair dotted, in that mother's colour), then a log-frequency and a linear-frequency row in which each
        column (scale) overlays the amplitude spectra of all mothers at that scale; side panels with the pooling weights
        (readoutType resultant/resultant_only, bars in the mother colours), the mother filter(s) of a multiscale bank
        (amplitude solid and phase/pi dotted against log(k/k_peak)), and the pooling-weighted sum of the spectra of each
        scale on linear and log frequency axes. 2D: real part and amplitude spectrum (linear frequency axes) of every
        filter, with peak frequency, orientation and interleaving in the panel titles. name (default unit.name) titles the
        figure; bankInfo supplies bank metadata for units without multiscale_values()/parametric_values(). Saves to fname
        when given; returns the figure.
        """
        flt=self.filter
        dims=tuple(flt.pix_dims)
        g=np.asarray(flt.implied_spatial())
        parts=None
        if flt.bSplit:                                                    # one column per sub-filter (e.g. per eye)
            nSp=int(flt.nSplit)
            g=np.reshape(g,tuple(flt.pix_dims)+(-1,))
            parts=np.tile(np.arange(nSp),g.shape[-1]//nSp) if g.shape[-1]%nSp==0 else None
        n=g.shape[-1]
        F=np.asarray(self.out).reshape(dims+(n,)) if flt.bIsFourier and not flt.bSplit else None
        w=None if self.pool_p is None else np.asarray(Nrn.pool_weights(np.asarray(self.pool_p,float).ravel()))
        mv=self._bank_info(bankInfo)
        per_row=perRow if n%perRow==0 else min(n,perRow)
        rows=int(np.ceil(n/per_row))
        kp=lambda j: '' if mv is None or 'k_peak' not in mv else ' k=%.3f'%np.asarray(mv['k_peak'])[j]
        if len(dims)==1:
            N=dims[0]
            x=np.arange(N)-N/2
            f=np.fft.fftshift(np.fft.fftfreq(N))
            pos=f>0
            amp=np.abs(F) if F is not None else np.abs(np.fft.fftshift(np.fft.fft(np.real(g),axis=0),axes=0))
            nM=int(mv['nMothers']) if mv is not None else 1
            nS=max(1,n//nM)
            midx=np.asarray(mv['mother_index']) if mv is not None else np.zeros(n,int)
            colors=[cm.tab10(i%10) for i in range(nM)]
            # rows: one spatial row per mother, then the log- and linear-frequency rows (all mothers of a scale overlaid)
            fig=plt.figure(figsize=(2.2*nS+3.6,2.6*max(nM+2,4)+1.4))
            outer=fig.add_gridspec(1,2,width_ratios=[nS,1.7],wspace=0.18)
            gs=outer[0,0].subgridspec(nM+2,nS)                             # filters: nM spatial rows, then log and linear
            side=outer[0,1].subgridspec(4,1,hspace=0.45)                   # weights, mother(s), weighted sums (linear, log)
            for j in range(n):
                m_,s_=(int(midx[j]),j%nS) if mv is not None else (0,j)
                ax=fig.add_subplot(gs[m_,s_])
                ax.plot(x,np.real(g[:,j]),lw=0.8,color=colors[m_],label='real')
                ax.plot(x,np.imag(g[:,j]),lw=0.8,ls=':',color=colors[m_],label='quadrature')
                ttl='f'+str(j)+(' m'+str(m_) if nM>1 else '')+('' if parts is None else ' part'+str(parts[j]))+kp(j)
                if w is not None and w.size==n:
                    ttl+='\nw=%.2f'%w[j]
                ax.set_title(ttl,fontsize=6)
                ax.tick_params(labelsize=5)
                ax.set_xlabel('px',fontsize=5)
                if j==0:
                    ax.legend(fontsize=5)
            for s_ in range(nS):
                js=[j for j in range(n) if (int(midx[j]),j%nS)[1]==s_] if mv is not None else [s_]
                for row,(scale,xlab) in enumerate(((True,'cycles/px (log)'),(False,'cycles/px (linear)'))):
                    ax=fig.add_subplot(gs[nM+row,s_])
                    for j in js:
                        (ax.semilogx if scale else ax.plot)(f[pos],amp[pos,j],lw=0.8,color=colors[int(midx[j])])
                    ax.set_xlim(f[pos][0] if scale else 0,0.5)
                    ax.tick_params(labelsize=5)
                    ax.set_xlabel(xlab,fontsize=5)
                    if s_==0:
                        ax.set_ylabel('amplitude',fontsize=5)
            ax=fig.add_subplot(side[0])
            if w is not None:
                ax.bar(np.arange(w.size),w,color=[colors[int(midx[j])] for j in range(min(n,w.size))])
                ax.set_title('pooling weights',fontsize=7)
            else:
                ax.text(0.5,0.5,'no pooling readout',ha='center',va='center',fontsize=7)
                ax.axis('off')
            ax.tick_params(labelsize=6)
            ax=fig.add_subplot(side[1])
            if mv is not None and mv.get('mother') is not None and 'log_freq_knots' in mv:
                M=np.asarray(mv['mother'])
                M=M[None] if M.ndim==1 else M
                for im in range(M.shape[0]):
                    ax.plot(mv['log_freq_knots'],np.abs(M[im]),lw=1,color=colors[im])
                    ax.plot(mv['log_freq_knots'],np.angle(M[im])/np.pi*np.abs(M[im]).max(),lw=0.7,ls=':',color=colors[im])
                handles=[plt.Line2D([],[],color='k',lw=1,label='amplitude'),plt.Line2D([],[],color='k',lw=0.7,ls=':',label='phase')]
                handles+=[plt.Line2D([],[],color=colors[im],lw=1,label='mother '+str(im)) for im in range(M.shape[0])]
                ax.legend(handles=handles,fontsize=5)
                ax.set_xlabel('log(k / k_peak)',fontsize=6)
                ax.set_title('mother filter(s)',fontsize=7)
            else:
                for j in range(n):
                    ax.semilogx(f[pos],amp[pos,j],lw=0.7,color=colors[int(midx[j])])
                ax.set_title('all amplitude spectra (log frequency)',fontsize=7)
                ax.set_xlabel('cycles/px (log)',fontsize=6)
            ax.tick_params(labelsize=6)
            # weighted sum of each scale (pooling weights; uniform when there is no readout), linear above log
            ww=np.ones(n)/n if w is None or w.size!=n else w
            scale_sum=np.stack([sum(ww[j]*amp[:,j] for j in range(n) if (j%nS)==s_) for s_ in range(nS)],1)
            scolors=[cm.viridis(i/max(nS-1,1)) for i in range(nS)]
            for row,scale in ((2,False),(3,True)):
                ax=fig.add_subplot(side[row])
                for s_ in range(nS):
                    (ax.semilogx if scale else ax.plot)(f[pos],scale_sum[pos,s_],lw=0.8,color=scolors[s_],
                                                        label=('scale '+str(s_)) if not scale else None)
                (ax.semilogx if scale else ax.plot)(f[pos],scale_sum[pos].sum(1),lw=1.1,color='k',label='total' if not scale else None)
                ax.set_xlim(f[pos][0] if scale else 0,0.5)
                ax.set_title('weighted sum per scale ('+('log' if scale else 'linear')+' frequency)',fontsize=7)
                ax.set_xlabel('cycles/px ('+('log' if scale else 'linear')+')',fontsize=6)
                ax.tick_params(labelsize=6)
                if not scale:
                    ax.legend(fontsize=5,ncol=2)
        elif len(dims)==2:
            fig=plt.figure(figsize=(1.6*per_row+2.5,3.4*rows+1.2))
            gs=fig.add_gridspec(2*rows,per_row+1)
            for j in range(n):
                r_,c_=divmod(j,per_row)
                ax=fig.add_subplot(gs[2*r_,c_])
                v=np.real(g[...,j])
                lim=np.abs(v).max()
                ax.imshow(v,cmap='RdBu_r',vmin=-lim,vmax=lim)
                ori='' if mv is None or 'orientations' not in mv else ' %d deg'%round(np.degrees(np.asarray(mv['orientations'])[j]))
                il='*' if mv is not None and 'interleaved' in mv and np.asarray(mv['interleaved'])[j] else ''
                ax.set_title('f'+str(j)+il+kp(j)+ori,fontsize=5)
                ax.axis('off')
                ax=fig.add_subplot(gs[2*r_+1,c_])
                A=np.abs(F[...,j]) if F is not None else np.abs(np.fft.fftshift(np.fft.fft2(v)))
                ax.imshow(A,cmap='magma',extent=(-0.5,0.5,0.5,-0.5))
                ax.set_xticks([-0.5,0,0.5])
                ax.set_yticks([-0.5,0,0.5])
                ax.tick_params(labelsize=4)
                if j==0:
                    ax.set_xlabel('cycles/px (linear)',fontsize=5)
            ax=fig.add_subplot(gs[:,per_row])
            if w is not None:
                ax.barh(np.arange(w.size),w)
                ax.set_title('pooling weights',fontsize=7)
            else:
                ax.text(0.5,0.5,'no pooling readout\n* = interleaved\nrows: real part,\none-sided |spectrum|',ha='center',va='center',fontsize=7)
                ax.axis('off')
        else:
            raise Exception('plot_filter_bank supports 1D and 2D filters')
        fig.suptitle(self._figure_title(name,'filters'),fontsize=9)
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname,dpi=dpi)
        return fig

    @staticmethod
    def _embed(feats,method,seed,**kw):
        """2D embedding of [ nStim x nFeatures ]: 't-sne' / 'tsne' (sklearn), 'pacmap', or 'phate'"""
        method=method.lower().replace('-','')
        if method=='tsne':
            from sklearn.manifold import TSNE
            kw.setdefault('perplexity',30)
            return TSNE(n_components=2,init='pca',learning_rate='auto',random_state=seed,**kw).fit_transform(feats),'t-SNE'
        if method=='pacmap':
            import pacmap
            kw.setdefault('n_neighbors',10)
            return pacmap.PaCMAP(n_components=2,random_state=seed,**kw).fit_transform(feats,init='pca'),'PaCMAP'
        if method=='phate':
            import phate
            kw.setdefault('knn',10)
            return phate.PHATE(n_components=2,random_state=seed,verbose=0,**kw).fit_transform(feats),'PHATE'
        raise Exception("method must be 'tsne', 'pacmap' or 'phate'")

    def _response_features(self,stim,nMax,seed,cmap=None):
        """(features [ nStim x nFeatures ], responses u [ nF x nStim ], latent [ nStim ], cmap) for the embeddings:
        noise-free responses of every filter to up to nMax category-balanced stimuli, as unit phasors for normalizeType
        'phase'"""
        st=copy.copy(stim)
        if st.bIsFourier:
            st._ifft()
        if self.filter.bSplit and not st.bIsSplit:
            st.nSplit=int(self.filter.nSplit)                              # stimuli built without nSplit: the filters know it
            st.split()
        elif st.bIsSplit and not self.filter.bSplit:
            st.unsplit()
        rng=np.random.default_rng(seed)
        valid=np.asarray(st.weights)>0                                    # [ nStim_Ctg x nCtg ]
        per=max(1,nMax//st.nCtg)
        cols=[]
        for c in range(st.nCtg):
            idx=np.flatnonzero(valid[:,c])
            k=min(per,idx.size)
            cols.append(np.stack([rng.choice(idx,k,replace=False),np.full(k,c)],1))
        sel=np.concatenate(cols)
        Yv=np.asarray(st.Y) if np.ndim(st.Y)==1 else np.asarray(st.Y)[:,0]      # color by the first latent dimension
        lat=Yv[sel[:,1]]
        g=np.asarray(self.filter.implied_spatial())
        if self.filter.bSplit:
            # each sub-filter (e.g. one eye) responds as its own neuron: [ nSplit x nF ] response dimensions
            nS=int(self.filter.nSplit)
            nP=st.nPix//nS
            g=g.reshape(nP,nS,-1)
            S=np.real(np.asarray(st.val).reshape(nP,nS,st.nStim_Ctg,st.nCtg)[:,:,sel[:,0],sel[:,1]])
            r=np.einsum('psj,psn->sjn',g,S).reshape(-1,S.shape[-1])
        else:
            S=np.real(np.asarray(st.val).reshape(st.nPix,st.nStim_Ctg,st.nCtg)[:,sel[:,0],sel[:,1]])
            r=g.reshape(st.nPix,-1).T@S                                   # [ nF x nStim ]
        u=r/np.maximum(np.abs(r),1e-12) if self.nrn.normalizeType=='phase' else r
        feats=(np.concatenate([u.real,u.imag],0) if np.iscomplexobj(u) else u).T
        if cmap is None:
            bCirc=getattr(st,'Yperiod',None) is not None and st.Yperiod[0] is not None
            cmap='twilight' if bCirc or (Yv.min()>=0 and Yv.max()<2*np.pi+1e-9 and np.ptp(Yv)>np.pi) else 'viridis'
        return feats,u,lat,cmap

    def plot_response_embeddings(self,stim,methods=('tsne','pacmap','phate'),fname=None,name=None,nMax=2000,seed=0,cmap=None,
                                 dpi=110,method_kw=None,**embed_kw):
        """
        one figure with every embedding of the same responses side by side (t-SNE, PaCMAP, PHATE by default), and, with
        pooling weights, the pooled resultant as a final panel. The responses (and the colour scale) are computed once, so
        the panels are comparable. method_kw: {method: its own keyword arguments}; embed_kw apply to every method.
        name (default unit.name) titles the figure. Saves to fname when given; returns the figure.
        """
        feats,u,lat,cmap=self._response_features(stim,nMax,seed,cmap)
        w=None if self.pool_p is None else np.asarray(Nrn.pool_weights(np.asarray(self.pool_p,float).ravel()))
        bPooled=w is not None and w.size==u.shape[0] and np.iscomplexobj(u)
        method_kw=method_kw or {}
        embs=[self._embed(feats,m,seed,**{**embed_kw,**method_kw.get(m,{})}) for m in methods]
        nax=len(embs)+(1 if bPooled else 0)
        fig,axes=plt.subplots(1,nax,figsize=(5.2*nax,5.3),squeeze=False)
        for i,(emb,label) in enumerate(embs):
            sc=axes[0,i].scatter(emb[:,0],emb[:,1],c=lat,cmap=cmap,s=4)
            axes[0,i].set_title(label,fontsize=9)
            axes[0,i].set_xticks([])
            axes[0,i].set_yticks([])
            if i==len(embs)-1:
                fig.colorbar(sc,ax=axes[0,i],label='latent')
        if bPooled:
            z=(w[:,None]*u).sum(0)
            axes[0,-1].scatter(z.real,z.imag,c=lat,cmap=cmap,s=4)
            axes[0,-1].set_aspect('equal')
            axes[0,-1].set_title('pooled resultant sum_j p_j r_j',fontsize=9)
            axes[0,-1].tick_params(labelsize=6)
        fig.suptitle(self._figure_title(name,('phase-normalized ' if self.nrn.normalizeType=='phase' else '')
                                        +'responses of '+str(u.shape[0])+' filters, '+str(feats.shape[0])+' stimuli'),fontsize=9)
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname,dpi=dpi)
        return fig

    def plot_response_embedding(self,stim,method='tsne',fname=None,name=None,nMax=2000,seed=0,cmap=None,dpi=110,**embed_kw):
        """
        2D embedding of the noise-free responses of every filter to up to nMax category-balanced stimuli (a Stim, e.g.
        held-out test stimuli; fourier-domain stimuli are transformed back), coloured by the latent value. method: 'tsne'
        (sklearn t-SNE, perplexity 30, PCA initialization), 'pacmap' (PaCMAP, n_neighbors 10, PCA initialization) or 'phate'
        (PHATE, knn 10); embed_kw go to the method. Responses are unit phasors (real and imaginary parts of r_j / |r_j|) for
        normalizeType 'phase', else the responses themselves. With pooling weights (readoutType resultant/resultant_only) a
        second panel shows the pooled resultant sum_j p_j r_j (what the likelihood sees) in the complex plane. cmap defaults
        to 'twilight' (circular latents) when Y spans more than pi within [0, 2 pi), else 'viridis'. name (default
        unit.name) titles the figure. Saves to fname when given; returns the figure.
        """
        feats,u,lat,cmap=self._response_features(stim,nMax,seed,cmap)
        emb,label=self._embed(feats,method,seed,**embed_kw)
        w=None if self.pool_p is None else np.asarray(Nrn.pool_weights(np.asarray(self.pool_p,float).ravel()))
        bPooled=w is not None and w.size==u.shape[0] and np.iscomplexobj(u)
        fig,axes=plt.subplots(1,2 if bPooled else 1,figsize=(11 if bPooled else 6,5.3),squeeze=False)
        sc=axes[0,0].scatter(emb[:,0],emb[:,1],c=lat,cmap=cmap,s=4)
        axes[0,0].set_title(label+' of '+('phase-normalized ' if self.nrn.normalizeType=='phase' else '')+'responses ('
                            +str(u.shape[0])+' filters, '+str(feats.shape[0])+' stimuli)',fontsize=8)
        axes[0,0].set_xticks([])
        axes[0,0].set_yticks([])
        fig.colorbar(sc,ax=axes[0,0],label='latent')
        if bPooled:
            z=(w[:,None]*u).sum(0)
            axes[0,1].scatter(z.real,z.imag,c=lat,cmap=cmap,s=4)
            axes[0,1].set_aspect('equal')
            axes[0,1].set_title('pooled resultant sum_j p_j r_j (what the likelihood sees)',fontsize=8)
            axes[0,1].tick_params(labelsize=6)
        fig.suptitle(self._figure_title(name,'response '+label),fontsize=9)
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname,dpi=dpi)
        return fig

    def plot_response_tsne(self,stim,fname=None,name=None,**kw):
        """plot_response_embedding with method 'tsne'"""
        return self.plot_response_embedding(stim,'tsne',fname=fname,name=name,**kw)

    def plot_response_pacmap(self,stim,fname=None,name=None,**kw):
        """plot_response_embedding with method 'pacmap'"""
        return self.plot_response_embedding(stim,'pacmap',fname=fname,name=name,**kw)

    def plot_response_phate(self,stim,fname=None,name=None,**kw):
        """plot_response_embedding with method 'phate'"""
        return self.plot_response_embedding(stim,'phate',fname=fname,name=name,**kw)

    def save_figures(self,stem,stim,name=None,bankInfo=None,methods=('tsne','pacmap','phate'),**embed_kw):
        """
        write <stem>_filters.png (plot_filter_bank) and <stem>_<method>.png (plot_response_embedding on stim) for each
        embedding method; closes the figures. With more than one method it also writes <stem>_embeddings.png, all of them in
        one figure (plot_response_embeddings). methods: a sequence of method names, or a dict of method -> its own keyword
        arguments (embed_kw, applying to every method, is for arguments they share, e.g. nMax and seed).
        """
        plt.close(self.plot_filter_bank(str(stem)+'_filters.png',name=name,bankInfo=bankInfo))
        items=dict(methods) if isinstance(methods,dict) else {m:{} for m in methods}
        for m,kw in items.items():
            plt.close(self.plot_response_embedding(stim,m,str(stem)+'_'+m.replace('-','')+'.png',name=name,**{**embed_kw,**kw}))
        if len(items)>1:
            plt.close(self.plot_response_embeddings(stim,tuple(items),str(stem)+'_embeddings.png',name=name,method_kw=items,**embed_kw))



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

        from sklearn.manifold import TSNE
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


__all__=['Unit', 'Response']
