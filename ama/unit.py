"""The user entry point (Unit), which wires Stim, Nrn, Model, Objective and Optimizer together."""
from ._base import *
from .model import Model
from .banks import _Banks
from .evaluation import _Evaluation
from .persist import _Persistence
from .plotting import _Plotting
from .response import Response


class Unit(_Static,_Banks,_Evaluation,_Persistence,_Plotting):
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
        if last is not None and self.nrn.bFinalized and (bFourier!=self.nrn.bFourier or bool(bAnalytic)!=bool(self.nrn.bAnalytic)
                                                         or bool(bSplit)!=bool(self.nrn.bSplit)):
            # the kept filters are parameters of the current domain and layout, which the new one would misread
            raise Exception('train_append and train_recurse keep the current filters: they can not change fourierType or '
                            'bSplit (train_new can)')

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
        if self.model.bWithin and self.model.modelType!='full':
            # the within-category regression uses each category's own (unpooled) covariance, left out with bLeaveOneOut
            nMin=3 if self.model.bLeaveOneOut else 2
            if np.any(counts<nMin):
                raise Exception('bWithin needs at least ' + str(nMin) + ' stimuli in every category for its within-category '
                                'regression (fewest: ' + str(int(counts.min())) + ')')
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
        return (tuple(shape),cols,str(self.optimizer.optimizerType),np.dtype(self.nrn.dtype).name,
                getattr(self.optimizer,'lbfgsMemory',None))

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
        p_start=(self.pool_p,self.nrn_p)       # every restart starts its response-model parameters from here
        for i in range(nRestarts):
            if i>0:
                rng=jxrandom.fold_in(rng,i)
            rng,rng_key = jxrandom.split(rng)
            f0=self.filter.get_f0(rng_key,self.optimizer._f0_jxrand_fun)
            self.pool_p,self.nrn_p=p_start
            self._run(f0,rng,stimVal=stimVal)
            if nRestarts==1:
                self.restart_costs=None
                return
            cost=float(self.loss) if stimVal is None else self.evaluate(stimVal)
            self.restart_costs.append(cost)
            if best is None or cost<best[0]:
                best=(cost,jnp.asarray(self.filter.out),self.out_params,self.opt_state,self.rng_last,
                      list(self.optimizer.loss_hist),list(self.optimizer.val_hist),self.optimizer.best_step,
                      self._opt_param_shape,self._opt_state_key,self.pool_p,self.nrn_p,getattr(self.optimizer,'stop_reason',None))
        (_,self.filter.out,self.out_params,self.opt_state,self.rng_last,self.optimizer.loss_hist,self.optimizer.val_hist,
         self.optimizer.best_step,self._opt_param_shape,self._opt_state_key,self.pool_p,self.nrn_p,
         self.optimizer.stop_reason)=best

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
        return self._lrn_model().lrn_main(R,Rm,RVar,noiseCov,noiseCorr,stimweights,Y,None,yRef)

    def _lrn_model(self):
        # the Model that scores the training stimuli. With a noise variance that does not depend on the response (fano 0,
        # no stage-1 noise), leaving a stimulus out of its category's noise covariance (bLooNoise) changes nothing, so
        # the same Model with bLooNoise False is exact, and takes the rank-one / low-rank leave-one-out paths
        m=self.model
        if not (m.bLeaveOneOut and m.bLooNoise and self.nrn.fano==0 and not self.nrn.bNoise_1):
            return m
        cache=getattr(self,'_lrn_model_cache',None)
        if cache is None or cache[0]!=m._key():
            mm=m.copy()
            mm.bLooNoise=False
            mm._Yperiod=m._Yperiod
            mm._bGeometrySet=True
            cache=self._lrn_model_cache=(m._key(),mm)
        return cache[1]

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


__all__=['Unit']
