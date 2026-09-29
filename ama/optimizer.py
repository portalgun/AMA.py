"""The optimization loop (Optimizer) and the AMA-SGD update."""
from ._base import *


def _ama_sgd(step0,stepMin,stepDecay):
    """
    the AMA-SGD step of Burge & Jaini (2017) and burgelab/AMA (gradSGD.m, updateSGD.m, amaR01sgdObjFunc.m) as an optax
    transformation: each filter moves a fixed distance eps along its unit-normalized (tangent-plane) gradient, with eps
    = max(stepMin, step0*(1-stepDecay)^t) at iteration t. Filters are the last axis of params['f']; other parameters
    (pooling weights) take the same step along their normalized gradient. Accepting a step only when it does not raise
    the batch cost happens in Optimizer._run_chunk.
    """
    def unit(g,bPerFilter):
        axes=tuple(range(g.ndim-1)) if bPerFilter and g.ndim>1 else None
        n=jnp.sqrt(jnp.sum(jnp.abs(g)**2,axis=axes,keepdims=bPerFilter and g.ndim>1))
        return g/jnp.where(n>0,n,1)

    def init(params):
        return {'count':jnp.zeros((),dtype=jnp.int32)}

    def update(grads,state,params=None):
        eps=jnp.maximum(stepMin,step0*(1-stepDecay)**state['count'])
        updates={k: -(eps*unit(g,k=='f')).astype(g.dtype) for k,g in grads.items()}
        return updates,{'count':state['count']+1}

    return optax.GradientTransformation(init,update)


class Optimizer():
    def __init__(self,optimizerType='adam',projectionType=['l2_sphere',1],lRate0=1e-1,nIterMax=1000,f0_jxrand_fun=['ball',1],
                 batchSize=None,nStepsPerChunk=100,bVerbose=True,nBatchMinCtg=2,patience=None,stepMin=0.,stepDecay=0.01,
                 tolFun=None,tolX=None,lbfgsMemory=10):
        """
        optimizerType  - any optax optimizer (e.g. 'adam', 'sgd'), with learning rate lRate0 (a number or an optax
                         schedule), or 'ama_sgd': the step of Burge & Jaini (2017) and burgelab/AMA. Each filter moves
                         lRate0*(1-stepDecay)^t (at least stepMin) along its unit-normalized tangent-plane gradient, and a
                         step is kept only if it does not raise the cost of the batch it was computed on (evaluated again,
                         with the same noise). Needs projectionType 'l2_sphere'. With batchSize, this is AMA-SGD.
                         'lbfgs': limited-memory BFGS with a zoom line search (optax.lbfgs, lbfgsMemory pairs), the
                         quasi-Newton method of matlab's fminunc/fmincon: full batch (not with batchSize), and lRate0 is
                         not used. The cost is evaluated at the normalized filters, so the unit-norm constraint is exact
                         and the line search moves along the sphere's tangent space.
        stepMin, stepDecay - 'ama_sgd' only: the smallest step, and the fraction the step shrinks each iteration (0.01:
                         1% per iteration, as in the paper)
        tolFun, tolX   - fmincon-like stopping tolerances, checked after every chunk (nStepsPerChunk iterations): stop when
                         the cost changed by less than tolFun*(1+|cost|) over the chunk (first vs last iteration), or when
                         no parameter moved by more than tolX. stop_reason records why training stopped ('nIterMax',
                         'patience', 'tolFun', 'tolX'). With batches or sampled noise the cost is noisy, so tolFun needs a
                         matching size
        batchSize      - None for full-batch learning, or the approximate number of stimuli per iteration (AMA-SGD,
                         Burge & Jaini 2017). Each iteration draws a new random batch, stratified so every category keeps
                         its share of the training set (the prior) with at least nBatchMinCtg stimuli. Posteriors are computed
                         against the batch, so full AMA costs O(batchSize^2) per iteration. loss_hist holds batch costs.
        nStepsPerChunk - iterations compiled into one lax.scan; the loss is reported once per chunk
        nBatchMinCtg   - minimum stimuli per category in a batch (AMA-Gauss needs more than the number of response dimensions)
        patience       - early stopping, when training is given validation stimuli (stimVal in Unit.train_*): the cost on
                         them is computed after every chunk (nStepsPerChunk iterations), training stops after this many
                         chunks without improvement (None = never stop early), and the filters with the lowest validation
                         cost are kept. val_hist holds the validation costs, best_step the kept iteration.
        """
        # filters are constrained to unit length, ||f||=1 (Burge & Jaini 2017)
        self.optimizerType=optimizerType
        self.projectionType=list(projectionType)
        self.lRate0=lRate0
        self.nIterMax=nIterMax
        self.f0_jxrand_fun=list(f0_jxrand_fun)
        self._f0_jxrand_fun=list(f0_jxrand_fun)
        if isinstance(f0_jxrand_fun[0],str):
            self._f0_jxrand_fun[0]=getattr(jxrandom,f0_jxrand_fun[0])
        self.batchSize=batchSize
        self.nBatchMinCtg=nBatchMinCtg
        self.nStepsPerChunk=nStepsPerChunk
        self.bVerbose=bVerbose
        self.patience=patience
        self.stepMin=float(stepMin)
        self.stepDecay=float(stepDecay)
        self.tolFun=None if tolFun is None else float(tolFun)
        self.tolX=None if tolX is None else float(tolX)
        self.lbfgsMemory=int(lbfgsMemory)
        if self.optimizerType=='lbfgs' and batchSize is not None:
            raise Exception("optimizerType='lbfgs' is a full-batch method; batchSize must be None")
        if self.optimizerType=='ama_sgd':
            if self.projectionType[0]!='l2_sphere':
                raise Exception("optimizerType='ama_sgd' steps on the unit sphere: projectionType must be 'l2_sphere'")
            if not isinstance(lRate0,(int,float)) or not lRate0>0 or not 0<=self.stepDecay<1 or self.stepMin<0:
                raise Exception("optimizerType='ama_sgd' needs a positive number lRate0, 0 <= stepDecay < 1 and stepMin >= 0")

    @property
    def _bAccept(self):
        # keep a step only if it does not raise the batch cost (AMA-SGD)
        return self.optimizerType=='ama_sgd'

    @property
    def _bLBFGS(self):
        return self.optimizerType=='lbfgs'

    def copy(self):
        return Optimizer(**_get_copy_dict(self,_OPT_EXCL))

    @property
    def _projection(self):
        #l2_sphere, l2_ball, l1_all, l1_sphere
        return getattr(optax.projections,'projection_' + self.projectionType[0])

    @property
    def _projection_params(self):
        return tuple(self.projectionType[1:])

    @property
    def optimizer(self):
        # the optax optimizer (or the AMA-SGD step) as a function of the learning rate
        if self.optimizerType=='ama_sgd':
            return lambda lRate: _ama_sgd(float(lRate),self.stepMin,self.stepDecay)
        if self._bLBFGS:
            return lambda lRate: optax.lbfgs(memory_size=self.lbfgsMemory)
        return getattr(optax,self.optimizerType)

    @property
    def tx(self):
        # one optax transformation per setting, shared by all Optimizers: a new transformation object would compile the
        # training step again (e.g. for every new Unit or cross-validation fold)
        if self.optimizerType=='ama_sgd':
            key=(self.optimizerType,self.lRate0,self.stepMin,self.stepDecay)
            if key not in _TX_CACHE:
                _TX_CACHE[key]=self.optimizer(self.lRate0)
            return _TX_CACHE[key]
        key=(self.optimizerType,self.lRate0) if not self._bLBFGS else (self.optimizerType,self.lbfgsMemory)
        if key not in _TX_CACHE:
            _TX_CACHE[key]=self.optimizer(self.lRate0)
        return _TX_CACHE[key]

    @staticmethod
    @partial(jit, static_argnames=['proj_fun','proj_params'])
    def insert_project_extract(params,prepped,index,proj_fun,proj_params):
        out=dict(params)
        fNew=prepped.at[index].set(params['f'])

        # constraint applies to each filter (last axis) separately
        vectorized_proj_fun = vmap(lambda f: proj_fun({'f': f}, *proj_params)['f'], in_axes=-1, out_axes=-1)
        fNew=vectorized_proj_fun(fNew)

        out['f']=fNew[index]
        return out

    #- batches
    def _batch_plan(self,weights):
        # number of stimuli drawn from each category per iteration, proportional to the category's size
        counts=np.asarray(weights).sum(0).astype(int)
        m=np.maximum(self.nBatchMinCtg,np.round(self.batchSize*counts/counts.sum())).astype(int)
        m=np.minimum(m,counts)
        mMax=int(m.max())
        mask=(np.arange(mMax)[:,None] < m[None,:]).astype(np.asarray(weights).dtype)  # [ mMax x nCtg ]
        return mMax,jnp.asarray(mask)

    @staticmethod
    def _batch_index(rng,mMax,mask,stimweights):
        # a random subset of the valid (non-padding) stimuli within each category: indices [ mMax x nCtg ] and weights
        score=jxrandom.uniform(rng,stimweights.shape) + jnp.where(stimweights>0,0.,jnp.inf)
        idx=jnp.argsort(score,axis=0)[:mMax]
        return idx,jnp.take_along_axis(stimweights,idx,axis=0)*mask

    @staticmethod
    def _sample_batch(rng,mMax,mask,stimval,stimweights,yCtg):
        idx,w=Optimizer._batch_index(rng,mMax,mask,stimweights)
        val=jnp.take_along_axis(stimval,jnp.broadcast_to(idx,stimval.shape[:-2]+idx.shape),axis=-2)
        y=_take_stim(yCtg,idx)
        return val,w,y

    #- learning
    @staticmethod
    def _tangent(g,f):
        # remove the radial component of each filter's gradient (last axis = filters; Burge & Jaini 2017, Eq 19).
        # Without this, optimizers that rescale coordinates (e.g. adam) turn the large radial component, which the
        # unit-sphere projection then discards, into tangent steps that can increase the cost.
        axes=tuple(range(f.ndim-1))
        radial=jnp.sum(jnp.real(jnp.conj(f)*g),axis=axes,keepdims=True)/jnp.sum(jnp.abs(f)**2,axis=axes,keepdims=True)
        return g-radial*f

    @staticmethod
    @partial(jit, static_argnames=['tx','loss_fun','proj_fun','proj_params','nSteps','mMax','bTangent','bAccept','bState','state_fun',
                                   'bLBFGS'])
    def _run_chunk(tx,loss_fun,proj_fun,proj_params,nSteps,mMax,bTangent,bAccept,bState,nActive,
                   params,opt_state,rng,prepped,prepped_exp,index,index_exp,batch_mask,stimval,stimweights,yCtg,Y,state_fun=None,
                   bLBFGS=False):
        if state_fun is not None:
            # a model state recomputed from the parameters at the start of every chunk (e.g. the nearest references)
            opt_state=(opt_state[0],state_fun(params,prepped,index,stimval,stimweights))
        # batches with a per-chunk state (nNeighbors): the loss gets all stimuli, which are the references, and decodes
        # the batch (indices and weights passed with the state)
        bRefBatch=state_fun is not None and mMax is not None

        def body(carry,i):
            params,opt_state,rng=carry
            rng,rng_batch,rng_key=jxrandom.split(rng,3)
            if mMax is None or bRefBatch:
                val,w,y=stimval,stimweights,yCtg
            else:
                val,w,y=Optimizer._sample_batch(rng_batch,mMax,batch_mask,stimval,stimweights,yCtg)

            # L-BFGS: the cost of the normalized filters (scale invariant), so its gradient and line search stay tangent
            norm=(lambda prm: Optimizer.insert_project_extract(prm,prepped_exp,index_exp,proj_fun,proj_params)) if bLBFGS else (lambda prm: prm)
            if bState:
                # a model state (e.g. the previous mixture fit) goes in with the parameters and comes out updated
                ost,mst=opt_state
                ext=(mst,)+Optimizer._batch_index(rng_batch,mMax,batch_mask,stimweights) if bRefBatch else mst
                cost=lambda prm: loss_fun(norm(prm),rng_key,prepped,index,val,w,y,Y,ext)
                (loss_value,mnew),grads=value_and_grad(cost,has_aux=True)(params)
                if bRefBatch:
                    mnew=mst
                value_fn=lambda prm: cost(prm)[0]
            else:
                ost=opt_state
                cost=lambda prm: loss_fun(norm(prm),rng_key,prepped,index,val,w,y,Y)
                loss_value,grads=value_and_grad(cost)(params)
                value_fn=cost
            # complex params: jax returns the conjugate of the ascent direction
            grads=tree_util.tree_map(jnp.conjugate,grads)
            if bTangent and not bLBFGS:
                grads=dict(grads,f=Optimizer._tangent(grads['f'],params['f']))
            if bLBFGS:
                # the line search's step sizes are float64 under jax x64: its trial points keep the parameters' precision
                cast=lambda prm: tree_util.tree_map(lambda a,q: a.astype(q.dtype),prm,params)
                updates,new_ost=tx.update(grads,ost,params,value=loss_value,grad=grads,value_fn=lambda prm: value_fn(cast(prm)))
                updates=cast(updates)
            else:
                updates,new_ost=tx.update(grads,ost,params)
            new_state=(new_ost,mnew) if bState else new_ost
            new_params=Optimizer.insert_project_extract(optax.apply_updates(params,updates),prepped_exp,index_exp,proj_fun,proj_params)
            if bAccept:
                # AMA-SGD: keep the step only if the cost of this batch (same noise) does not increase
                new_value=cost(new_params)
                ok=(new_value[0] if bState else new_value)<=loss_value
                new_params=tree_util.tree_map(lambda a,b: jnp.where(ok,a,b),new_params,params)

            # iterations past nIterMax in the last chunk leave the state unchanged
            keep=lambda new,old: tree_util.tree_map(lambda a,b: jnp.where(i<nActive,a,b),new,old)
            return (keep(new_params,params),keep(new_state,opt_state),rng),loss_value

        (params,opt_state,rng),losses=lax.scan(body,(params,opt_state,rng),jnp.arange(nSteps))
        return params,opt_state,rng,losses

    @staticmethod
    @partial(jit, static_argnames=['tx','loss_fun','nSteps','mMax','bLBFGS'])
    def _run_generated_chunk(tx,loss_fun,nSteps,mMax,nActive,params,opt_state,rng,batch_mask,stimval,stimweights,yCtg,Y,bLBFGS=False):
        # as _run_chunk, for parameters of generated filters (no filter insertion, projection or tangent step)
        def body(carry,i):
            params,opt_state,rng=carry
            rng,rng_batch,rng_key=jxrandom.split(rng,3)
            if mMax is None:
                val,w,y=stimval,stimweights,yCtg
            else:
                val,w,y=Optimizer._sample_batch(rng_batch,mMax,batch_mask,stimval,stimweights,yCtg)
            loss_value,grads=value_and_grad(loss_fun)(params,rng_key,val,w,y,Y)
            grads=tree_util.tree_map(jnp.conjugate,grads)
            if bLBFGS:
                cast=lambda prm: tree_util.tree_map(lambda a,q: a.astype(q.dtype),prm,params)
                updates,new_state=tx.update(grads,opt_state,params,value=loss_value,grad=grads,
                                            value_fn=lambda prm: loss_fun(cast(prm),rng_key,val,w,y,Y))
                updates=cast(updates)
            else:
                updates,new_state=tx.update(grads,opt_state,params)
            new_params=optax.apply_updates(params,updates)
            keep=lambda new,old: tree_util.tree_map(lambda a,b: jnp.where(i<nActive,a,b),new,old)
            return (keep(new_params,params),keep(new_state,opt_state),rng),loss_value

        (params,opt_state,rng),losses=lax.scan(body,(params,opt_state,rng),jnp.arange(nSteps))
        return params,opt_state,rng,losses

    def minimize_generated(self,params,rng,stim,loss_fun,val_fun=None):
        """minimize loss_fun (a _GeneratedLoss) over params, in the same chunks, batches and early stopping as minimize"""
        if self.optimizerType=='ama_sgd':
            raise Exception("optimizerType='ama_sgd' learns unit-norm filters directly; use another optimizer for generated filters")
        tx=self.tx
        mMax,batch_mask=(None,None) if self.batchSize is None else self._batch_plan(stim.weights)
        nSteps=max(1,min(self.nStepsPerChunk,self.nIterMax))
        run=lambda params,opt_state,rng,nActive: self._run_generated_chunk(tx,loss_fun,nSteps,mMax,nActive,params,opt_state,rng,
                                                                          batch_mask,stim.val,stim.weights,stim.yCtg,stim.Y,
                                                                          self._bLBFGS)
        params,_,rng=self._loop(run,nSteps,params,tx.init(params),rng,val_fun)
        return params,rng

    def _loop(self,run,nSteps,params,opt_state,rng,val_fun):
        # chunks of nSteps iterations (a fixed length keeps one compiled trace; the last chunk masks its extra
        # iterations), with early stopping on val_fun: best_step is the number of iterations of the kept parameters
        self.loss_hist=[]
        self.val_hist=[]
        self.best_step=None
        self.stop_reason='nIterMax'
        best=None
        nBad=0
        step=0
        tolFun,tolX=getattr(self,'tolFun',None),getattr(self,'tolX',None)
        while step < self.nIterMax:
            nActive=min(nSteps,self.nIterMax-step)
            prev=params
            params,opt_state,rng,losses=run(params,opt_state,rng,nActive)
            chunk=np.asarray(losses)[:nActive]
            self.loss_hist.extend(chunk.tolist())
            step+=nActive

            if self.bVerbose:
                print(f'step {step-1}, loss: {self.loss_hist[-1]}')

            if val_fun is not None:
                v=float(val_fun(params))
                self.val_hist.append(v)
                if best is None or v < best[0]:
                    best=(v,params,opt_state,step)
                    nBad=0
                else:
                    nBad+=1
                if self.bVerbose:
                    print(f'step {step-1}, validation loss: {v}')
                if self.patience is not None and nBad >= self.patience:
                    self.stop_reason='patience'
                    break

            # fmincon-like tolerances on the cost change and the parameter step over the chunk
            if tolFun is not None and len(chunk)>1 and abs(chunk[-1]-chunk[0])<=tolFun*(1+abs(chunk[-1])):
                self.stop_reason='tolFun'
                break
            if tolX is not None:
                dx=max(float(jnp.max(jnp.abs(a-b))) for a,b in zip(tree_util.tree_leaves(params),tree_util.tree_leaves(prev)))
                if dx<=tolX:
                    self.stop_reason='tolX'
                    break

        if best is not None:
            _,params,opt_state,self.best_step=best
        return params,opt_state,rng

    def minimize(self,f0,rng,stim,filter,loss_fun,opt_state=None,extra_params=None,val_fun=None,mstate0=None,state_fun=None):
        tx=self.tx
        proj_fun=self._projection
        proj_params=self._projection_params
        bTangent=self.projectionType[0]=='l2_sphere'
        index=tuple(jnp.asarray(i) for i in filter._insert_index_jx)
        index_exp=tuple(jnp.asarray(i) for i in filter._insert_index_exp_jx)

        # f0 (and any other learned parameters, e.g. pooling weights)
        params=self.insert_project_extract({'f':f0,**(extra_params or {})},filter.prepped_exp_jx,index_exp,proj_fun,proj_params)
        if opt_state is None:
            opt_state=tx.init(params)

        if self.batchSize is None:
            mMax,batch_mask=None,None
        else:
            mMax,batch_mask=self._batch_plan(stim.weights)

        nSteps=max(1,min(self.nStepsPerChunk,self.nIterMax))
        # mstate0: an initial model state to carry through training (returned states are the optax part only); state_fun:
        # a model state recomputed at the start of every chunk instead
        if state_fun is not None:
            mstate0=state_fun(params,filter.prepped_jx,index,stim.val,stim.weights)
        bState=mstate0 is not None
        run=lambda params,opt_state,rng,nActive: self._run_chunk(tx,loss_fun,proj_fun,proj_params,nSteps,mMax,bTangent,self._bAccept,
                                                                bState,nActive,params,opt_state,rng,
                                                                filter.prepped_jx,filter.prepped_exp_jx,index,index_exp,batch_mask,
                                                                stim.val,stim.weights,stim.yCtg,stim.Y,state_fun,self._bLBFGS)
        params,opt_state,rng=self._loop(run,nSteps,params,(opt_state,mstate0) if bState else opt_state,rng,val_fun)
        return params,(opt_state[0] if bState else opt_state),rng


__all__=['_ama_sgd', 'Optimizer']
