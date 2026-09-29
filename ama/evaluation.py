"""Held-out evaluation for Unit: decoding other stimuli against the training set, estimates, performance, cross-validation."""
from ._base import *
from .objective import Objective


class _Evaluation:
    """Unit methods: decoding other stimuli with the training set"""

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

        # the references enter through their mean responses and variances only, which do not depend on the noise draw: once
        ref=self.nrn.main(rng_key,refval,f,refweights,W,p,G)

        def one(key):
            obs=self.nrn.main(key,stimval,f,stimweights,W,p,G)
            lAll,Yc=self._lik_parts(self._likelihoods_heldout(obs,ref,refweights,Y,refy))
            return self.objective.lrn_main(lAll,stimweights,yCtg,Y,refweights,Yc),None

        return self._noise_average(one,rng_key)[0]

    def _decoder(self,model):
        # this unit, or a copy of it (same filters, stimuli and learned parameters) that decodes with another Model
        if model is None:
            return self
        if not self.nrn.bFinalized:
            raise Exception('train (or finalize) the unit first')
        unit=self.split()
        unit.model=model.copy()
        unit._set_geometry()
        unit._check()
        return unit

    def evaluate(self,stim,model=None):
        """
        cost of decoding other stimuli (e.g. a held-out test set) with the current filters. This unit's stimuli are the
        training set: the category response distributions (AMA-Gauss), the reference stimuli (full AMA), the prior,
        and the whitening all come from them. model: decode with this Model instead of the unit's (e.g. filters
        trained under 'gss', decoded with 'student'); the unit's own Model is not changed
        """
        if model is not None:
            return self._decoder(model).evaluate(stim)
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

    def estimates(self,estType='mode',stim=None,model=None):
        """
        estimates of the latent variable [ nStim_Ctg x nCtg (x nDim) ] (grouped like Stim.val; see Stim.weights) for the
        training stimuli, or for other stimuli decoded with the training set: 'mode' (MAP), 'mean' (MMSE; circular on
        circular dimensions), 'median', or 'cmean' (circular mean, Y in radians). With Model bWithin, continuous
        estimates within the categories. model: decode with this Model instead of the unit's (see evaluate). With
        noisy observations (responseType 'basic'), one noise draw (the unit's key), not the nNoiseSamples average of loss
        and evaluate
        """
        lpost,st,Yc=self._decoder(model)._log_posterior(stim)
        return np.asarray(getattr(Objective,'_est__'+estType)(lpost,st.Y,st.Yperiod,Yc))

    def performance(self,estType='mode',stim=None,model=None):
        """
        estimation performance per latent level for the training stimuli, or for other stimuli decoded with the training
        set: bias, sd, and rmse of the estimates ([ nCtg (x nDim) ]; errors wrap on circular dimensions) and over all
        stimuli (rmseAll); pCorrect and confusion [ true x MAP category ] of the MAP category; and cost, the mean -log
        posterior at the correct level. Errors are measured from each stimulus's own latent value (Stim y), which for
        stimuli without their own values is their category's level. model: decode with this Model instead of the unit's
        (see evaluate). With noisy observations (responseType 'basic'), one noise draw (the unit's key): its cost is not
        the nNoiseSamples average of loss and evaluate
        """
        lpost,st,Yc=self._decoder(model)._log_posterior(stim)
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
            unit=type(self)(train,self.nrn.copy(),self.model.copy(),self.objective.copy(),
                      optimizer=None if self.optimizer is None else self.optimizer.copy(),seed=self.seed+i)
            unit.train_new(n,**train_kw)
            res['train'].append(float(unit.loss))
            res['test'].append(unit.evaluate(test))
            res['filters'].append(np.asarray(unit.out))
        return {key:np.array(val) for key,val in res.items()}


__all__=['_Evaluation']
