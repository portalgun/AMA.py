"""Posteriors, estimates, errors and losses (Objective)."""
from ._base import *


class Objective(_Static):
    """
    loss(error(estimate(posterior(log-likelihood))))

    errType
        'map' -log p(X_k|R) at the correct level  (0,1 cost / KL, Burge & Jaini 2017 Eq 9)
        'mle' -log p(R|X_k) at the correct level
        'l2'  (Xhat - X_k)^2, estType defaults to posterior 'mean' (MMSE, Eq 13-15)
        'l1'  |Xhat - X_k|,   estType defaults to posterior 'median'
    Divergences between the posterior over levels p and a target q around X_k (targetSigma):
        'xent'        cross-entropy -sum_i q_i log p_i; equals 'map' with the one-hot target
        'js'          Jensen-Shannon divergence, bounded by log 2
        'wasserstein' 1-Wasserstein (earth mover's) distance over Y, sum |F_p - F_q| dY;
                      with the one-hot target the posterior mean of |Y - X_k|
        'fisher'      Fisher divergence sum q (s_p - s_q)^2 of the scores s = d log / dY, as finite differences
                      between neighbouring levels (independent of the posterior's normalization); needs targetSigma
        With several latent dimensions (euclidean distance, wrapped on circular dimensions): 'wasserstein' is exact for the
        one-hot target (the posterior mean distance to X_k) and the debiased entropic (Sinkhorn) divergence for a gaussian
        target (otEps, nOtIter); 'fisher' needs the levels on a cartesian grid and sums the per-axis divergences.

    Several latent dimensions: 'l1' and 'l2' sum over them (L1 and squared euclidean distance), 'mean' and 'median'
    estimate each dimension (the marginal median minimizes the expected L1 distance), and the gaussian target is
    isotropic. Circular dimensions (Stim Yperiod) wrap all differences, and 'mean' is their circular mean.
    """
    _posterior_fun=_id
    _est_fun=_id
    _err_fun=_id
    _loss_fun=_id
    bPosterior=_TypeFunc(True)
    estType=_TypeFunc()
    errType=_TypeFunc()
    lossType=_TypeFunc()
    _DIVERGENCES=('xent','js','wasserstein','fisher')
    _Yperiod=None                                                               # from the Stim, set by Unit._set_geometry
    _Ygrid=None                                                                 # _latent_grid of the Stim's Y
    _bContinuous=False                                                          # Stim y: stimuli have their own latent values
    def __init__(self,errType='map',bPosterior=None,estType=None,lossType='mean',regType='None',regWeight=0.,targetSigma=None,
                 otEps=0.05,nOtIter=200,_bCopy=False):
        """
        targetSigma - target distribution over the levels for the divergence errTypes: None for one-hot at X_k, or the
                      standard deviation (in units of Y) of a gaussian target q_i ~ exp(-(Y_i - X_k)^2 / 2 targetSigma^2)
        otEps, nOtIter - 'wasserstein' with several latent dimensions and a gaussian target: the debiased entropic
                      (Sinkhorn) divergence with regularization otEps times the mean distance between levels, from nOtIter
                      Sinkhorn iterations (smaller otEps approaches the exact distance but needs more iterations)
        regType   - penalty on the filters while training (not included in Unit.loss; see Unit.penalty), in the
                    learning domain (spatial, or fourier for fourierType >= 1), over the learned coefficients:
                    'None'
                    'l1'     - sum |f| per filter: sparse filters (sparse spectra when learning in the fourier domain)
                    'smooth' - sum of squared differences between neighbouring coefficients per filter: smooth
                               spatial filters, or smooth spectra (compact spatial support) in the fourier domain
        regWeight - weight of the penalty, added to the cost as regWeight * mean penalty over filters
        """
        self.regType=regType
        self.regWeight=float(regWeight)
        if str(regType).lower() not in ('none','l1','smooth'):
            raise Exception("regType must be 'None', 'l1', or 'smooth'")

        #- errType
        if isinstance(errType,(int, float)) and not isinstance(errType,bool):
            if errType not in (1,2):
                raise Exception('numeric errType must be 1 or 2')
            errType='l' + str(int(errType))
        self.errType=errType
        self.lossType=lossType
        self.targetSigma=None if targetSigma is None else float(targetSigma)
        if self.targetSigma is not None and not self.targetSigma>0:
            raise Exception('targetSigma must be None or positive')
        self.otEps=float(otEps)
        self.nOtIter=int(nOtIter)
        if not self.otEps>0 or self.nOtIter<1:
            raise Exception('otEps must be positive and nOtIter at least 1')

        if _bCopy:
            self.bPosterior=bPosterior
            self.estType=estType
            return

        #- posterior
        if self.errType=='fisher' and self.targetSigma is None:
            raise Exception("errType='fisher' needs a gaussian target (targetSigma): the one-hot target has no score")
        if self.targetSigma is not None and self.errType not in self._DIVERGENCES:
            raise Exception('targetSigma is only used by errType ' + ', '.join(self._DIVERGENCES))
        if   self.errType == 'mle':
            if bPosterior:
                raise Exception('bPosterior must not be set for errType=mle')
            bPosterior=False
        elif self.errType == 'map' or self.errType in self._DIVERGENCES:
            if bPosterior is False:
                raise Exception('bPosterior must not be False for errType=' + self.errType)
            bPosterior=True
        elif bPosterior is None:
            bPosterior=True
        self.bPosterior=bPosterior

        #- estType
        if self.errType in ('mle','map') + self._DIVERGENCES:
            if estType is not None:
                raise Exception('estType must not be set for errType=' + self.errType)
        elif estType is None:
            estType='median' if self.errType=='l1' else 'mean'
        self.estType=estType


    def _key(self):
        return (self.errType,self.bPosterior,self.estType,self.lossType,str(self.regType).lower(),self.regWeight,self.targetSigma,
                self.otEps,self.nOtIter,self._Yperiod,self._Ygrid,self._bContinuous)

    def _err_opts(self):
        return _ErrOpts(self._Ygrid,self.targetSigma is None,self.otEps,self.nOtIter)

    def copy(self):
       return Objective(**_get_copy_dict(self),_bCopy=True)

    @partial(jit, static_argnames=['self'])
    def lrn_main(self,lAll,stimweights,yCtg,Y,priorweights=None,Yc=None):
        # the prior comes from priorweights (the training stimuli) when decoding other stimuli. Yc: the latent values that
        # stand for each candidate category, per stimulus (Model bWithin), instead of the levels Y
        prior=stimweights if priorweights is None else priorweights
        per=self._Yperiod
        return self._loss_fun(self._err_fun(self._est_fun(self._posterior_fun(lAll,prior),Y,per,Yc),yCtg,self.log_target(Y,yCtg),
                                            Y,per,self._err_opts()),stimweights)

    def log_target(self,Y,yCtg=None):
        """
        log target distribution over the levels for the divergence errTypes: [ nCtg (correct) x nCtg ], or with stimuli
        that have their own latent values (Stim y) and a gaussian target [ nStim_Ctg x nCtg x nCtg ], centered at each
        stimulus's value yCtg
        """
        if self.targetSigma is None:
            return jnp.where(jnp.eye(len(Y),dtype=bool),0.,-jnp.inf).astype(Y.dtype)
        if self._bContinuous and yCtg is not None:
            d=_wrap(Y-(yCtg[...,None] if Y.ndim==1 else yCtg[...,None,:]),self._Yperiod)
            lq=-_ysq(d,Y)/(2*self.targetSigma**2)
        else:
            lq=-_ydist2(Y,self._Yperiod)/(2*self.targetSigma**2)
        return lq - logsumexp(lq,axis=-1,keepdims=True)

    #- posterior (log domain)
    @staticmethod
    def _posterior__true(lAll,stimweights):
        # prior p(X_i) = N_i / N from the training set
        wc=jnp.sum(stimweights,axis=0)
        lp=lAll + jnp.log(wc/jnp.sum(wc))
        return lp - logsumexp(lp,axis=-1,keepdims=True)

    @staticmethod
    def _posterior__none(lAll,stimweights):
        return lAll

    #- estimation
    @staticmethod
    def _prob(lpost):
        return jnp.exp(lpost - logsumexp(lpost,axis=-1,keepdims=True))

    # estimates [ ... ] for 1-D Y, [ ... x nDim ] for several latent dimensions. Yc [ ... x nCtg (x nDim) ] (Model bWithin):
    # the latent value each candidate category stands for, per stimulus, in place of the levels Y
    @staticmethod
    def _est__none(lpost,Y,per=None,Yc=None):
        return lpost

    @staticmethod
    def _wmedian(p,y):
        # median of point masses p at y (both [ ... x n ]), per row
        order=jnp.argsort(y,axis=-1)
        cdf=jnp.cumsum(jnp.take_along_axis(p,order,axis=-1),axis=-1)
        return jnp.vectorize(lambda c,v: jnp.interp(0.5,c,v),signature='(n),(n)->()')(cdf,jnp.take_along_axis(y,order,axis=-1))

    @staticmethod
    def _est__median(lpost,Y,per=None,Yc=None):
        # (marginal) median: cdf over latent values in ascending order (Y need not be sorted)
        p=Objective._prob(lpost)
        if Yc is not None:
            return Objective._median_within(p,Y,per,Yc)
        def median(y):
            order=jnp.argsort(y)
            cdf=jnp.cumsum(p[...,order],axis=-1)
            return jnp.vectorize(lambda c: jnp.interp(0.5,c,y[order]),signature='(n)->()')(cdf)
        def circ_median(y,P):
            # the circular median m minimizes the expected wrapped distance, and the diameter through m halves the
            # probability: find it among the levels and their antipodes (no gradient), cut the circle at its antipode,
            # and take the median of the unwrapped distribution
            cand=jnp.concatenate((y,y+P/2))
            dist=jnp.abs(_wrap(y[:,None]-cand[None,:],(P,)))                     # [ nCtg x 2 nCtg ]
            m0=lax.stop_gradient(cand[jnp.argmin(p@dist,axis=-1)])                 # [ ... ]
            yu=m0[...,None]+_wrap(y-m0[...,None],(P,))                             # [ ... x nCtg ]
            order=jnp.argsort(yu,axis=-1)
            cdf=jnp.cumsum(jnp.take_along_axis(p,order,axis=-1),axis=-1)
            ys=jnp.take_along_axis(yu,order,axis=-1)
            return _wrap(jnp.vectorize(lambda c,v: jnp.interp(0.5,c,v),signature='(n),(n)->()')(cdf,ys),(P,))
        if Y.ndim==1:
            return median(Y) if per is None else circ_median(Y,per[0])
        pd=per if per is not None else (None,)*Y.shape[1]
        return jnp.stack([median(Y[:,d]) if pd[d] is None else circ_median(Y[:,d],pd[d]) for d in range(Y.shape[1])],axis=-1)

    @staticmethod
    def _median_within(p,Y,per,Yc):
        # median of point masses at the within-category values Yc; on circular dimensions cut the circle opposite the
        # circular mean and take the median of the unwrapped values
        def one(y,P):
            if P is None:
                return Objective._wmedian(p,y)
            m0=lax.stop_gradient(jnp.angle(jnp.sum(p*jnp.exp(2j*jnp.pi*y/P),axis=-1))*P/(2*jnp.pi))
            return _wrap(Objective._wmedian(p,m0[...,None]+_wrap(y-m0[...,None],(P,))),(P,))
        if Y.ndim==1:
            return one(Yc,None if per is None else per[0])
        pd=per if per is not None else (None,)*Y.shape[1]
        return jnp.stack([one(Yc[...,d],pd[d]) for d in range(Y.shape[1])],axis=-1)

    @staticmethod
    def _est__mean(lpost,Y,per=None,Yc=None):
        # posterior mean; on circular dimensions the circular mean
        p=Objective._prob(lpost)
        if Yc is None:
            avg=lambda V: p @ V
            V=Y
        else:
            avg=lambda V: jnp.sum(p*V,axis=-1) if Y.ndim==1 else jnp.sum(p[...,None]*V,axis=-2)
            V=Yc
        m=avg(V)
        if per is None:
            return m
        P,bC=_period_arrays(per,Y.dtype)
        if Y.ndim==1:
            P,bC=P[0],bC[0]
        # zero phase on linear dimensions keeps angle() away from 0, where its gradient is not finite
        mc=jnp.angle(avg(jnp.exp(1j*jnp.where(bC,2*jnp.pi*V/P,0.))))*P/(2*jnp.pi)
        return jnp.where(bC,mc,m)

    @staticmethod
    def _est__mode(lpost,Y,per=None,Yc=None):
        # MAP estimate; piecewise constant, so provides no gradient
        i=jnp.argmax(lpost,axis=-1)
        if Yc is None:
            return Y[i]
        if Y.ndim==1:
            return jnp.take_along_axis(Yc,i[...,None],axis=-1)[...,0]
        return jnp.take_along_axis(Yc,i[...,None,None],axis=-2)[...,0,:]

    @staticmethod
    def _est__cmean(lpost,Y,per=None,Yc=None):
        # circular mean, Y in radians (see also Stim Yperiod, with which 'mean' is circular)
        p=Objective._prob(lpost)
        return jnp.angle(jnp.sum(p*jnp.exp(1j*Yc),axis=-1) if Yc is not None else p @ jnp.exp(1j*Y))

    #- error
    @staticmethod
    def _err__mle(lAll,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        return -Objective._at_correct(lAll)

    @staticmethod
    def _err__map(lpost,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        return -Objective._at_correct(lpost)

    @staticmethod
    def _err__l1(yHat,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        d=jnp.abs(_wrap(yHat-yCtg,per))
        return d if Y is None or Y.ndim==1 else jnp.sum(d,axis=-1)

    @staticmethod
    def _err__l2(yHat,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        d=jnp.abs(_wrap(yHat-yCtg,per))
        return d**2 if Y is None or Y.ndim==1 else jnp.sum(d**2,axis=-1)

    # divergences: lpost [ nStim_Ctg x nCtg x nCtg ] against the target row of each correct level, lQ [ nCtg x nCtg ]
    @staticmethod
    def _err__xent(lpost,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        q=jnp.exp(lQ)
        return -jnp.sum(jnp.where(q>0,q*lpost,0),axis=-1)

    @staticmethod
    def _err__js(lpost,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        # 0 log 0 = 0; lQ is -inf off the one-hot target, so both logs go through where
        p,q=jnp.exp(lpost),jnp.exp(lQ)
        lm=jnp.logaddexp(lpost,lQ)-jnp.log(2.)
        lms=jnp.where((p>0)|(q>0),lm,0)
        tp=jnp.where(p>0,p*(jnp.where(p>0,lpost,0)-lms),0)
        tq=jnp.where(q>0,q*(jnp.where(q>0,lQ,0)-lms),0)
        return 0.5*jnp.sum(tp+tq,axis=-1)

    @staticmethod
    def _err__wasserstein(lpost,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        if Y.ndim>1:
            return Objective._wasserstein_nd(lpost,lQ,Y,per,opts or _ErrOpts())
        # both distributions sit on the levels, so their cdfs are constant between neighbouring levels
        y,order,dY=Objective._levels(Y,per)
        dF=jnp.cumsum(jnp.exp(lpost)[...,order] - jnp.exp(lQ)[...,order],axis=-1)
        if per is None:
            return jnp.sum(jnp.abs(dF[...,:-1])*dY,axis=-1)
        # on a circle the cdf difference is defined up to a constant, W1 = min_c sum |dF - c| dY: c is the dY-weighted median
        o=jnp.argsort(dF,axis=-1)
        cw=jnp.cumsum(dY[o],axis=-1)
        c=jnp.take_along_axis(jnp.take_along_axis(dF,o,axis=-1),jnp.argmax(cw>=cw[...,-1:]/2,axis=-1)[...,None],axis=-1)
        return jnp.sum(jnp.abs(dF-c)*dY,axis=-1)

    @staticmethod
    def _wasserstein_nd(lpost,lQ,Y,per,opts):
        # euclidean ground distance between levels (wrapped on circular dimensions)
        D=jnp.sqrt(_ydist2(Y,per))
        if opts.bOneHot:
            # all the target's mass sits at X_k: the posterior mean distance to it, exactly
            return jnp.sum(jnp.exp(lpost)*D[None],axis=-1)
        n=D.shape[0]
        eps=opts.otEps*jnp.sum(D)/(n*(n-1))
        ot=lambda la,lb: _sinkhorn_value(la,lb,D,eps,opts.nOtIter)
        # Sinkhorn divergence: 0 at the target (one target per correct level, or per stimulus with Stim y)
        if lQ.ndim<lpost.ndim:
            lQ=lQ[None]
        return ot(lpost,lQ) - 0.5*ot(lpost,lpost) - 0.5*ot(lQ,lQ)

    @staticmethod
    def _err__fisher(lpost,yCtg=None,lQ=None,Y=None,per=None,opts=None):
        # scores at the midpoints between neighbouring levels (on a circle also the last and first), weighted by the
        # target there
        if Y.ndim>1:
            return Objective._fisher_grid(lpost,lQ,Y,per,opts)
        y,order,dY=Objective._levels(Y,per)
        lp,lq=lpost[...,order],lQ[...,order]
        nxt=lambda a: jnp.roll(a,-1,axis=-1) if per is not None else a[...,1:]
        cur=lambda a: a if per is not None else a[...,:-1]
        sp=(nxt(lp)-cur(lp))/dY
        sq=(nxt(lq)-cur(lq))/dY
        q=jnp.exp(lq)
        w=(nxt(q)+cur(q))/2
        return jnp.sum(w*(sp-sq)**2,axis=-1)/jnp.sum(w,axis=-1)

    @staticmethod
    def _fisher_grid(lpost,lQ,Y,per,opts):
        # levels on a cartesian grid: finite-difference scores along each axis (wrapping on circular axes); the sum over
        # axes of the target-weighted mean squared score difference along that axis (the 1-D formula per axis)
        if opts is None or opts.grid is None:
            raise Exception("errType='fisher' with several latent dimensions needs the levels on a cartesian grid")
        shape,order=opts.grid
        order=jnp.asarray(order)
        lp=lpost[...,order].reshape(lpost.shape[:-1]+shape)
        lq=lQ[...,order].reshape(lQ.shape[:-1]+shape)
        Yg=Y[order].reshape(shape+(Y.shape[1],))
        nd=len(shape)
        tot=0.
        for d in range(nd):
            if shape[d]<2:                                                   # no neighbours along this axis
                continue
            ax=-nd+d
            bCirc=per is not None and per[d] is not None
            if bCirc:
                nxt=lambda a: jnp.roll(a,-1,axis=ax)
                cur=lambda a: a
            else:
                nxt=lambda a: lax.slice_in_dim(a,1,None,axis=a.ndim+ax)
                cur=lambda a: lax.slice_in_dim(a,0,a.shape[a.ndim+ax]-1,axis=a.ndim+ax)
            # spacing along axis d (the same at every position of the other axes); circular axes span less than a period
            # (Unit._check), so ascending values are in circular order
            yd=Yg[(0,)*d+(slice(None),)+(0,)*(nd-d-1)+(d,)]
            dY=jnp.diff(jnp.concatenate((yd,yd[:1]+per[d]))) if bCirc else jnp.diff(yd)
            dY=dY.reshape((-1,)+(1,)*(nd-d-1))
            sp=(nxt(lp)-cur(lp))/dY
            sq=(nxt(lq)-cur(lq))/dY
            q=jnp.exp(lq)
            w=(nxt(q)+cur(q))/2
            ax_sum=tuple(range(-nd,0))
            tot=tot+jnp.sum(w*(sp-sq)**2,axis=ax_sum)/jnp.sum(w,axis=ax_sum)
        return tot

    @staticmethod
    def _levels(Y,per):
        """
        a single latent dimension in ascending order: positions, order, and the spacings to the next level [ nCtg-1 ],
        or on a circle positions in [0, P) and spacings [ nCtg ] including the one from the last level around to the first
        """
        if Y.ndim>1:
            raise Exception("errType 'wasserstein' and 'fisher' need a single latent dimension")
        if per is None:
            order=jnp.argsort(Y)
            return Y[order],order,jnp.diff(Y[order])
        P=per[0]
        y=jnp.mod(Y,P)
        order=jnp.argsort(y)
        y=y[order]
        return y,order,jnp.diff(jnp.concatenate((y,y[:1]+P)))

    #- loss (padding in the category-grouped stimuli has weight 0)
    @staticmethod
    def _loss__mean(err,stimweights):
        return jnp.sum(jnp.where(stimweights>0,err,0)*stimweights)/jnp.sum(stimweights)

    @staticmethod
    def _loss__median(err,stimweights):
        return jnp.nanmedian(jnp.where(stimweights>0,err,jnp.nan))

    #- helpers
    @staticmethod
    def _at_correct(inAll):
        # [ nStim_Ctg x nCtg x nCtg ] -> [ nStim_Ctg x nCtg ]
        return jnp.diagonal(inAll, axis1=-2, axis2=-1)


__all__=['Objective']
