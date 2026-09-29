"""
Full AMA from the nearest references (Model nNeighbors, nTail): the search, the likelihoods against a brute-force
computation over the neighbours and the drawn tail, the limit of all references, unbiasedness of the tail, and learning.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.special import logsumexp

import ama
import stimuli as ts


def full_unit(nNeighbors=None,nTail=8,bLeaveOneOut=False,gen=ts.unequal_counts,n=2,seed=1,nrn=None,**model_kw):
    x,s,ci,Y,_=gen()
    unit=ama.Unit(ama.Stim(x,s,ci,Y),nrn or ama.Nrn(),
                  ama.Model('full','mean',bLeaveOneOut=bLeaveOneOut,nNeighbors=nNeighbors,nTail=nTail,**model_kw),
                  ama.Objective('map'),ama.Optimizer(nIterMax=1,bVerbose=False))
    unit._finalize(n,np.arange(n),dtype=jnp.float64)
    f=np.random.default_rng(seed).standard_normal(unit.filter._shape)
    unit.filter.out=jnp.asarray(f/np.linalg.norm(f,axis=0))
    return unit


def lrn_parts(unit,key):
    """(learnable params, neighbour table, cost, likelihoods, responses) of the training cost with this key"""
    flt=unit.filter
    prm={'f':flt.out_flat[flt._insert_index_jx]}
    nbr=unit._nbr_state(prm,flt.prepped_jx,flt._insert_index_jx,unit.stim.val,unit.stim.weights)
    cost,_=unit._loss_fun_lrn(prm,key,flt.prepped_jx,flt._insert_index_jx,unit.stim.val,unit.stim.weights,unit.stim.yCtg,
                              unit.stim.Y,nbr)
    out=unit.nrn.lrn_main(key,unit.stim.val,prm['f'],flt.prepped_jx,flt._insert_index_jx,unit.stim.weights)
    lAll=unit._likelihoods(out,unit.stim.weights,unit.stim.Y,jax.random.fold_in(key,7),nbr=nbr)
    return np.asarray(nbr),float(cost),np.asarray(lAll),[np.asarray(ama._flatten_responses(v)) for v in unit.model._response_fun(*out)]


def exact_cost(unit):
    return float(unit.loss)


def term(R,Rm,RVar,l,k,j,i):
    v=RVar[:,j,i]
    return -0.5*np.sum((R[:,l,k]-Rm[:,j,i])**2/v)-0.5*np.sum(np.log(2*np.pi*v))


class TestSearch:
    @pytest.mark.parametrize('bLeaveOneOut',[False,True])
    def test_nearest_by_likelihood_term(self,bLeaveOneOut):
        unit=full_unit(nNeighbors=5,bLeaveOneOut=bLeaveOneOut)
        nbr,_,_,(R,Rm,RVar)=lrn_parts(unit,jax.random.key(0))
        w=np.asarray(unit.stim.weights)
        nC=w.shape[1]
        assert nbr.shape==w.shape+(nC,5)
        for (l,k) in [(0,0),(4,1),(9,2),(25,1)]:
            if w[l,k]==0:
                continue
            for i in range(nC):
                t=np.array([term(Rm,Rm,RVar,l,k,j,i) if w[j,i]>0 and not (bLeaveOneOut and (j,i)==(l,k)) else -np.inf
                            for j in range(w.shape[0])])
                assert set(nbr[l,k,i])==set(np.argsort(-t)[:5])             # the nearest of each category

    def test_chunked_search_is_identical(self,monkeypatch):
        unit=full_unit(nNeighbors=6,bLeaveOneOut=True)
        a=lrn_parts(unit,jax.random.key(0))[0]
        monkeypatch.setattr(ama.model,'_FULL_CHUNK',7*unit.stim.weights.size)
        u=full_unit(nNeighbors=6,bLeaveOneOut=True)
        b=lrn_parts(u,jax.random.key(0))[0]
        assert np.array_equal(np.sort(a,-1),np.sort(b,-1))
        # and the likelihoods from them
        out=u._nrn_out()
        lik=lambda: np.asarray(u._likelihoods(out,u.stim.weights,u.stim.Y,jax.random.key(2),nbr=jnp.asarray(a)))
        chunked=lik()
        monkeypatch.setattr(ama.model,'_FULL_CHUNK',2**24)
        assert np.allclose(chunked,lik(),rtol=1e-12)


class TestLikelihoods:
    @pytest.mark.parametrize('bLeaveOneOut',[False,True])
    @pytest.mark.parametrize('nTail',[1,5])
    def test_all_references_is_full_ama(self,bLeaveOneOut,nTail):
        exact=full_unit(bLeaveOneOut=bLeaveOneOut)
        nb=full_unit(nNeighbors=1000,nTail=nTail,bLeaveOneOut=bLeaveOneOut)
        key=jax.random.key(3)
        _,cost,_,_=lrn_parts(nb,key)
        e=float(exact._loss_fun_lrn({'f':exact.filter.out_flat[exact.filter._insert_index_jx]},key,exact.filter.prepped_jx,
                                    exact.filter._insert_index_jx,exact.stim.val,exact.stim.weights,exact.stim.yCtg,exact.stim.Y))
        assert np.isclose(cost,e,rtol=1e-10)

    def test_correlated_noise_all_references(self):
        nrn=lambda: ama.Nrn(rho=0.3)
        exact=full_unit(nrn=nrn())
        nb=full_unit(nNeighbors=1000,nrn=nrn())
        assert np.isclose(lrn_parts(nb,jax.random.key(1))[1],
                          float(exact._loss_fun_lrn({'f':exact.filter.out_flat[exact.filter._insert_index_jx]},jax.random.key(1),
                                                    exact.filter.prepped_jx,exact.filter._insert_index_jx,exact.stim.val,
                                                    exact.stim.weights,exact.stim.yCtg,exact.stim.Y)),rtol=1e-10)

    @pytest.mark.parametrize('bLeaveOneOut',[False,True])
    def test_matches_brute_force(self,bLeaveOneOut):
        unit=full_unit(nNeighbors=6,nTail=4,bLeaveOneOut=bLeaveOneOut)
        key=jax.random.key(5)
        nbr,_,lAll,(R,Rm,RVar)=lrn_parts(unit,key)
        tail=np.asarray(ama.Model._ref_subset(jax.random.fold_in(key,7),unit.stim.weights,4))
        w=np.asarray(unit.stim.weights)
        nC=w.shape[1]
        for (l,k,i) in [(0,0,0),(3,1,1),(7,2,0),(2,0,2),(15,1,2)]:
            mine=[j for j in nbr[l,k,i] if w[j,i]>0 and not (bLeaveOneOut and (j,i)==(l,k))]
            tj=[j for j in tail[:,i] if w[j,i]>0 and j not in mine and not (bLeaveOneOut and (j,i)==(l,k))]
            rest=w[:,i].sum()-len(mine)-(1 if (bLeaveOneOut and i==k) else 0)
            L=sum(np.exp(term(R,Rm,RVar,l,k,j,i)) for j in mine)
            if tj:
                L+=rest*np.mean([np.exp(term(R,Rm,RVar,l,k,j,i)) for j in tj])
            assert np.isclose(lAll[l,k,i],np.log(L)-np.log(w[:,i].sum()),rtol=1e-9)

    def test_tail_is_unbiased(self):
        # the likelihood (not its log) averaged over tail draws approaches the exact one
        unit=full_unit(nNeighbors=3,nTail=4,gen=ts.gaussian_ctg)
        flt=unit.filter
        prm={'f':flt.out_flat[flt._insert_index_jx]}
        nbr=unit._nbr_state(prm,flt.prepped_jx,flt._insert_index_jx,unit.stim.val,unit.stim.weights)
        out=unit.nrn.main(unit.rng,unit.stim.val,flt.out_flat,unit.stim.weights)
        exact=np.exp(np.asarray(full_unit(gen=ts.gaussian_ctg)._likelihoods(out,unit.stim.weights,unit.stim.Y)))
        draws=np.mean([np.exp(np.asarray(unit._likelihoods(out,unit.stim.weights,unit.stim.Y,jax.random.key(i),nbr=nbr)))
                       for i in range(400)],0)
        R,Rm,RVar=[ama._flatten_responses(v) for v in unit.model._response_fun(*out)]
        trunc=np.exp(np.asarray(ama.Model._full_neighbors(R,Rm,RVar,None,unit.stim.weights,nbr,None,False)))   # no tail
        assert np.median(np.abs(draws/exact-1))<0.02                        # unbiased (up to the sampling error)
        assert np.all(trunc<=exact*(1+1e-9))                                # without a tail: a lower bound
        assert np.median(np.abs(trunc/exact-1))>0.1

    def test_within_values(self):
        x,s,y,_=ts.continuous_gaussian(nStim=90)
        st=ama.Stim.binned(x,s,y,nBins=3)
        unit=ama.Unit(st,ama.Nrn(),ama.Model('full','mean',nNeighbors=1000,bWithin=True),ama.Objective('l2'),
                      ama.Optimizer(nIterMax=1,bVerbose=False))
        unit._finalize(2,np.arange(2),dtype=jnp.float64)
        f=np.random.default_rng(0).standard_normal(unit.filter._shape)
        unit.filter.out=jnp.asarray(f/np.linalg.norm(f,axis=0))
        flt=unit.filter
        prm={'f':flt.out_flat[flt._insert_index_jx]}
        nbr=unit._nbr_state(prm,flt.prepped_jx,flt._insert_index_jx,unit.stim.val,unit.stim.weights)
        out=unit._nrn_out()
        a=unit._likelihoods(out,unit.stim.weights,unit.stim.Y,jax.random.key(0),yRef=unit.stim.yCtg,nbr=nbr)
        b=unit._likelihoods(out,unit.stim.weights,unit.stim.Y,yRef=unit.stim.yCtg)
        assert np.allclose(np.asarray(a[0]),np.asarray(b[0]),rtol=1e-10) and np.allclose(np.asarray(a[1]),np.asarray(b[1]),rtol=1e-10)


class TestTraining:
    def test_learning(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        costs={}
        for K in (None,12):
            unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('full','mean',nNeighbors=K),ama.Objective('map'),
                          ama.Optimizer(nIterMax=150,lRate0=0.05,bVerbose=False,nStepsPerChunk=25))
            unit.train_new(1)
            h=np.asarray(unit.optimizer.loss_hist)
            assert np.all(np.isfinite(h)) and h[-1]<h[0]
            costs[K]=float(unit.loss)
        assert costs[12]<costs[None]+0.05                                    # nearly as good, by the exact cost

    def test_neighbours_follow_the_filters(self):
        # a chunk starts with a new search: training from filters whose neighbours are wrong still learns
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('full','mean',nNeighbors=8,nTail=1),ama.Objective('map'),
                      ama.Optimizer(nIterMax=60,lRate0=0.05,bVerbose=False,nStepsPerChunk=10))
        unit.train_new(1)
        unit.train_recurse()
        assert np.all(np.isfinite(unit.optimizer.loss_hist))

    def test_validation(self):
        with pytest.raises(Exception,match='nTail'):
            ama.Model('full','mean',nNeighbors=5,nTail=0)
        x,s,ci,Y,_=ts.gaussian_ctg()
        for model,opt,match in [(ama.Model('gss','mean',nNeighbors=5),{},'nNeighbors'),
                                (ama.Model('full','mean',nNeighbors=5,nRef=5),{},'alternatives')]:
            unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),model,ama.Objective('map'),ama.Optimizer(nIterMax=1,bVerbose=False,**opt))
            with pytest.raises(Exception,match=match):
                unit.train_new(1)


class TestBatches:
    """nNeighbors with batchSize: a batch of observed stimuli decoded against all training stimuli"""

    @staticmethod
    def batch_parts(unit,key,bWithin=False):
        flt=unit.filter
        prm={'f':flt.out_flat[flt._insert_index_jx]}
        nbr=unit._nbr_state(prm,flt.prepped_jx,flt._insert_index_jx,unit.stim.val,unit.stim.weights)
        mMax,mask=ama.Optimizer(batchSize=30)._batch_plan(unit.stim.weights)
        idx,wb=ama.Optimizer._batch_index(jax.random.key(3),mMax,mask,unit.stim.weights)
        cost,st=unit._loss_fun_lrn(prm,key,flt.prepped_jx,flt._insert_index_jx,unit.stim.val,unit.stim.weights,unit.stim.yCtg,
                                   unit.stim.Y,(nbr,idx,wb))
        out=unit.nrn.lrn_main(key,unit.stim.val,prm['f'],flt.prepped_jx,flt._insert_index_jx,unit.stim.weights)
        lik=unit._likelihoods(out,unit.stim.weights,unit.stim.Y,jax.random.fold_in(key,7),nbr=nbr,obs=(idx,wb),
                              yRef=unit.stim.yCtg if bWithin else None)
        return prm,nbr,idx,wb,float(cost),lik

    @pytest.mark.parametrize('bLeaveOneOut',[False,True])
    def test_all_references_is_exact_on_the_batch(self,bLeaveOneOut):
        unit=full_unit(nNeighbors=100,bLeaveOneOut=bLeaveOneOut)          # K at least the largest category: exact
        prm,nbr,idx,wb,cost,lAll=self.batch_parts(unit,jax.random.key(0))
        exact=np.asarray(unit.likelihoods)
        idx,wb=np.asarray(idx),np.asarray(wb)
        ref=exact[idx,np.arange(idx.shape[1])[None,:]]                     # [ mMax x nCtg x nCtg ]
        valid=wb>0
        assert np.allclose(np.asarray(lAll)[valid],ref[valid],rtol=1e-10)
        # the cost: the batch's posteriors with the prior of all training stimuli
        w=np.asarray(unit.stim.weights)
        lp=ref+np.log(w.sum(0))[None,None]
        lpost=lp-logsumexp(lp,axis=-1,keepdims=True)
        own=lpost[:,np.arange(w.shape[1]),np.arange(w.shape[1])]
        assert np.isclose(cost,-np.sum(own*wb)/np.sum(wb),rtol=1e-10)

    def test_leave_one_out_uses_the_original_indices(self):
        """a batch stimulus's self match is its own index among the references, not its position in the batch"""
        unit=full_unit(nNeighbors=100,bLeaveOneOut=True)
        _,_,idx,wb,_,lAll=self.batch_parts(unit,jax.random.key(0))
        plain=full_unit(nNeighbors=100,bLeaveOneOut=False)
        _,_,_,_,_,lPlain=self.batch_parts(plain,jax.random.key(0))
        own=np.eye(lAll.shape[-1],dtype=bool)[None]&(np.asarray(wb)>0)[...,None]
        assert np.all(np.asarray(lAll)[own]<np.asarray(lPlain)[own])

    def test_within_values(self):
        x,s,y,_=ts.continuous_gaussian()
        st=ama.Stim.binned(x,s,y,nBins=4)
        unit=ama.Unit(st,ama.Nrn(),ama.Model('full','mean',nNeighbors=100,bWithin=True),ama.Objective('l2'),
                      ama.Optimizer(nIterMax=1,bVerbose=False))
        unit._finalize(2,np.arange(2),dtype=jnp.float64)
        f=np.random.default_rng(1).standard_normal(unit.filter._shape)
        unit.filter.out=jnp.asarray(f/np.linalg.norm(f,axis=0))
        _,_,idx,wb,_,lik=self.batch_parts(unit,jax.random.key(0),bWithin=True)
        _,Yc=lik
        exact=unit._lik_parts(unit._likelihoods(unit._nrn_out(),unit.stim.weights,unit.stim.Y,yRef=unit.stim.yCtg))[1]
        idx=np.asarray(idx)
        ref=np.asarray(exact)[idx,np.arange(idx.shape[1])[None,:]]
        valid=np.asarray(wb)>0
        assert np.allclose(np.asarray(Yc)[valid],ref[valid],rtol=1e-8)

    @pytest.mark.parametrize('opt',[dict(),dict(optimizerType='ama_sgd')])
    def test_training(self,opt):
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('full','mean',nNeighbors=8,nTail=4),ama.Objective('map'),
                      ama.Optimizer(**dict(dict(nIterMax=80,lRate0=0.05,batchSize=40,bVerbose=False,nStepsPerChunk=20),**opt)))
        unit.train_new(1)
        h=np.asarray(unit.optimizer.loss_hist)
        assert np.all(np.isfinite(h))
        assert np.mean(h[-20:])<np.mean(h[:20])
