"""
Full AMA with random reference subsets (Model nRef): the training cost against a brute-force computation over the
drawn references, the limit of all references, leave-one-out, and learning.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.special import logsumexp

import ama
import stimuli as ts


def full_unit(nRef=None,bLeaveOneOut=False,gen=ts.unequal_counts,n=2,seed=1,**opt_kw):
    x,s,ci,Y,_=gen()
    opt={'nIterMax':1,'bVerbose':False,**opt_kw}
    unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('full','mean',bLeaveOneOut=bLeaveOneOut,nRef=nRef),
                  ama.Objective('map'),ama.Optimizer(**opt))
    unit._finalize(n,np.arange(n),dtype=jnp.float64)
    f=np.random.default_rng(seed).standard_normal(unit.filter._shape)
    unit.filter.out=jnp.asarray(f/np.linalg.norm(f,axis=0))
    return unit


def lrn_likelihoods(unit,key):
    """the likelihoods the training cost uses with this key"""
    f=unit.filter.out_flat[unit.filter._insert_index_jx]
    out=unit.nrn.lrn_main(key,unit.stim.val,f,unit.filter.prepped_jx,unit.filter._insert_index_jx,unit.stim.weights)
    return unit._likelihoods(out,unit.stim.weights,unit.stim.Y,jax.random.fold_in(key,7) if unit._bRefSubset() else None),out


def lrn_cost(unit,key):
    return float(unit._loss_fun_lrn({'f':unit.filter.out_flat[unit.filter._insert_index_jx]},key,unit.filter.prepped_jx,
                                    unit.filter._insert_index_jx,unit.stim.val,unit.stim.weights,unit.stim.yCtg,unit.stim.Y))


class TestReferenceSubsets:
    @pytest.mark.parametrize('bLeaveOneOut',[False,True])
    def test_all_references_is_full_ama(self,bLeaveOneOut):
        exact=full_unit(bLeaveOneOut=bLeaveOneOut)
        sub=full_unit(nRef=1000,bLeaveOneOut=bLeaveOneOut)
        key=jax.random.key(3)
        assert np.isclose(lrn_cost(sub,key),lrn_cost(exact,key),rtol=1e-10)

    @pytest.mark.parametrize('bLeaveOneOut',[False,True])
    def test_matches_brute_force_over_the_drawn_references(self,bLeaveOneOut):
        unit=full_unit(nRef=4,bLeaveOneOut=bLeaveOneOut)
        key=jax.random.key(5)
        lAll,out=lrn_likelihoods(unit,key)
        lAll=np.asarray(lAll)
        idx=np.asarray(ama.Model._ref_subset(jax.random.fold_in(key,7),unit.stim.weights,4))
        R,Rm,RVar=[np.asarray(ama._flatten_responses(x)) for x in unit.model._response_fun(*out)]
        w=np.asarray(unit.stim.weights)
        for (l,k,i) in [(0,0,0),(3,1,1),(7,2,0),(2,0,2)]:
            terms=[]
            for j in idx[:,i]:
                if w[j,i]==0 or (bLeaveOneOut and i==k and j==l):
                    continue
                v=RVar[:,j,i]
                terms.append(-0.5*np.sum((R[:,l,k]-Rm[:,j,i])**2/v)-0.5*np.sum(np.log(2*np.pi*v)))
            ref=logsumexp(terms)-np.log(w[idx[:,i],i].sum())
            assert np.isclose(lAll[l,k,i],ref)
        assert len(np.unique(idx[:,1]))==4 and np.all(w[idx[:,1],1]>0)   # distinct, valid references

    def test_more_references_approach_the_exact_cost(self):
        exact=lrn_cost(full_unit(gen=ts.gaussian_ctg),jax.random.key(0))
        err={}
        for nRef in (3,25):
            unit=full_unit(nRef=nRef,gen=ts.gaussian_ctg)
            err[nRef]=abs(np.mean([lrn_cost(unit,jax.random.key(i)) for i in range(20)])-exact)
        assert err[25]<err[3]

    def test_evaluation_uses_all_references(self):
        assert float(full_unit(nRef=3).loss)==float(full_unit().loss)

    def test_learning(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        costs={}
        for nRef in (None,8):
            unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('full','mean',nRef=nRef),ama.Objective('map'),
                          ama.Optimizer(nIterMax=150,lRate0=0.05,bVerbose=False))
            unit.train_new(1)
            h=np.asarray(unit.optimizer.loss_hist)
            assert np.all(np.isfinite(h)) and h[-1]<h[0]
            costs[nRef]=float(unit.loss)
        assert costs[8]<costs[None]+0.05                                   # nearly as good, by the exact cost

    def test_validation(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('gss','mean',nRef=5),ama.Objective('map'),
                      ama.Optimizer(nIterMax=1,bVerbose=False))
        with pytest.raises(Exception,match='nRef'):
            unit._finalize(1,[0])


class TestChunks:
    @pytest.mark.parametrize('bLeaveOneOut',[False,True])
    def test_chunked_evaluation_is_identical(self,bLeaveOneOut,monkeypatch):
        unit=full_unit(bLeaveOneOut=bLeaveOneOut)
        out=unit.nrn.main(unit.rng,unit.stim.val,unit.filter.out_flat,unit.stim.weights)
        R,Rm,RVar=[ama._flatten_responses(x) for x in unit.model._response_fun(*out)]
        args=(R,Rm,RVar,None,None,unit.stim.weights,bLeaveOneOut,unit.model,unit.stim.Y)
        whole=np.asarray(ama.Model._model__full(*args))
        monkeypatch.setattr(ama,'_FULL_CHUNK',7*30*3)              # 7 observed stimuli per chunk, 30 x 3 references
        chunked=np.asarray(ama.Model._model__full(*args))
        assert np.array_equal(np.isinf(whole),np.isinf(chunked))
        assert np.allclose(whole[np.isfinite(whole)],chunked[np.isfinite(chunked)],rtol=1e-12)
