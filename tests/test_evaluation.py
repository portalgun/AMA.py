"""
Held-out evaluation: stratified splits, decoding other stimuli with the training set, estimates and performance,
restarts, cross-validation, and save/load.
"""
import numpy as np
import pytest
import jax.numpy as jnp
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as smvn

import ama
import stimuli as ts
from test_ama import make_unit
from test_extended import unit_from, ungroup


#- helpers

def columns(st,c):
    """valid stimuli of category c as a sorted list of tuples"""
    w=np.asarray(st.weights)[:,c]>0
    return sorted(map(tuple,np.round(np.asarray(st.val)[:,w,c].T,10)))


def heldout_reference(unit,test):
    """numpy: decode test mean responses with the training stimuli (AMA-Gauss or full AMA, Eq 5 sum form)"""
    test=unit._prepare_stim(test)
    f=np.asarray(unit.filter.out_flat)
    resp=lambda st: ungroup(5.7*np.einsum('pf,pnc->fnc',f,np.asarray(st.val)),st.weights)
    rtr,ltr=resp(unit.stim)
    rte,lte=resp(test)
    Vtr=1.36*np.abs(rtr)+0.23
    Ntr=len(ltr)
    L=np.zeros((len(lte),ltr.max()+1))
    for i in range(L.shape[1]):
        m=ltr==i
        if unit.model.modelType=='gss':
            C=np.atleast_2d(np.cov(rtr[:,m]))+np.diag(Vtr[:,m].mean(1))
            L[:,i]=smvn(rtr[:,m].mean(1),C).logpdf(rte.T)+np.log(m.sum()/Ntr)
        else:
            LL=(-0.5*(((rte[:,:,None]-rtr[:,None,m])**2)/Vtr[:,None,m]).sum(0)
                -0.5*np.log(2*np.pi*Vtr[:,m]).sum(0)[None,:])
            L[:,i]=logsumexp(LL,axis=1)-np.log(Ntr)
    lp=L-logsumexp(L,1,keepdims=True)
    return -lp[np.arange(len(lte)),lte].mean()


def separable_stim(nPix=8,nCtg=4,nPer=6):
    rng=np.random.default_rng(0)
    base=np.eye(nPix)[:nCtg].T
    s=ts.contrast_normalize(np.repeat(base,nPer,axis=1)+1e-3*rng.standard_normal((nPix,nCtg*nPer)))
    x=ama.filt.X(ndim=1,n=nPix,totS=1)
    return ama.Stim(x,s,np.repeat(np.arange(nCtg),nPer),np.arange(nCtg)*10.),ts.contrast_normalize(base[:,:3])


#- splits

class TestSplits:
    def test_train_test_is_stratified_and_complete(self):
        x,s,ci,Y,info=ts.unequal_counts(counts=(10,30,20))
        st=ama.Stim(x,s,ci,Y)
        train,test=st.train_test(0.25,seed=1)
        for c,n in enumerate(info['counts']):
            nTest=int(np.asarray(test.weights)[:,c].sum())
            nTrain=int(np.asarray(train.weights)[:,c].sum())
            assert nTest==max(1,int(np.round(0.25*n))) and nTrain+nTest==n
            assert sorted(columns(train,c)+columns(test,c))==columns(st,c)
            assert np.allclose(np.asarray(test.yCtg)[np.asarray(test.weights)[:,c]>0,c],Y[c])
        pad=np.asarray(test.weights)==0
        assert np.all(np.asarray(test.val)[:,pad]==0)                        # padding stays zero

    def test_folds_test_every_stimulus_once(self):
        x,s,ci,Y,info=ts.unequal_counts(counts=(10,30,20))
        st=ama.Stim(x,s,ci,Y)
        folds=st.folds(4,seed=2)
        assert len(folds)==4
        for c in range(3):
            tested=sum((columns(te,c) for _,te in folds),[])
            assert sorted(tested)==columns(st,c)
            for tr,te in folds:
                assert sorted(columns(tr,c)+columns(te,c))==columns(st,c)


#- held-out evaluation

class TestHeldOut:
    @pytest.mark.parametrize('modelType',['gss','full'])
    def test_matches_reference(self,modelType):
        x,s,ci,Y,_=ts.unequal_counts(counts=(12,30,20))
        train,test=ama.Stim(x,s,ci,Y).train_test(0.3,seed=0)
        unit=unit_from(train,None,modelType)
        assert np.isclose(unit.evaluate(test),heldout_reference(unit,test),rtol=1e-8)

    @pytest.mark.parametrize('modelType',['gss','full'])
    @pytest.mark.parametrize('nrn_kw',[{},{'whitenType':'response'},{'rho':0.3}])
    def test_training_stimuli_give_the_training_loss(self,modelType,nrn_kw):
        x,s,ci,Y,_=ts.unequal_counts(counts=(12,30,20))
        train,_=ama.Stim(x,s,ci,Y).train_test(0.3,seed=0)
        unit=unit_from(train,ama.Nrn(**nrn_kw),modelType)
        assert np.isclose(unit.evaluate(train),float(unit.loss),rtol=1e-9)

    def test_other_categories_raise(self):
        x,s,ci,Y,_=ts.unequal_counts()
        unit=unit_from(ama.Stim(x,s,ci,Y))
        with pytest.raises(Exception,match='same categories'):
            unit.evaluate(ama.Stim(x,s,ci,Y+1))


#- estimates and performance

class TestEstimates:
    def test_separable_data_is_decoded_perfectly(self):
        stim,f=separable_stim()
        unit=unit_from(stim,ama.Nrn(rmax=50.),n=3,f=f)
        perf=unit.performance()
        assert np.all(perf['pCorrect']==1)
        assert np.array_equal(perf['confusion'],6*np.eye(4,dtype=int))
        assert np.allclose(perf['bias'],0) and np.allclose(perf['rmse'],0)
        w=np.asarray(unit.stim.weights)>0
        assert np.allclose(unit.estimates('mode')[w],np.asarray(unit.stim.yCtg)[w])

    def test_estimates_follow_the_posterior(self):
        unit,(s,ci,Y),_=make_unit()
        p=np.exp(np.asarray(unit.posterior))
        w=np.asarray(unit.stim.weights)>0
        assert np.allclose(unit.estimates('mean')[w],(p@np.asarray(unit.stim.Y))[w])

    def test_performance_cost_is_the_map_loss(self):
        unit,*_=make_unit()
        assert np.isclose(unit.performance()['cost'],float(unit.loss))

    def test_held_out_performance(self):
        x,s,ci,Y,_=ts.unequal_counts(counts=(12,30,20))
        train,test=ama.Stim(x,s,ci,Y).train_test(0.3,seed=0)
        unit=unit_from(train,None,'full')
        perf=unit.performance(stim=test)
        assert np.isclose(perf['cost'],unit.evaluate(test))
        assert np.array_equal(perf['confusion'].sum(1),np.asarray(test.weights).sum(0))


#- restarts and cross-validation

def fresh_unit(gen=ts.gaussian_ctg,nrn=None,model=None,**opt_kw):
    opt_kw.setdefault('nIterMax',15)
    return ama.Unit(ama.Stim(*gen()[:4]),nrn or ama.Nrn(),model or ama.Model('gss','mean'),ama.Objective('map'),
                    ama.Optimizer(lRate0=0.05,bVerbose=False,**opt_kw))


class TestRestarts:
    def test_keeps_the_best_run(self):
        unit=fresh_unit()
        unit.train_new(2,nRestarts=3)
        assert len(unit.restart_costs)==3
        assert np.isclose(float(unit.loss),min(unit.restart_costs))
        assert len(unit.optimizer.loss_hist)==15

    def test_one_restart_is_the_default(self):
        a,b=fresh_unit(),fresh_unit()
        a.train_new(2)
        b.train_new(2,nRestarts=1)
        assert np.array_equal(np.asarray(a.out),np.asarray(b.out)) and b.restart_costs is None

    def test_append_restarts_keep_fixed_filters(self):
        unit=fresh_unit()
        unit.train_new(1)
        f1=np.asarray(unit.out)[:,0].copy()
        unit.train_append(1,nRestarts=2)
        assert len(unit.restart_costs)==2
        assert np.allclose(np.asarray(unit.out)[:,0],f1)


def test_cross_validation_recovers_the_informative_direction():
    info=ts.gaussian_ctg()[4]
    unit=fresh_unit(nIterMax=150)
    res=unit.cross_validate(1,k=3)
    assert res['train'].shape==(3,) and res['test'].shape==(3,) and res['filters'].shape==(3,16,1)
    assert np.all(np.isfinite(res['test']))
    assert np.all(np.abs(res['filters'][:,:,0]@info['direction'])>0.9)


#- save/load

@pytest.mark.parametrize('gen,nrn_kw,train_kw',[
    (ts.gaussian_ctg,{},{}),
    (ts.sine_frequency,{},{'fourierType':2}),
    (ts.gaussian_ctg,{'whitenType':'response'},{}),
])
def test_save_load_roundtrip_and_resume(tmp_path,gen,nrn_kw,train_kw):
    def make():
        return fresh_unit(gen,ama.Nrn(**nrn_kw),ama.Model('full','mean',bLeaveOneOut=True),nIterMax=20,batchSize=40)
    unit=make()
    unit.train_new(2,**train_kw)
    if nrn_kw:
        unit.freeze_whitening()
    path=tmp_path/'unit.pkl'
    unit.save(path)
    loaded=ama.Unit.load(path,ama.Stim(*gen()[:4]))
    assert np.allclose(np.asarray(loaded.out),np.asarray(unit.out))
    assert np.isclose(float(loaded.loss),float(unit.loss))
    assert loaded.optimizer.loss_hist==unit.optimizer.loss_hist
    for u in (unit,loaded):
        u.train_recurse()
    assert np.allclose(np.asarray(loaded.out),np.asarray(unit.out))
