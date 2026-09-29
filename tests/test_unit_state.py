"""
Unit state across training calls, splitting, restarts, and save/load: shared stimuli, random keys, optimizer states,
validation histories, and training subsets.
"""
import numpy as np
import pytest
import jax.numpy as jnp

import ama
import stimuli as ts


def mk(stim=None,nrn=None,model=None,**opt_kw):
    if stim is None:
        x,s,ci,Y,_=ts.sine_frequency()
        stim=ama.Stim(x,s,ci,Y)
    opt={'nIterMax':60,'lRate0':0.05,'bVerbose':False,**opt_kw}
    return ama.Unit(stim,nrn or ama.Nrn(),model or ama.Model('gss','mean'),ama.Objective('map'),ama.Optimizer(**opt))


class TestSharedStim:
    def test_units_sharing_a_stim_do_not_change_each_other(self):
        x,s,ci,Y,_=ts.sine_frequency()
        st=ama.Stim(x,s,ci,Y)
        val0=np.asarray(st.val)
        a=mk(st)
        a.train_new(1)
        la=float(a.loss)
        b=mk(st)
        b.train_new(1,fourierType=2)
        assert not st.bIsFourier and np.array_equal(np.asarray(st.val),val0)   # the caller's Stim is untouched
        assert a.stim is not b.stim and not a.stim.bIsFourier
        assert float(a.loss)==la

    def test_float32_training_keeps_float64_stimuli(self):
        x,s,ci,Y,_=ts.sine_frequency()
        st=ama.Stim(x,s,ci,Y)
        a=mk(st,nIterMax=1)
        a.train_new(1,dtype=jnp.float32)
        assert st.val.dtype==jnp.float64
        b=mk(st,nIterMax=1)
        b.train_new(1,dtype=jnp.float64)
        assert np.array_equal(np.asarray(b.stim.val),np.asarray(st.val))


class TestSplit:
    def test_split_keeps_the_pooled_readout(self):
        u=mk(nrn=ama.Nrn(normalizeType='phase',readoutType='resultant'))
        u.train_new(2,fourierType=2)
        sp=u.split()
        assert sp.pool_p is not None
        assert np.isclose(float(sp.loss),float(u.loss))

    def test_split_does_not_change_the_parent(self):
        noisy=lambda: mk(nrn=ama.Nrn(bNoise_2=True),model=ama.Model('gss','basic'),nIterMax=50)
        a,b=noisy(),noisy()
        a.train_new(1)
        b.train_new(1)
        b.split()
        assert float(a.loss)==float(b.loss)
        a.train_new(1)
        b.train_new(1)
        assert np.array_equal(np.asarray(a.out),np.asarray(b.out))


class TestOptimizerState:
    def test_recurse_does_not_reuse_another_filters_state(self):
        u=mk()
        u.train_new(1)
        u.train_append(1)                                          # state of filter 1
        w=mk()
        w.train_new(1)
        w.train_append(1)
        w.opt_state=None                                           # a fresh state
        u.train_recurse(ind_rec=[0])
        w.train_recurse(ind_rec=[0])
        assert np.allclose(np.asarray(u.out),np.asarray(w.out))

    def test_recurse_continues_its_own_state(self):
        u=mk()
        u.train_new(1)
        key=u._opt_state_key
        u.train_recurse()
        assert u._opt_state_key==key
        w=mk()
        w.train_new(1)
        w.opt_state=None
        w.train_recurse()
        assert not np.allclose(np.asarray(u.out),np.asarray(w.out))  # the continued state was used


class TestRestarts:
    def test_restarts_with_validation_keep_the_kept_runs_history(self):
        x,s,ci,Y,_=ts.sine_frequency()
        tr,te=ama.Stim(x,s,ci,Y).train_test(0.3)
        u=mk(tr,nIterMax=200,nStepsPerChunk=20,patience=2)
        u.train_new(2,nRestarts=3,stimVal=te)
        kept=int(np.argmin(u.restart_costs))
        assert np.isclose(u.restart_costs[kept],u.evaluate(te))    # selected by the validation cost
        assert np.isclose(min(u.optimizer.val_hist),u.evaluate(te))


class TestSaveLoad:
    def test_training_subset_survives_save_and_load(self,tmp_path):
        x,s,ci,Y,_=ts.sine_frequency()
        u=mk()
        u.train_new(1,stimInd=np.arange(10))
        u.save(tmp_path/'u.pkl')
        v=ama.Unit.load(tmp_path/'u.pkl',ama.Stim(x,s,ci,Y))
        assert v.stim.nStim_Ctg==10
        assert np.isclose(float(v.loss),float(u.loss))
        t=ama.Stim(x,s,ci,Y)
        assert np.isclose(v.evaluate(t),u.evaluate(t))

    def test_generated_results_do_not_outlive_their_training_call(self):
        u=mk(nIterMax=5)
        u.train_multiscale(2)
        assert u.multiscale_out is not None
        u.train_new(3)
        assert u.multiscale_out is None


class TestGeneratedTraining:
    def test_repeated_training_reuses_the_compiled_steps(self):
        mk(nIterMax=20).train_parametric(2)
        n=ama.Optimizer._run_generated_chunk._cache_size()
        mk(nIterMax=20).train_parametric(2)
        mk(nIterMax=20).train_multiscale(2)
        m=ama.Optimizer._run_generated_chunk._cache_size()
        mk(nIterMax=20).train_multiscale(2)
        assert ama.Optimizer._run_generated_chunk._cache_size()==m and m<=n+1

    def test_batches_and_best_step_match_train_new(self):
        x,s,ci,Y,_=ts.sine_frequency()
        tr,te=ama.Stim(x,s,ci,Y).train_test(0.3)
        for meth in ('train_new','train_parametric'):
            u=mk(tr,nIterMax=40,nStepsPerChunk=20,patience=5,batchSize=30)
            getattr(u,meth)(2,stimVal=te)
            assert u.optimizer.best_step in (20,40)                    # iterations completed at the kept chunk
            assert len(u.optimizer.loss_hist)==40
            assert u.rng_last is not None


class TestReviewRegressions:
    def test_restarts_start_from_the_same_response_parameters(self,monkeypatch):
        seen=[]
        orig=ama.Unit._p0
        def spy(self):
            p=orig(self)
            seen.append({k:np.asarray(v).copy() for k,v in p.items()})
            return p
        monkeypatch.setattr(ama.Unit,'_p0',spy)
        u=mk(nrn=ama.Nrn(bBias=True),nIterMax=20)
        u.train_new(2,nRestarts=3)
        assert len(seen)==3
        assert all(np.array_equal(p['bias'],seen[0]['bias']) for p in seen)

    def test_config_replay_uses_the_optimizer_of_each_call(self):
        x,s,ci,Y,_=ts.sine_frequency()
        st=ama.Stim(x,s,ci,Y)
        u=mk(st,nIterMax=30)
        u.train_new(2,dtype=jnp.float64)
        u.train_recurse(optimizer=ama.Optimizer(nIterMax=3,lRate0=0.01,bVerbose=False))
        assert u.config()['train'][0]['args']['optimizer']['nIterMax']==30
        w=ama.Unit.from_config(u.config(),st,bTrain=True)
        assert np.isclose(float(w.loss),float(u.loss),rtol=1e-9) and np.allclose(np.asarray(w.out),np.asarray(u.out))

    @pytest.mark.parametrize('first,then',[(dict(fourierType=1),dict(fourierType=0)),(dict(fourierType=1),dict(fourierType=2)),
                                           (dict(bSplit=True),dict(bSplit=False))])
    def test_kept_filters_can_not_change_domain_or_layout(self,first,then):
        x,s,ci,Y,_=ts.binocular_shift() if 'bSplit' in first else ts.sine_frequency()
        u=mk(ama.Stim(x,s,ci,Y,nSplit=2) if 'bSplit' in first else ama.Stim(x,s,ci,Y),nIterMax=5)
        u.train_new(2,**first)
        with pytest.raises(Exception,match='can not change fourierType or bSplit'):
            u.train_append(1,**then)
        with pytest.raises(Exception,match='can not change fourierType or bSplit'):
            u.train_recurse(**then)
        u.train_append(1)                                                    # the same domain still works

    def test_optimizer_state_follows_the_lbfgs_memory(self):
        u=mk(nIterMax=5,optimizerType='lbfgs')
        u.train_new(2)
        k=u._opt_state_key
        u.optimizer=ama.Optimizer('lbfgs',nIterMax=5,lbfgsMemory=4,bVerbose=False)
        assert u._opt_key(u._opt_param_shape)!=k
