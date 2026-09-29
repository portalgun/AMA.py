"""
Research extensions of the response model and the optimizer, against independent computations: saturating and softmax
activations, learned offsets (bBias), ON/OFF channels (bSplitNegatives), normalization pools (normPool, bLearnNormPool),
a learned second layer (readoutType 'linear'), a learned phase for generated filter banks (bPhase), L-BFGS and
fmincon-like stopping tolerances, and complex factor analysis for 'circ' (covRank).
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import ama
import stimuli as ts


def unit_for(nrn=None,gen=ts.gaussian_ctg,model=None,objective=None,n=2,fourierType=0,finalize=True,seed=3,**opt_kw):
    x,s,ci,Y,_=gen()
    opt={'nIterMax':1,'bVerbose':False,'lRate0':0.05,**opt_kw}
    unit=ama.Unit(ama.Stim(x,s,ci,Y),nrn or ama.Nrn(),model or ama.Model('gss','mean'),objective or ama.Objective('map'),
                  ama.Optimizer(**opt))
    if finalize:
        dtype=jnp.complex128 if fourierType else jnp.float64
        unit._finalize(n,np.arange(n),fourierType=fourierType,dtype=dtype)
        rng=np.random.default_rng(seed)
        f=rng.standard_normal(unit.filter._shape)
        if fourierType:
            f=f+1j*rng.standard_normal(unit.filter._shape)
        f=f/np.linalg.norm(f.reshape(-1,n),axis=0)
        unit.filter.out=jnp.reshape(jnp.asarray(f,dtype=dtype),unit.filter._shape_exp)
    return unit


def linear(unit):
    """linear responses rmax f^H s of a unit's filters (no activation, noise, normalization)"""
    base=ama.Nrn(rmax=unit.nrn.rmax)
    base._finalize(unit.stim,unit.nrn.dtype,unit.filter.n,np.arange(unit.filter.n),(),(),bFourier=unit.nrn.bFourier,
                   bAnalytic=bool(unit.nrn.bAnalytic),bSplit=unit.nrn.bSplit)
    return np.asarray(base.main(unit.rng,unit.stim.val,unit.filter.out_flat)[0])


def responses(unit,p=None):
    return [np.asarray(v) for v in unit.nrn.main(unit.rng,unit.stim.val,unit.filter.out_flat,unit.stim.weights,None,p)]


def train(unit,n=2,**kw):
    unit.train_new(n,**kw)
    h=np.asarray(unit.optimizer.loss_hist)
    assert np.all(np.isfinite(h)) and h[-1]<h[0]
    return unit


class TestActivations:
    def test_ramp_and_naka_rushton(self):
        u=unit_for(ama.Nrn(activationType='ramp',rsat=1.5))
        r=linear(u)
        assert np.allclose(responses(u)[0],np.clip(r,0,1.5))
        u=unit_for(ama.Nrn(activationType='ramp'))
        assert np.allclose(responses(u)[0],np.clip(r,0,5.7/2))                   # default rsat = rmax/2
        u=unit_for(ama.Nrn(activationType='nakarushton',c50=0.8,nNR=2.5))
        rp=np.maximum(r,0)**2.5
        assert np.allclose(responses(u)[0],5.7*rp/(rp+0.8**2.5))

    def test_naka_rushton_below_one_has_finite_gradients(self):
        # x**nNR with nNR < 1 has an infinite derivative at 0; the rectified branch must not turn it into NaN
        u=unit_for(ama.Nrn(activationType='nakarushton',nNR=0.5))
        r=linear(u)
        rp=np.maximum(r,0)**0.5
        assert np.allclose(responses(u)[0],5.7*rp/(rp+(5.7/4)**0.5))
        train(unit_for(ama.Nrn(activationType='nakarushton',nNR=0.5),finalize=False,nIterMax=20))

    def test_softmax_over_filters(self):
        u=unit_for(ama.Nrn(activationType='softmax',softmaxT=2.),n=3)
        r=linear(u)
        e=np.exp(r/2.)
        assert np.allclose(responses(u)[0],5.7*e/e.sum(0,keepdims=True))

    def test_softmax_needs_real_responses(self):
        with pytest.raises(Exception,match='softmax'):
            unit_for(ama.Nrn(activationType='softmax'),gen=ts.sine_frequency,fourierType=2)

    def test_components_of_quadrature_pairs(self):
        u=unit_for(ama.Nrn(activationType='ramp',rsat=1.),gen=ts.sine_frequency,fourierType=2)
        r=linear(u)
        assert np.allclose(responses(u)[0],np.clip(r.real,0,1)+1j*np.clip(r.imag,0,1))


class TestBias:
    def test_offsets_before_the_activation(self):
        u=unit_for(ama.Nrn(activationType='relu',bBias=True),n=3)
        p=u._p0()
        assert set(p)=={'bias'} and p['bias'].shape==(1,3)
        b=np.array([[0.5,-1.,2.]])
        r=linear(u)
        assert np.allclose(responses(u,{'bias':jnp.asarray(b)})[0],np.maximum(r+b[0][:,None,None],0))

    def test_complex_and_split_offsets(self):
        u=unit_for(ama.Nrn(bBias=True),gen=ts.sine_frequency,fourierType=2)
        assert u._p0()['bias'].shape==(2,2)
        b=np.array([[0.3,-0.2],[1.,0.5]])
        r=linear(u)
        assert np.allclose(responses(u,{'bias':jnp.asarray(b)})[0],r+(b[0]+1j*b[1])[:,None,None])
        x,s,ci,Y,_=ts.binocular_shift()
        u=ama.Unit(ama.Stim(x,s,ci,Y,nSplit=2),ama.Nrn(bBias=True),ama.Model('gss','mean'),ama.Objective('map'),
                   ama.Optimizer(nIterMax=1,bVerbose=False))
        u._finalize(2,np.arange(2),bSplit=True)
        assert u._p0()['bias'].shape==(1,2,2)

    def test_learned_saved_and_extended(self,tmp_path):
        u=train(unit_for(ama.Nrn(activationType='relu',bBias=True),finalize=False,nIterMax=150))
        assert u.nrn_p is not None and not np.allclose(u.nrn_p['bias'],0)
        fname=str(tmp_path/'u.pkl')
        u.save(fname)
        back=ama.Unit.load(fname,u.stim_full)
        assert np.allclose(back.nrn_p['bias'],u.nrn_p['bias']) and np.isclose(float(back.loss),float(u.loss))
        old=np.array(u.nrn_p['bias'])
        u.train_append(1)
        assert u.nrn_p['bias'].shape==(1,3)
        u2=unit_for(ama.Nrn(activationType='relu',bBias=True),finalize=False)
        u2.nrn_p={'bias':old}; u2.nrn._finalize(u2.stim,jnp.float32,3,np.arange(3),(),())
        assert np.allclose(u2._p0()['bias'][:,:2],old) and np.allclose(u2._p0()['bias'][:,2],0)


class TestSplitNegatives:
    def test_on_off_channels(self):
        u=unit_for(ama.Nrn(bSplitNegatives=True),n=2)
        r=linear(u)
        out=responses(u)[0]
        assert out.shape[0]==4 and np.allclose(out,np.concatenate((np.maximum(r,0),np.maximum(-r,0))))
        assert u._nDim==4
        u=unit_for(ama.Nrn(bSplitNegatives=True),gen=ts.sine_frequency,fourierType=2)
        r=linear(u)
        on=lambda x: np.maximum(x,0)
        ref=np.concatenate((on(r.real)+1j*on(r.imag),on(-r.real)+1j*on(-r.imag)))
        assert np.allclose(responses(u)[0],ref) and u._nDim==8

    def test_training_and_validation(self):
        train(unit_for(ama.Nrn(bSplitNegatives=True),finalize=False,nIterMax=100))
        for nrn,match in [(ama.Nrn(bSplitNegatives=True,normalizeType='narrow'),'narrow'),
                          (ama.Nrn(bSplitNegatives=True,readoutType='resultant'),'pooled')]:
            with pytest.raises(Exception,match=match):
                unit_for(nrn,gen=ts.sine_frequency,fourierType=2)


class TestNormalizationPool:
    def test_all_ones_is_plain_gen_and_identity_is_self_normalization(self):
        n=3
        plain=unit_for(ama.Nrn(normalizeType='gen'),n=n)
        ones=unit_for(ama.Nrn(normalizeType='gen',normPool=np.ones((n,n))),n=n)
        assert np.allclose(responses(ones)[2],responses(plain)[2])
        own=unit_for(ama.Nrn(normalizeType='gen',normPool=np.eye(n),eps=0.1),n=n)
        r=linear(own)
        assert np.allclose(responses(own)[2],r/(0.1+np.abs(r)))

    def test_pool_over_split_parts(self):
        x,s,ci,Y,_=ts.binocular_shift()
        M=np.array([[1.,0.5],[0.,2.]])
        u=ama.Unit(ama.Stim(x,s,ci,Y,nSplit=2),ama.Nrn(normalizeType='gen',normPool=M),ama.Model('gss','mean'),
                   ama.Objective('map'),ama.Optimizer(nIterMax=1,bVerbose=False))
        u._finalize(2,np.arange(2),bSplit=True,dtype=jnp.float64)
        f=np.random.default_rng(0).standard_normal(u.filter._shape)
        u.filter.out=jnp.reshape(jnp.asarray(f/np.linalg.norm(f.reshape(-1,2),axis=0)),u.filter._shape_exp)
        r=linear(u)                                                          # [ nF x nSplit x ... ]
        a=np.abs(r).sum(1)                                                   # summed over the eyes
        D=u.nrn.eps+np.einsum('ij,jsc->isc',M,a)
        assert np.allclose(responses(u)[2],r/D[:,None])

    def test_stage_one_noise_variance_matches_the_jacobian(self):
        n=3
        M=np.array([[1.,0.2,0.],[0.5,1.,0.3],[0.,0.7,2.]])
        u=unit_for(ama.Nrn(normalizeType='gen',normPool=M,bNoise_1=True),n=n)
        r,_,R,_,RVar=responses(u)
        v=u.nrn.fano*np.abs(r)+u.nrn.var0
        for (l,k) in [(0,0),(5,2),(11,4)]:
            norm=lambda x: x/(u.nrn.eps+M@np.abs(x))
            J=np.asarray(jax.jacfwd(lambda x: x/(u.nrn.eps+jnp.asarray(M)@jnp.abs(x)))(jnp.asarray(r[:,l,k])))
            assert np.allclose(norm(r[:,l,k]),R[:,l,k])
            assert np.allclose(RVar[:,l,k],(J**2)@v[:,l,k])

    def test_float32_stays_float32_with_a_pool(self):
        # a float64 normPool (under jax x64) must not promote float32 responses and their variances
        M=np.array([[1.,0.2],[0.5,1.]])
        x,s,ci,Y,_=ts.gaussian_ctg()
        u=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(normalizeType='gen',normPool=M,bNoise_1=True),ama.Model('gss','mean'),
                   ama.Objective('map'),ama.Optimizer(nIterMax=1,bVerbose=False))
        u._finalize(2,np.arange(2),dtype=jnp.float32)
        assert all(np.asarray(v).dtype==np.float32 for v in responses(u))

    def test_learned_pool(self):
        M=np.array([[1.,0.5],[0.5,1.]])
        u=unit_for(ama.Nrn(normalizeType='gen',normPool=M,bLearnNormPool=True))
        p=u._p0()
        assert np.allclose(np.asarray(u.nrn._norm_pool(p)),M)                 # starts at normPool
        u=train(unit_for(ama.Nrn(normalizeType='gen',bLearnNormPool=True),finalize=False,nIterMax=100))
        assert not np.allclose(u.nrn_p['norm'],u.nrn.params0()['norm'])

    def test_validation(self):
        with pytest.raises(Exception,match='normPool'):
            unit_for(ama.Nrn(normalizeType='broad',normPool=np.eye(2)))
        with pytest.raises(Exception,match='3 x 3'):
            unit_for(ama.Nrn(normalizeType='gen',normPool=np.eye(2)),n=3)


class TestSecondLayer:
    def test_output_is_the_normalized_combination(self):
        u=unit_for(ama.Nrn(readoutType='linear',nReadout=4,readoutActivation='relu',bBias=True),n=3)
        p=u._p0()
        assert p['A'].shape==(4,3) and p['bias2'].shape==(1,4) and u._nDim==4
        rng=np.random.default_rng(0)
        A=rng.standard_normal((4,3)); b2=rng.standard_normal((1,4)); b=np.zeros((1,3))
        R=responses(u,{'A':jnp.asarray(A),'bias2':jnp.asarray(b2),'bias':jnp.asarray(b)})[2]
        An=A/np.linalg.norm(A,axis=1,keepdims=True)
        assert np.allclose(R,np.maximum(np.einsum('oi,isc->osc',An,linear(u))+b2[0][:,None,None],0))

    def test_complex_and_split_inputs(self):
        u=unit_for(ama.Nrn(readoutType='linear',nReadout=3),gen=ts.sine_frequency,fourierType=2)
        A=np.random.default_rng(1).standard_normal((3,2))
        R=responses(u,{'A':jnp.asarray(A)})[2]
        An=A/np.linalg.norm(A,axis=1,keepdims=True)
        r=linear(u)
        assert np.allclose(R,np.einsum('oi,isc->osc',An,r.real)+1j*np.einsum('oi,isc->osc',An,r.imag)) and u._nDim==6
        x,s,ci,Y,_=ts.binocular_shift()
        u=ama.Unit(ama.Stim(x,s,ci,Y,nSplit=2),ama.Nrn(readoutType='linear',nReadout=2),ama.Model('gss','mean'),
                   ama.Objective('map'),ama.Optimizer(nIterMax=1,bVerbose=False))
        u._finalize(2,np.arange(2),bSplit=True)
        assert u._p0()['A'].shape==(2,4) and u._nDim==2                     # mixes filters and eyes

    def test_training_and_saving(self,tmp_path):
        u=train(unit_for(ama.Nrn(activationType='relu',readoutType='linear',nReadout=2,readoutActivation='softplus'),
                         finalize=False,nIterMax=150),n=3)
        fname=str(tmp_path/'u.pkl')
        u.save(fname)
        back=ama.Unit.load(fname,u.stim_full)
        assert np.isclose(float(back.loss),float(u.loss))

    def test_validation(self):
        with pytest.raises(Exception,match='readoutType'):
            unit_for(ama.Nrn(nReadout=3))
        with pytest.raises(Exception,match='readoutActivation'):
            ama.Nrn(readoutType='linear',readoutActivation='nonsense')


class TestDeepReadout:
    def test_stacked_layers(self):
        u=unit_for(ama.Nrn(readoutType='linear',nReadout=(5,3),readoutActivation='relu',bBias=True),n=3)
        p=u._p0()
        assert p['A'].shape==(5,3) and p['A_1'].shape==(3,5) and p['bias2_1'].shape==(1,3) and u._nDim==3
        rng=np.random.default_rng(0)
        A,A1=rng.standard_normal((5,3)),rng.standard_normal((3,5))
        b2,b21=rng.standard_normal((1,5)),rng.standard_normal((1,3))
        R=responses(u,{'A':jnp.asarray(A),'A_1':jnp.asarray(A1),'bias2':jnp.asarray(b2),'bias2_1':jnp.asarray(b21),
                       'bias':jnp.zeros((1,3))})[2]
        nrm=lambda M: M/np.linalg.norm(M,axis=1,keepdims=True)
        h=np.maximum(np.einsum('oi,isc->osc',nrm(A),linear(u))+b2[0][:,None,None],0)
        assert np.allclose(R,np.maximum(np.einsum('oi,isc->osc',nrm(A1),h)+b21[0][:,None,None],0))

    def test_one_width_is_the_second_layer(self):
        one=unit_for(ama.Nrn(readoutType='linear',nReadout=4,readoutActivation='relu'),n=3)
        tup=unit_for(ama.Nrn(readoutType='linear',nReadout=(4,),readoutActivation='relu'),n=3)
        assert np.allclose(responses(one,one._p0())[2],responses(tup,tup._p0())[2])

    def test_complex_layers(self):
        u=unit_for(ama.Nrn(readoutType='linear',nReadout=(3,2)),gen=ts.sine_frequency,fourierType=2)
        R=responses(u,u._p0())[2]
        assert u._nDim==4 and np.iscomplexobj(R) and R.shape[0]==2 and np.all(np.isfinite(R))

    def test_training_config_and_saving(self,tmp_path):
        u=train(unit_for(ama.Nrn(activationType='relu',readoutType='linear',nReadout=(4,2),readoutActivation='softplus'),
                         finalize=False,nIterMax=150),n=3)
        back=ama.Unit.from_config(u.config(),u.stim_full)
        assert back.nrn.nReadout==(4,2)
        fname=str(tmp_path/'u.pkl')
        u.save(fname)
        assert np.isclose(float(ama.Unit.load(fname,u.stim_full).loss),float(u.loss))

    def test_validation(self):
        with pytest.raises(Exception,match='nReadout'):
            ama.Nrn(readoutType='linear',nReadout=(3,0))


class TestPhase:
    def test_phase_rotates_the_generated_spectra(self):
        u=unit_for(gen=ts.sine_frequency,finalize=False)
        u.train_parametric(3,family='loggabor',fourierType=1,bPhase=True)
        params=u._parametric_init('loggabor',3,True,None,False)
        cfg=dict(family='loggabor',n=3,bTied=True,orientations=np.zeros(3))
        base=np.asarray(u._generated_filters('parametric',params,cfg))
        ph=np.array([0.,np.pi/2,1.])
        rot=np.asarray(u._generated_filters('parametric',dict(params,phase=jnp.asarray(ph)),cfg))
        assert np.allclose(rot,base*np.exp(1j*ph)[None,:])
        assert 'phase' in u.param_out and u.param_out['phase'].shape==(3,)

    def test_phase_is_learned(self):
        # a single real filter (fourierType=1) of a zero-phase profile: the phase selects its symmetry, and learning moves it
        u=unit_for(gen=ts.sine_frequency,finalize=False,nIterMax=150)
        u.train_parametric(2,family='loggabor',fourierType=1,bPhase=True)
        assert np.all(np.isfinite(u.optimizer.loss_hist)) and np.any(np.abs(u.param_out['phase'])>1e-3)
        u2=unit_for(gen=ts.sine_frequency,finalize=False,nIterMax=20)
        u2.train_multiscale(2,nKnot=9,bPhase=True)
        assert u2.multiscale_out['phase'].shape==(2,)


class TestOptimizer:
    def test_lbfgs(self):
        costs={}
        for opt in ('adam','lbfgs'):
            u=unit_for(finalize=False,optimizerType=opt,nIterMax=40 if opt=='lbfgs' else 300,lRate0=0.02)
            u.train_new(2)
            f=np.asarray(u.out).reshape(-1,2)
            assert np.allclose(np.linalg.norm(f,axis=0),1)
            costs[opt]=float(u.loss)
        assert costs['lbfgs']<=costs['adam']+1e-3

    def test_lbfgs_fourier_and_generated(self):
        u=unit_for(gen=ts.sine_frequency,finalize=False,optimizerType='lbfgs',nIterMax=20)
        train(u,1,fourierType=2)
        u.train_parametric(2,family='loggabor')
        assert np.all(np.isfinite(u.optimizer.loss_hist))

    def test_lbfgs_needs_full_batches(self):
        with pytest.raises(Exception,match='batchSize'):
            ama.Optimizer('lbfgs',batchSize=50)

    def test_tolerances(self):
        u=unit_for(finalize=False,optimizerType='lbfgs',nIterMax=2000,nStepsPerChunk=10,tolFun=1e-7)
        u.train_new(1)
        assert u.optimizer.stop_reason=='tolFun' and len(u.optimizer.loss_hist)<2000
        u=unit_for(finalize=False,nIterMax=5000,nStepsPerChunk=50,tolX=1e-4,lRate0=0.01,optimizerType='sgd')
        u.train_new(1)
        assert u.optimizer.stop_reason=='tolX' and len(u.optimizer.loss_hist)<5000
        u=unit_for(finalize=False,nIterMax=20)
        u.train_new(1)
        assert u.optimizer.stop_reason=='nIterMax'


class TestCircLowRank:
    def test_complex_factor_analysis_recovers_a_low_rank_model(self):
        rng=np.random.default_rng(0)
        p,r=5,2
        L=rng.standard_normal((p,r))+1j*rng.standard_normal((p,r))
        S=L@L.conj().T+np.diag(rng.uniform(0.5,1.5,p))
        fit=np.asarray(ama.Model._low_rank(jnp.asarray(S),ama.Model('circ',covRank=r,nFA=2000)))
        assert np.allclose(fit,fit.conj().T) and np.allclose(fit,S,atol=1e-6)

    def test_circ_training(self):
        u=unit_for(ama.Nrn(),gen=ts.sine_frequency,model=ama.Model('circ','mean',covRank=1),finalize=False,nIterMax=100)
        train(u,3,fourierType=2)
        with pytest.raises(Exception,match='complex response dimensions'):
            unit_for(gen=ts.sine_frequency,model=ama.Model('circ','mean',covRank=2),fourierType=2)
