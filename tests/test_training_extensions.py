"""
Training extensions: filter regularization, phase normalization and pooled (congruency) readouts with learned weights,
and early stopping on validation stimuli.
"""
import numpy as np
import pytest
import jax.numpy as jnp

import ama
import stimuli as ts


def unit_for(gen=ts.sine_frequency,nrn=None,objective=None,model=None,nIterMax=100,**opt_kw):
    x,s,ci,Y,_=gen()
    return ama.Unit(ama.Stim(x,s,ci,Y),nrn or ama.Nrn(),model or ama.Model('gss','mean'),objective or ama.Objective('map'),
                    ama.Optimizer(nIterMax=nIterMax,lRate0=0.05,bVerbose=False,**opt_kw))


#- regularization

class TestRegularization:
    def test_penalties_match_numpy_in_the_spatial_domain(self):
        for reg in ('l1','smooth'):
            unit=unit_for(ts.image_orientation,objective=ama.Objective('map',regType=reg,regWeight=0.1),nIterMax=1)
            unit.train_new(2)
            F=np.asarray(unit.out)                                          # [ 8 x 8 x 2 ]
            if reg=='l1':
                ref=np.abs(F).sum()/2
            else:
                ref=(np.sum(np.diff(F,axis=0)**2)+np.sum(np.diff(F,axis=1)**2))/2
            assert np.isclose(unit.penalty,ref)

    def test_smooth_penalty_uses_only_learned_fourier_coefficients(self):
        unit=unit_for(objective=ama.Objective('map',regType='smooth',regWeight=0.1),nIterMax=1)
        unit.train_new(1,fourierType=2)
        f=np.asarray(unit.out)[:,0]                                         # centered 1D spectrum, 32 bins
        pix=unit.filter.index.pix
        keep=np.zeros(len(f),bool); keep[pix]=True
        valid=keep[:-1]&keep[1:]
        ref=np.sum(np.abs(np.diff(f))**2*valid)
        assert np.isclose(unit.penalty,ref)

    def test_loss_excludes_penalty_and_l1_sparsifies_spectra(self):
        def train(weight):
            unit=unit_for(objective=ama.Objective('map',regType='l1',regWeight=weight),nIterMax=300)
            unit.train_new(2,fourierType=2)
            return unit
        plain,sparse=train(0.),train(0.05)
        count=lambda u: int((np.abs(np.asarray(u.out))>0.05).sum())
        assert count(sparse) < count(plain)
        # Unit.loss is the data cost only
        ref=sparse._loss_fun(sparse._params_out(),sparse.rng,sparse.stim.val,sparse.stim.weights,sparse.stim.yCtg,sparse.stim.Y,None)
        assert np.isclose(float(sparse.loss),float(ref))

    def test_invalid_regType_raises(self):
        with pytest.raises(Exception):
            ama.Objective('map',regType='l2')


#- phase normalization and pooled readouts

class TestPhaseAndReadout:
    def test_phase_responses_are_unit_phasors_and_contrast_invariant(self):
        unit=unit_for(nrn=ama.Nrn(normalizeType='phase',eps=1e-9),nIterMax=1)
        unit.train_new(2,fourierType=2)
        f=unit.filter.out_flat
        _,_,R,_,_=unit.nrn.main(unit.rng,unit.stim.val,f)
        w=np.asarray(unit.stim.weights)>0
        mag=np.abs(np.asarray(R))[:,w]
        assert np.allclose(mag,1,atol=1e-6)
        _,_,R3,_,_=unit.nrn.main(unit.rng,3*unit.stim.val,f)
        assert np.allclose(np.asarray(R3)[:,w],np.asarray(R)[:,w],atol=1e-6)

    @pytest.mark.parametrize('readout',['resultant','resultant_only'])
    def test_readout_is_the_weighted_resultant(self,readout):
        unit=unit_for(nrn=ama.Nrn(normalizeType='phase',readoutType=readout),nIterMax=1)
        unit.train_new(3,fourierType=2)
        f=unit.filter.out_flat
        p=jnp.asarray([0.3,-1.0,2.0])
        _,_,R,_,_=unit.nrn.main(unit.rng,unit.stim.val,f,None,None,p)
        base=ama.Nrn(normalizeType='phase')
        base._finalize(unit.stim,unit.nrn.dtype,3,np.arange(3),(),(),bFourier=True,bAnalytic=True)
        _,_,U,_,_=base.main(unit.rng,unit.stim.val,f)
        w=np.log1p(np.exp(np.asarray(p))); w=w/w.sum()
        pooled=np.tensordot(w,np.asarray(U),axes=(0,0))
        R=np.asarray(R)
        if readout=='resultant':
            assert R.shape[0]==4 and np.allclose(R[:3],np.asarray(U)) and np.allclose(R[3],pooled)
        else:
            assert R.shape[0]==1 and np.allclose(R[0],pooled)
        assert unit._nDim==(8 if readout=='resultant' else 2)

    def test_pooling_weights_are_learned_saved_and_loaded(self,tmp_path):
        unit=unit_for(nrn=ama.Nrn(normalizeType='phase',readoutType='resultant'),model=ama.Model('circ','mean',circMean='zero'),nIterMax=150)
        unit.train_new(3,fourierType=2)
        hist=unit.optimizer.loss_hist
        assert np.all(np.isfinite(hist)) and hist[-1]<hist[0]
        assert unit.pool_p is not None and not np.allclose(unit.pool_p,0)
        fname=str(tmp_path/'u.pkl')
        unit.save(fname)
        x,s,ci,Y,_=ts.sine_frequency()
        back=ama.Unit.load(fname,ama.Stim(x,s,ci,Y))
        assert np.allclose(back.pool_p,unit.pool_p)
        assert np.isclose(float(back.loss),float(unit.loss))

    def test_readout_checks(self):
        unit=unit_for(nrn=ama.Nrn(readoutType='resultant',bNoise_1=True),nIterMax=1)
        with pytest.raises(Exception,match='readout'):
            unit.train_new(2,fourierType=2)


#- early stopping

class TestEarlyStopping:
    def test_keeps_best_validation_filters_and_stops(self):
        x,s,ci,Y,_=ts.sine_frequency(nStimPerCtg=40)
        train,val=ama.Stim(x,s,ci,Y).train_test(0.3,seed=1)
        unit=ama.Unit(train,ama.Nrn(),ama.Model('gss','mean'),ama.Objective('map'),
                      ama.Optimizer(nIterMax=2000,lRate0=0.2,nStepsPerChunk=20,patience=3,bVerbose=False))
        unit.train_new(4,fourierType=2,stimVal=val)
        opt=unit.optimizer
        assert len(opt.val_hist)>=1 and opt.best_step is not None
        assert len(opt.loss_hist) <= 2000
        assert np.isclose(unit.evaluate(val),min(opt.val_hist),rtol=1e-4,atol=1e-5)
        if len(opt.val_hist)*20 < 2000:                                       # stopped early
            assert len(opt.val_hist)-1-int(np.argmin(opt.val_hist)) >= 3

    def test_no_validation_leaves_training_unchanged(self):
        a=unit_for(nIterMax=60)
        a.train_new(2,fourierType=2)
        b=unit_for(nIterMax=60,patience=2)
        b.train_new(2,fourierType=2)
        assert np.allclose(np.asarray(a.out),np.asarray(b.out)) and b.optimizer.val_hist==[]


#- parametric filter banks

class TestParametric:
    def trained(self,family='morse',n=3,nIterMax=150,**kw):
        unit=unit_for(nIterMax=nIterMax)
        unit.train_parametric(n,family=family,init={'kTop':0.2,'ratio':1.5},**kw)
        return unit

    @pytest.mark.parametrize('family',['morse','loggabor'])
    def test_filters_have_the_parametric_form(self,family):
        unit=unit_for(nIterMax=1)
        unit.train_parametric(3,family=family,init={'kTop':0.25,'ratio':2.0,'gamma':3.0,'b':2.0,'sigma_u':0.4})
        pv=unit.parametric_values()
        F=np.asarray(unit.out)                                            # [ 32 x 3 ] centered spectrum
        assert np.allclose(np.linalg.norm(F,axis=0),1,atol=1e-5)
        pix=unit.filter.index.pix
        outside=np.setdiff1d(np.arange(F.shape[0]),pix)
        assert np.allclose(F[outside],0)                                  # only the learned half-space
        k=np.fft.fftshift(np.fft.fftfreq(32))
        for j in range(3):
            if family=='morse':
                x=np.where(k>0,k/pv['k_peak'][j],1)
                ref=np.where(k>0,np.exp(pv['b']*np.log(x)+pv['b']/pv['gamma']*(1-x**pv['gamma'])),0)
            else:
                x=np.where(k>0,k/pv['k_peak'][j],1)
                ref=np.where(k>0,np.exp(-np.log(x)**2/(2*pv['sigma_u']**2)),0)
            ref=ref/np.linalg.norm(ref)
            assert np.allclose(np.abs(F[:,j]),ref,atol=1e-5)
        assert np.allclose(pv['k_peak'][:-1]/pv['k_peak'][1:],pv['ratio'],rtol=1e-5)   # tied dilations

    def test_learns_and_supports_evaluation_saving_and_refinement(self,tmp_path):
        unit=self.trained()
        hist=unit.optimizer.loss_hist
        assert hist[-1]<hist[0]
        x,s,ci,Y,_=ts.sine_frequency(seed=4)
        test=ama.Stim(x,s,ci,Y)
        assert np.isfinite(unit.evaluate(test))
        fname=str(tmp_path/'p.pkl')
        unit.save(fname)
        x,s,ci,Y,_=ts.sine_frequency()
        back=ama.Unit.load(fname,ama.Stim(x,s,ci,Y))
        assert np.allclose(np.asarray(back.out),np.asarray(unit.out)) and np.isclose(float(back.loss),float(unit.loss))
        before=float(unit.loss)
        unit.train_recurse()
        assert float(unit.loss) <= before + 1e-6

    def test_2d_orientations_and_untied(self):
        unit=unit_for(ts.image_orientation,nIterMax=5)
        unit.train_parametric(2,family='loggabor',bTied=False,orientations=[0,np.pi/2])
        pv=unit.parametric_values()
        assert pv['k_peak'].shape==(2,) and pv['sigma_theta'].shape==(2,)
        F=np.abs(np.asarray(unit.out))                                    # [ 8 x 8 x 2 ]
        g0,g1=np.meshgrid(np.fft.fftshift(np.fft.fftfreq(8)),np.fft.fftshift(np.fft.fftfreq(8)),indexing='ij')
        # energy of filter 0 lies along the first axis, of filter 1 along the second
        assert (F[...,0]**2*np.abs(g0)).sum() > (F[...,0]**2*np.abs(g1)).sum()
        assert (F[...,1]**2*np.abs(g1)).sum() > (F[...,1]**2*np.abs(g0)).sum()

    def test_checks(self):
        unit=unit_for(nIterMax=1)
        with pytest.raises(Exception,match='fourierType'):
            unit.train_parametric(2,fourierType=0)
        with pytest.raises(Exception,match='family'):
            unit.train_parametric(2,family='gabor')


#- non-parametric multiscale filter banks

class TestMultiscale:
    def test_filters_are_dilations_of_the_mother(self):
        unit=unit_for(nIterMax=1)
        unit.optimizer.lRate0=1e-12                                       # keep the random mother as initialized
        unit.train_multiscale(3,nKnot=31,uWidth=3.0,scaleType='fixed',kPeak=[0.25,0.125,0.0625],init='random')
        mv=unit.multiscale_values()
        F=np.asarray(unit.out)                                            # [ 32 x 3 ] centered spectrum
        assert np.allclose(np.linalg.norm(F,axis=0),1,atol=1e-5)
        k=np.fft.fftshift(np.fft.fftfreq(32))
        pix=unit.filter.index.pix
        assert np.allclose(np.delete(F,pix,axis=0),0)
        # mother evaluated by independent interpolation, per filter, up to each filter's normalization
        for j,kp in enumerate(mv['k_peak']):
            u=np.log(np.where(k>0,k,1e-12)/kp)
            ref=np.interp(u,mv['log_freq_knots'],mv['mother'].real,left=0,right=0)+1j*np.interp(u,mv['log_freq_knots'],mv['mother'].imag,left=0,right=0)
            ref=np.where(k>0,ref,0)
            ref=ref/np.linalg.norm(ref)
            assert np.allclose(F[:,j],ref,atol=1e-5)
        # filters an octave apart: filter 1 at k and k' has the same ratio as filter 0 at 2k and 2k'
        idx=lambda f: int(np.argmin(np.abs(k-f)))
        r1=F[idx(2/32),1]/F[idx(3/32),1]
        r0=F[idx(4/32),0]/F[idx(6/32),0]
        assert np.isclose(r1,r0,rtol=1e-4)

    @pytest.mark.parametrize('scaleType',['tied','free'])
    def test_learning_saving_and_refinement(self,scaleType,tmp_path):
        unit=unit_for(nIterMax=200)
        unit.train_multiscale(3,nKnot=15,scaleType=scaleType,kTop=0.2,ratio=1.5,knotSmooth=0.01)
        hist=unit.optimizer.loss_hist
        assert np.all(np.isfinite(hist)) and hist[-1]<hist[0]
        mv=unit.multiscale_values()
        assert mv['mother'].shape==(15,) and mv['k_peak'].shape==(3,)
        if scaleType=='tied':
            assert np.allclose(mv['k_peak'][:-1]/mv['k_peak'][1:],mv['ratio'],rtol=1e-5)
        fname=str(tmp_path/'m.pkl')
        unit.save(fname)
        x,s,ci,Y,_=ts.sine_frequency()
        back=ama.Unit.load(fname,ama.Stim(x,s,ci,Y))
        assert np.allclose(np.asarray(back.out),np.asarray(unit.out))
        before=float(unit.loss)
        unit.train_recurse()
        assert float(unit.loss)<=before+1e-6

    def test_2d_rotated_copies(self):
        unit=unit_for(ts.image_orientation,nIterMax=5)
        unit.train_multiscale(2,nKnot=9,nKnotTheta=7,scaleType='fixed',kPeak=[0.25,0.25],orientations=[0,np.pi/2])
        F=np.abs(np.asarray(unit.out))                                    # [ 8 x 8 x 2 ]
        g0,g1=np.meshgrid(np.fft.fftshift(np.fft.fftfreq(8)),np.fft.fftshift(np.fft.fftfreq(8)),indexing='ij')
        assert (F[...,0]**2*np.abs(g0)).sum() > (F[...,0]**2*np.abs(g1)).sum()
        assert (F[...,1]**2*np.abs(g1)).sum() > (F[...,1]**2*np.abs(g0)).sum()
        assert unit.multiscale_values()['mother'].shape==(9,7)

    @pytest.mark.parametrize('scaleType',['tied','fixed'])
    def test_scale_orientation_grid_and_interleave(self,scaleType):
        unit=unit_for(ts.image_orientation,nIterMax=40)
        unit.train_multiscale(nKnot=9,nKnotTheta=7,scaleType=scaleType,kTop=0.3,ratio=1.6,nScales=2,nOrientations=3,
                              bInterleave=True)
        hist=unit.optimizer.loss_hist
        assert np.all(np.isfinite(hist)) and hist[-1]<=hist[0]
        mv=unit.multiscale_values()
        assert np.asarray(unit.out).shape[-1]==12 and mv['mother'].shape==(9,7)
        il=mv['interleaved']
        assert il.sum()==6 and not il[:6].any()
        kp,th=np.asarray(mv['k_peak']),np.asarray(mv['orientations'])
        r=mv['ratio'] if scaleType=='tied' else 1.6
        assert np.allclose(kp[:6],kp[0]/r**np.repeat([0,1],3),rtol=1e-5)
        assert np.allclose(kp[6:],kp[:6]/np.sqrt(r),rtol=1e-5)                     # tritones: half a scale step
        assert np.allclose(th[:3],[0,np.pi/3,-np.pi/3])                            # wrapped to (-pi/2, pi/2]
        dth=(th[6:]-th[:6]+np.pi/2)%np.pi-np.pi/2
        assert np.allclose(dth,np.pi/6)                                            # half a rotation step

    def test_grid_checks(self):
        unit=unit_for(ts.image_orientation,nIterMax=1)
        with pytest.raises(Exception,match='bInterleave needs'):
            unit.train_multiscale(4,bInterleave=True)
        with pytest.raises(Exception,match='does not match'):
            unit.train_multiscale(5,nScales=2,nOrientations=2,bInterleave=True)
        with pytest.raises(Exception,match='set by the grid'):
            unit.train_multiscale(nScales=2,nOrientations=2,scaleType='fixed',kPeak=[.2,.2,.1,.1])
        with pytest.raises(Exception,match='give both'):
            unit.train_multiscale(nScales=2)
        with pytest.raises(Exception,match='2D stimuli'):
            unit_for(nIterMax=1).train_multiscale(nScales=2,nOrientations=1)

    def test_several_mothers(self,tmp_path):
        unit=unit_for(nIterMax=150)
        unit.train_multiscale(3,nKnot=15,scaleType='tied',kTop=0.2,ratio=1.5,nMothers=3,knotSmooth=0.01)
        hist=unit.optimizer.loss_hist
        assert np.all(np.isfinite(hist)) and hist[-1]<hist[0]
        mv=unit.multiscale_values()
        F=np.asarray(unit.out)                                            # [ 32 x 9 ]
        assert F.shape[-1]==9 and mv['mother'].shape==(3,15) and mv['nMothers']==3
        assert np.array_equal(mv['mother_index'],np.repeat(np.arange(3),3))
        assert np.allclose(mv['k_peak'][:3],mv['k_peak'][3:6]) and np.allclose(mv['k_peak'][:3],mv['k_peak'][6:])
        assert np.allclose(np.linalg.norm(F,axis=0),1,atol=1e-5)
        # the mothers differ, and each mother's filters are its own dilations (same independent interpolation as above)
        M=mv['mother']
        assert min(np.linalg.norm(M[a]-M[b]) for a,b in ((0,1),(0,2),(1,2)))>1e-3
        k=np.fft.fftshift(np.fft.fftfreq(32))
        for im in range(3):
            for j in range(3):
                kp=mv['k_peak'][3*im+j]
                u=np.log(np.where(k>0,k,1e-12)/kp)
                ref=np.interp(u,mv['log_freq_knots'],M[im].real,left=0,right=0)+1j*np.interp(u,mv['log_freq_knots'],M[im].imag,left=0,right=0)
                ref=np.where(k>0,ref,0)
                ref=ref/np.linalg.norm(ref)
                assert np.allclose(F[:,3*im+j],ref,atol=1e-5)
        fname=str(tmp_path/'m3.pkl')
        unit.save(fname)
        x,s,ci,Y,_=ts.sine_frequency()
        assert np.allclose(np.asarray(ama.Unit.load(fname,ama.Stim(x,s,ci,Y)).out),F)

    def test_several_mothers_grid_2d_and_checks(self):
        unit=unit_for(ts.image_orientation,nIterMax=5)
        unit.train_multiscale(nKnot=9,nKnotTheta=7,kTop=0.3,ratio=1.6,nScales=2,nOrientations=2,bInterleave=True,nMothers=2)
        mv=unit.multiscale_values()
        assert np.asarray(unit.out).shape[-1]==16 and mv['mother'].shape==(2,9,7)
        assert np.allclose(mv['orientations'][:8],mv['orientations'][8:]) and mv['interleaved'].sum()==8
        with pytest.raises(Exception,match='nMothers'):
            unit_for(nIterMax=1).train_multiscale(2,nMothers=0)

    def test_edge_taper(self):
        unit=unit_for(nIterMax=150)
        unit.train_multiscale(3,nKnot=21,scaleType='tied',kTop=0.2,ratio=1.5,nMothers=2,edgeTaper=0.2,init='random',knotSmooth=0.01)
        hist=unit.optimizer.loss_hist
        assert np.all(np.isfinite(hist)) and hist[-1]<hist[0]
        mv=unit.multiscale_values()
        M=mv['mother']
        assert mv['edgeTaper']==0.2 and np.allclose(M[:,[0,-1]],0)
        w=ama.Unit._knot_taper(21,0.2)
        assert w[0]==0 and w[-1]==0 and np.allclose(w[5:16],1) and np.all(np.diff(w[:5])>0)
        # filters are still the dilations of the reported (tapered) mothers
        F=np.asarray(unit.out)
        k=np.fft.fftshift(np.fft.fftfreq(32))
        for im in range(2):
            for j in range(3):
                u=np.log(np.where(k>0,k,1e-12)/mv['k_peak'][3*im+j])
                ref=np.interp(u,mv['log_freq_knots'],M[im].real,left=0,right=0)+1j*np.interp(u,mv['log_freq_knots'],M[im].imag,left=0,right=0)
                ref=np.where(k>0,ref,0)
                assert np.allclose(F[:,3*im+j],ref/np.linalg.norm(ref),atol=1e-5)
        with pytest.raises(Exception,match='edgeTaper'):
            unit_for(nIterMax=1).train_multiscale(2,edgeTaper=0.6)
        u2=unit_for(ts.image_orientation,nIterMax=3)
        u2.train_multiscale(nKnot=9,nKnotTheta=7,kTop=0.3,ratio=1.6,nScales=2,nOrientations=2,edgeTaper=0.25)
        assert np.allclose(u2.multiscale_values()['mother'][[0,-1],:],0)

    def test_knot_smoothness_penalty_smooths_the_mother(self):
        def rough(w):
            unit=unit_for(nIterMax=300)
            unit.train_multiscale(2,nKnot=21,scaleType='fixed',kPeak=[0.2,0.1],init='random',knotSmooth=w)
            m=unit.multiscale_values()['mother']
            return np.sum(np.abs(np.diff(m,2))**2)/np.sum(np.abs(m)**2)
        assert rough(5.0) < rough(0.0)

    def test_checks(self):
        unit=unit_for(nIterMax=1)
        with pytest.raises(Exception,match='fourierType'):
            unit.train_multiscale(2,fourierType=0)
        with pytest.raises(Exception,match='scaleType'):
            unit.train_multiscale(2,scaleType='other')
        with pytest.raises(Exception,match='kPeak'):
            unit.train_multiscale(2,scaleType='fixed',kPeak=[0.1])
