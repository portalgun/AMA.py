"""
Response model checks: precision of fourier-domain learning under jax x64, the stage-1 noise variance through linear
normalizations (also where an activation zeroes the response), validation of settings the likelihood can not model,
and rho kept consistent with the noise correlation type.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import ama
import stimuli as ts
from test_extended import unit_from


class TestPrecision:
    @pytest.mark.parametrize('fourierType',[1,2])
    def test_complex64_learning_stays_single_precision(self,fourierType):
        x,s,ci,Y,_=ts.sine_frequency()
        unit=unit_from(ama.Stim(x,s,ci,Y),n=2,fourierType=fourierType,dtype=jnp.complex64)
        out=unit.nrn.main(unit.rng,unit.stim.val,unit.filter.out_flat,unit.stim.weights)
        assert all(np.dtype(o.dtype) in (np.float32,np.complex64) for o in out)
        assert unit.loss.dtype==jnp.float32

    def test_numpy_scalar_settings_do_not_promote(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        nrn=ama.Nrn(fano=np.float64(0.5),var0=np.float64(0.23),rmax=np.float64(5.7))
        unit=unit_from(ama.Stim(x,s,ci,Y),nrn=nrn,n=2,dtype=jnp.float32)
        assert unit.loss.dtype==jnp.float32


class TestStage1Variance:
    keys=jax.random.split(jax.random.PRNGKey(1),3000)

    def mc_ratio(self,unit):
        """model variance / Monte Carlo variance of the final responses: at responses zeroed by relu, and elsewhere"""
        f=unit.filter.out_flat
        out=unit.nrn.main(unit.rng,unit.stim.val,f,unit.stim.weights)
        RVar,r=np.asarray(out[4]).real,np.asarray(out[0]).real
        RN=np.asarray(jax.vmap(lambda k: unit.nrn.main(k,unit.stim.val,f,unit.stim.weights)[3])(self.keys)).real
        w=np.broadcast_to(np.asarray(unit.stim.weights)>0,r.shape)
        ratio=RVar/np.var(RN,0)
        return np.median(ratio[(r==0)&w]),np.median(ratio[(r!=0)&w])

    @pytest.mark.parametrize('fourierType',[1,2])
    def test_narrow_with_relu(self,fourierType):
        x,s,ci,Y,_=ts.sine_frequency()
        unit=unit_from(ama.Stim(x,s,ci,Y),nrn=ama.Nrn(bNoise_1=True,normalizeType='narrow',activationType='relu'),n=2,
                       fourierType=fourierType,dtype=jnp.complex128)
        zero,other=self.mc_ratio(unit)
        assert 0.9<zero<1.1 and 0.9<other<1.1

    def test_broad_with_relu_on_unnormalized_stimuli(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        s=s*np.linspace(0.3,3,s.shape[1])                          # broadband gain differs per stimulus
        with pytest.warns(UserWarning):
            stim=ama.Stim(x,s,ci,Y)
        unit=unit_from(stim,nrn=ama.Nrn(bNoise_1=True,normalizeType='broad',activationType='relu'),n=2)
        zero,other=self.mc_ratio(unit)
        assert 0.9<zero<1.1 and 0.9<other<1.1


class TestValidation:
    def test_phase_normalization_with_stage1_noise(self):
        x,s,ci,Y,_=ts.sine_frequency()
        with pytest.raises(Exception,match="normalizeType='phase'"):
            unit_from(ama.Stim(x,s,ci,Y),nrn=ama.Nrn(bNoise_1=True,normalizeType='phase'),n=2,fourierType=2,dtype=jnp.complex128)

    def test_var0_must_be_positive(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        with pytest.raises(Exception,match='var0 must be positive'):
            unit_from(ama.Stim(x,s,ci,Y),nrn=ama.Nrn(var0=0.),n=2)
        unit_from(ama.Stim(x,s,ci,Y),nrn=ama.Nrn(var0=0.,rho=None),n=2)  # no noise model: allowed

    def test_rho_set_later_updates_the_correlation_type(self):
        nrn=ama.Nrn(rho=0)
        assert nrn.corrType=='uncorr'
        nrn.rho=0.3
        assert nrn.corrType=='corr'
        nrn.rho=None
        assert nrn.corrType=='None'
        assert nrn.copy().rho is None and ama.Nrn(rho=0.3).copy().corrType=='corr'
