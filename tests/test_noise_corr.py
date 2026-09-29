"""
Specified noise correlations (Nrn rho as a correlation matrix over the flattened response dimensions): sampling, the
AMA-Gauss and full AMA likelihoods, equivalence with a scalar rho, validation, and the configuration round trip.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as smvn

import ama
import stimuli as ts
from test_extended import unit_from


def corr3():
    return np.array([[1.,0.5,-0.2],[0.5,1.,0.1],[-0.2,0.1,1.]])


class TestSampling:
    def test_real_samples_have_the_matrix_correlation(self):
        R=jnp.zeros((3,1,1))
        z=np.asarray(ama.Nrn._noise__true(R,1.,1.,40000,jax.random.key(0),corr3()))[:,0,0,:]
        assert np.allclose(np.corrcoef(z),corr3(),atol=0.02)

    def test_complex_dimensions_follow_the_flattening_order(self):
        # 2 complex filters -> dimensions (re0, re1, im0, im1): correlate re0 with im0 only
        P=np.eye(4); P[0,2]=P[2,0]=0.6
        R=jnp.zeros((2,1,1),dtype=jnp.complex128)
        z=np.asarray(ama.Nrn._noise__true(R,1.,1.,40000,jax.random.key(1),P))[:,0,0,:]
        flat=np.concatenate((z.real,z.imag),0)
        assert np.allclose(np.corrcoef(flat),P,atol=0.02)


class TestLikelihood:
    def test_uniform_matrix_equals_scalar_rho(self):
        x,s,ci,Y,_=ts.unequal_counts()
        for modelType,response in [('gss','mean'),('full','mean'),('gss','basic')]:
            losses=[]
            for rho in (0.3,0.3+0.7*np.eye(2)):
                nrn=ama.Nrn(rho=rho,bNoise_2=response=='basic')
                losses.append(float(unit_from(ama.Stim(x,s,ci,Y),nrn=nrn,modelType=modelType,responseType=response,n=2).loss))
            assert np.isclose(losses[0],losses[1],rtol=1e-12)

    def test_gss_noise_covariance(self):
        RVar=jnp.asarray(np.random.default_rng(0).uniform(0.5,2,(3,4,2)))
        w=jnp.ones((4,2))
        cov=np.asarray(ama.Nrn._corr__corr(RVar,w,corr3()))
        sd=np.sqrt(np.asarray(RVar).mean(1))                       # [ 3 x nCtg ]
        for c in range(2):
            assert np.allclose(cov[c],np.outer(sd[:,c],sd[:,c])*corr3())

    def test_full_ama_matches_brute_force(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        unit=unit_from(ama.Stim(x,s,ci,Y),nrn=ama.Nrn(rho=corr3()),modelType='full',n=3)
        out=unit.nrn.main(unit.rng,unit.stim.val,unit.filter.out_flat,unit.stim.weights)
        R,Rm,RVar=[np.asarray(ama._flatten_responses(v)) for v in unit.model._response_fun(*out)]
        lAll=np.asarray(unit.likelihoods)
        for (l,k,i) in [(0,0,0),(5,1,3),(9,4,2)]:
            terms=[smvn(Rm[:,j,i],np.outer(np.sqrt(RVar[:,j,i]),np.sqrt(RVar[:,j,i]))*corr3()).logpdf(R[:,l,k])
                   for j in range(R.shape[1])]
            assert np.isclose(lAll[l,k,i],logsumexp(terms)-np.log(R.shape[1]))


class TestSettings:
    @pytest.mark.parametrize('rho,match',[(np.eye(2),'must be 3 x 3'),(np.array([[1,.9,.9],[.9,1,-.9],[.9,-.9,1.]]),'positive definite'),
                                          (2*np.eye(3),'unit diagonal')])
    def test_validation(self,rho,match):
        x,s,ci,Y,_=ts.gaussian_ctg()
        with pytest.raises(Exception,match=match):
            unit_from(ama.Stim(x,s,ci,Y),nrn=ama.Nrn(rho=rho),n=3)

    def test_config_round_trip(self):
        x,s,ci,Y,_=ts.gaussian_ctg()
        stim=ama.Stim(x,s,ci,Y)
        unit=unit_from(stim,nrn=ama.Nrn(rho=corr3()),n=3)
        back=ama.Unit.from_config(unit.config(),stim)
        assert np.allclose(back.nrn.rho,corr3()) and back.nrn.corrType=='corr'
        assert ama._freeze(back.nrn.rho)==ama._freeze(unit.nrn.rho)
