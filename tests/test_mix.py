"""
Gaussian mixture category likelihood ('mix'): reduction to AMA-Gauss, the EM fit against sklearn, the likelihood
against an independent computation, gradients, and learning on categories with bimodal responses.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.stats import multivariate_normal as smvn
from sklearn.mixture import GaussianMixture

import ama
import stimuli as ts
from test_likelihood_extensions import model_inputs
from test_leave_one_out import loo_noise_prior


def bimodal_ctg(nPix=16,nCtg=4,nStimPerCtg=60,noise=0.3,seed=0):
    """category k: stimuli +-a_k along one direction (random sign), so every category has mean zero; plus a second,
    uninformative direction and noise"""
    rng=np.random.default_rng(seed)
    A=rng.standard_normal((nPix,2))
    d=np.linalg.qr(A-A.mean(0))[0]
    amp=np.linspace(0.5,2.,nCtg)
    ci=np.repeat(np.arange(nCtg),nStimPerCtg)
    sign=rng.choice([-1.,1.],len(ci))
    s=np.outer(d[:,0],sign*amp[ci]) + np.outer(d[:,1],rng.standard_normal(len(ci))) + noise*rng.standard_normal((nPix,len(ci)))
    s=s-s.mean(0)
    return np.arange(nPix),s,ci,np.arange(nCtg,dtype=float),d


class TestMixture:
    def test_one_component_is_ama_gauss(self):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=3)
        w=unit.stim.weights
        gss=np.asarray(ama.Model._model__gss(R,Rm,RVar,noiseCov,None,w,False,unit.model,unit.stim.Y))
        mix=np.asarray(ama.Model._model__mix(R,Rm,RVar,noiseCov,None,w,False,ama.Model('mix',nMix=1,mixReg=0.),unit.stim.Y))
        assert np.allclose(mix,gss,rtol=1e-10,atol=1e-10)

    def test_shrinkage_with_one_component_is_ama_gauss(self):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=3)
        w=unit.stim.weights
        gss=np.asarray(ama.Model._model__gss(R,Rm,RVar,noiseCov,None,w,False,ama.Model('gss',covShrink=0.3),unit.stim.Y))
        mix=np.asarray(ama.Model._model__mix(R,Rm,RVar,noiseCov,None,w,False,ama.Model('mix',nMix=1,mixReg=0.,covShrink=0.3),
                                             unit.stim.Y))
        assert np.allclose(mix,gss,rtol=1e-10,atol=1e-10)

    def test_em_matches_sklearn(self):
        rng=np.random.default_rng(0)
        # two well separated clusters per category, [ nF x nStim_Ctg x nCtg ]
        nS,nC=200,2
        X=np.zeros((2,nS,nC))
        for c in range(nC):
            lab=rng.random(nS)<0.3+0.3*c
            X[:,:,c]=np.where(lab,[[3.],[3.*c]],[[-3.],[0.]])+rng.standard_normal((2,nS))*np.array([[1.],[0.5]])
        w=jnp.ones((nS,nC))
        pi,mu,cov=[np.asarray(a) for a in ama.Model._mix_fit(jnp.asarray(X),w,ama.Model('mix',nMix=2,nEM=100,mixReg=0.))]
        for c in range(nC):
            gm=GaussianMixture(2,covariance_type='full',tol=1e-10,max_iter=1000,random_state=0).fit(X[:,:,c].T)
            o=np.argsort(mu[c,:,0]); g=np.argsort(gm.means_[:,0])
            n=nS*pi[c]
            assert np.allclose(pi[c,o],gm.weights_[g],atol=1e-3)
            assert np.allclose(mu[c,o],gm.means_[g],atol=1e-3)
            # ours is Bessel corrected: N_c/(N_c-1) times the ML covariance
            assert np.allclose(cov[c,o]*((n[o]-1)/n[o])[:,None,None],gm.covariances_[g],atol=1e-3)

    def test_likelihood_matches_reference(self):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=2)
        m=ama.Model('mix',nMix=3)
        w=unit.stim.weights
        lAll=np.asarray(ama.Model._model__mix(R,Rm,RVar,noiseCov,None,w,False,m,unit.stim.Y))
        pi,mu,cov=[np.asarray(a) for a in ama.Model._mix_fit(Rm,w,m)]
        N,Rn=np.asarray(noiseCov),np.asarray(R)
        for (l,k,i) in [(0,0,0),(5,1,3),(9,4,2)]:
            ref=np.log(sum(pi[i,c]*smvn(mu[i,c],cov[i,c]+N[i]).pdf(Rn[:,l,k]) for c in range(3)))
            assert np.isclose(lAll[l,k,i],ref)

    def test_gradient_matches_finite_difference(self):
        x,s,ci,Y,_=bimodal_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y,bContrastNormalize=True),ama.Nrn(),ama.Model('mix','mean',nMix=2),ama.Objective('map'),
                      ama.Optimizer(nIterMax=1,bVerbose=False))
        unit._finalize(2,np.arange(2),dtype=jnp.float64)
        lf=lambda f: unit._loss_fun_lrn({'f':f},unit.rng,unit.filter.prepped_jx,unit.filter._insert_index_jx,
                                        unit.stim.val,unit.stim.weights,unit.stim.yCtg,unit.stim.Y)
        f=jnp.asarray(np.random.default_rng(2).standard_normal(unit.filter._shape))
        g=jax.grad(lf)(f)
        assert np.all(np.isfinite(np.asarray(g)))
        for idx in [(0,0),(9,1)]:
            e=np.zeros(f.shape); e[idx]=1e-6
            assert np.isclose(g[idx],(lf(f+e)-lf(f-e))/2e-6,rtol=1e-4,atol=1e-7)

    def test_mixture_decodes_bimodal_categories_better(self):
        """every category has mean zero along the informative direction, with responses at +-a_k: a mixture resolves the
        two modes, a single gaussian only sees the variance"""
        x,s,ci,Y,d=bimodal_ctg()
        train,test=ama.Stim(x,s,ci,Y,bContrastNormalize=True).train_test(0.3)
        held={}
        for name,model in [('gss',ama.Model('gss','mean')),('mix',ama.Model('mix','mean',nMix=2))]:
            unit=ama.Unit(train,ama.Nrn(),model,ama.Objective('map'),ama.Optimizer(nIterMax=300,lRate0=0.02,bVerbose=False))
            unit._finalize(1,[0],dtype=jnp.float64)
            unit.filter.out=jnp.asarray(d[:,:1])
            held[name]=unit.evaluate(test)
            if name=='mix':
                # from a perturbed start, training returns to the informative direction (random starts can stall in a
                # poor basin of this nonconvex cost; see nRestarts)
                f0=d[:,0]+0.4*np.random.default_rng(1).standard_normal(len(x))
                unit.filter.out=jnp.asarray((f0/np.linalg.norm(f0))[:,None])
                start=abs(float(np.asarray(unit.out).ravel()@d[:,0]))
                unit.train_recurse()
                f=np.asarray(unit.out).ravel()
                assert abs(f@d[:,0])>max(0.9,start)
        assert held['mix']<held['gss']

    def test_validation(self):
        x,s,ci,Y,_=bimodal_ctg(nStimPerCtg=5)
        for model,match in [(ama.Model('mix','mean',nMix=3),'2\\*nMix'),
                            (ama.Model('mix','mean',ctgPoolWidth=1.,bPoolMeans=True),'bPoolMeans')]:
            unit=ama.Unit(ama.Stim(x,s,ci,Y,bContrastNormalize=True),ama.Nrn(),model,ama.Objective('map'),
                          ama.Optimizer(nIterMax=1,bVerbose=False))
            with pytest.raises(Exception,match=match):
                unit._finalize(1,[0])


class TestMixtureLeaveOneOut:
    def test_matches_downdating_with_fixed_responsibilities(self):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=2)
        m=ama.Model('mix',nMix=2,mixReg=1e-3)
        w=unit.stim.weights
        loo=np.asarray(ama.Model._model__mix(R,Rm,RVar,noiseCov,None,w,True,m,unit.stim.Y))
        plain=np.asarray(ama.Model._model__mix(R,Rm,RVar,noiseCov,None,w,False,m,unit.stim.Y))
        (pi,mu,cov),det=ama.Model._mix_fit(Rm,w,m,bDetail=True)
        r,X=np.asarray(det['r']),np.transpose(np.asarray(Rm),(2,1,0))  # [ nCtg x nStim x nMix ], [ nCtg x nStim x nF ]
        N,Rn,W=np.asarray(noiseCov),np.asarray(R),np.asarray(w)
        ridge=np.asarray(det['ridge'])
        for (l,k) in [(0,0),(7,2),(19,4)]:
            terms=[]
            rr=r[k].copy(); rr[l]=0                                  # the stimulus's share removed
            n=W[:,k].sum()-W[l,k]
            NL,lp=loo_noise_prior(RVar,w,l,k)
            for c in range(2):
                Nc=rr[:,c].sum()
                muc=(rr[:,c,None]*X[k]).sum(0)/Nc
                D=X[k]-muc
                Sc=(rr[:,c,None,None]*D[:,:,None]*D[:,None,:]).sum(0)/max(Nc-1,Nc/2)+ridge[k]*np.eye(2)
                terms.append(np.log(Nc/n)+smvn(muc,Sc+NL).logpdf(Rn[:,l,k]))
            assert np.isclose(loo[l,k,k],np.logaddexp(*terms)+lp)
        off=~np.eye(5,dtype=bool)
        assert np.allclose(loo[:,off],plain[:,off])

    def test_training(self):
        x,s,ci,Y,_=bimodal_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y,bContrastNormalize=True),ama.Nrn(),ama.Model('mix','mean',nMix=2,bLeaveOneOut=True),
                      ama.Objective('map'),ama.Optimizer(nIterMax=60,lRate0=0.05,bVerbose=False))
        unit.train_new(1)
        h=np.asarray(unit.optimizer.loss_hist)
        assert np.all(np.isfinite(h)) and h[-1]<h[0]


class TestWarmStart:
    def test_warm_start_at_a_converged_fit_stays_there(self):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=2)
        w=unit.stim.weights
        cold=ama.Model('mix',nMix=2,nEM=300)
        lc,fit=ama.Model._mix_likelihoods(R,Rm,noiseCov,w,False,cold)
        lw,_=ama.Model._mix_likelihoods(R,Rm,noiseCov,w,False,ama.Model('mix',nMix=2,nEMWarm=2),state=fit)
        assert np.allclose(np.asarray(lw),np.asarray(lc),atol=1e-6)

    def test_emptied_component_starts_again(self):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=2)
        w=unit.stim.weights
        m=ama.Model('mix',nMix=2,nEM=30,nEMWarm=2)
        lc,fit=ama.Model._mix_likelihoods(R,Rm,noiseCov,w,False,m)
        pi,mu,cov=[np.array(a) for a in fit]
        pi[1]=[1.,0.]                                                       # category 1: its second component emptied
        lw,new=ama.Model._mix_likelihoods(R,Rm,noiseCov,w,False,m,state=tuple(jnp.asarray(a) for a in (pi,mu,cov)))
        assert np.allclose(np.asarray(lw),np.asarray(lc),rtol=1e-10)       # the cold fit (nEM iterations) again
        assert np.all(np.asarray(new[0])>1e-8)

    def warm_unit(self,**opt_kw):
        x,s,ci,Y,_=bimodal_ctg()
        return ama.Unit(ama.Stim(x,s,ci,Y,bContrastNormalize=True),ama.Nrn(),
                        ama.Model('mix','mean',nMix=2,bWarmEM=True,nEMWarm=2),ama.Objective('map'),
                        ama.Optimizer(**{'nIterMax':100,'lRate0':0.05,'bVerbose':False,**opt_kw}))

    def test_gradient_with_a_fixed_state(self):
        unit=self.warm_unit(nIterMax=1)
        unit._finalize(1,[0],dtype=jnp.float64)
        f=jnp.asarray(np.random.default_rng(2).standard_normal(unit.filter._shape))
        state=unit._mix_state0(f)
        lf=lambda f: unit._loss_fun_lrn({'f':f},unit.rng,unit.filter.prepped_jx,unit.filter._insert_index_jx,
                                        unit.stim.val,unit.stim.weights,unit.stim.yCtg,unit.stim.Y,state)[0]
        g=jax.grad(lf)(f)
        for idx in [(0,0),(9,0)]:
            e=np.zeros(f.shape); e[idx]=1e-6
            assert np.isclose(g[idx],(lf(f+e)-lf(f-e))/2e-6,rtol=1e-4,atol=1e-7)

    @pytest.mark.parametrize('opt_kw',[{},{'batchSize':80}])
    def test_training(self,opt_kw):
        unit=self.warm_unit(**opt_kw)
        unit.train_new(1)
        h=np.asarray(unit.optimizer.loss_hist)
        assert np.all(np.isfinite(h)) and np.mean(h[-10:])<np.mean(h[:10])
        assert np.isfinite(float(unit.loss))


POOL=[dict(ctgPoolWidth=0.7),dict(covShrink=0.4,covTarget='pooled'),dict(ctgPoolWidth=0.5,covShrink=0.3,covTarget='pooled'),
      dict(ctgPoolWidth=0.5,covShrink=0.3)]


class TestMixturePooling:
    @pytest.mark.parametrize('kw',POOL)
    @pytest.mark.parametrize('loo',[False,True])
    def test_one_component_is_pooled_ama_gauss(self,kw,loo):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=3,**kw)
        R=R+0.2*jnp.asarray(np.random.default_rng(0).standard_normal(R.shape))
        w=unit.stim.weights
        gss=np.asarray(ama.Model._model__gss(R,Rm,RVar,noiseCov,None,w,loo,ama.Model('gss',**kw),unit.stim.Y))
        mix=np.asarray(ama.Model._model__mix(R,Rm,RVar,noiseCov,None,w,loo,ama.Model('mix',nMix=1,mixReg=0.,**kw),unit.stim.Y))
        valid=np.asarray(w)>0
        assert np.allclose(mix[valid],gss[valid],rtol=1e-9,atol=1e-9)

    def test_components_borrow_in_proportion_to_their_weight(self):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=2)
        m=ama.Model('mix',nMix=2,mixReg=1e-3,ctgPoolWidth=0.8)
        w=unit.stim.weights
        (pi,mu,cov),det=ama.Model._mix_fit(Rm,w,m,bDetail=True,Y=unit.stim.Y)
        pi,cov=np.asarray(pi),np.asarray(cov)
        X,W,Y=np.transpose(np.asarray(Rm),(2,1,0)),np.asarray(w)>0,np.asarray(unit.stim.Y)
        r,Nc,S=np.asarray(det['r']),np.asarray(det['Nc']),np.asarray(det['S'])
        K=np.exp(-(Y[:,None]-Y[None])**2/(2*0.8**2))
        for i in range(len(Y)):
            P=sum(K[i,j]*np.cov(X[j,W[:,j]].T)*(W[:,j].sum()-1) for j in range(len(Y)) if j!=i)
            Q=sum(K[i,j]*(W[:,j].sum()-1) for j in range(len(Y)) if j!=i)
            for c in range(2):
                ref=(S[i,c]+pi[i,c]*P)/(max(Nc[i,c]-1,Nc[i,c]/2)+pi[i,c]*Q)+np.asarray(det['ridge'])[i]*np.eye(2)
                assert np.allclose(cov[i,c],ref,rtol=1e-10)

    @pytest.mark.parametrize('kw',[dict(ctgPoolWidth=0.7),dict(ctgPoolWidth=0.6,covShrink=0.3,covTarget='pooled')])
    def test_leave_one_out_matches_refitting_with_fixed_responsibilities(self,kw):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=2)
        m=ama.Model('mix',nMix=2,mixReg=1e-3,**kw)
        w=unit.stim.weights
        loo=np.asarray(ama.Model._model__mix(R,Rm,RVar,noiseCov,None,w,True,m,unit.stim.Y))
        _,det=ama.Model._mix_fit(Rm,w,m,bDetail=True,Y=unit.stim.Y)
        r,X=np.asarray(det['r']),np.transpose(np.asarray(Rm),(2,1,0))
        Rn,W,Y,Nn=np.asarray(R),np.asarray(w)>0,np.asarray(unit.stim.Y),np.asarray(noiseCov)
        nC=len(Y)
        width=kw['ctgPoolWidth']
        K=np.exp(-(Y[:,None]-Y[None])**2/(2*width**2))
        s=kw.get('covShrink',0.)
        for (l,k) in [(0,0),(7,2),(19,4)]:
            keep=W.copy(); keep[l,k]=False
            scat=[np.cov(X[j,keep[:,j]].T)*(keep[:,j].sum()-1) for j in range(nC)]
            dofs=[keep[:,j].sum()-1 for j in range(nC)]
            T=sum(scat)/sum(dofs)
            ridge=np.asarray(det['ridge'])
            for i in range(nC):
                rr=r[i].copy()
                if i==k:
                    rr[l]=0
                n=keep[:,i].sum()
                P=sum(K[i,j]*scat[j] for j in range(nC) if j!=i)
                Q=sum(K[i,j]*dofs[j] for j in range(nC) if j!=i)
                NL,lp=loo_noise_prior(RVar,w,l,k)
                N=NL if i==k else Nn[i]
                terms=[]
                for c in range(2):
                    Nc=rr[:,c].sum()
                    muc=(rr[:,c,None]*X[i]).sum(0)/Nc
                    D=X[i]-muc
                    Sc=(rr[:,c,None,None]*D[:,:,None]*D[:,None,:]).sum(0)
                    pic=Nc/n
                    C=(Sc+pic*P)/(max(Nc-1,Nc/2)+pic*Q)
                    C=(1-s)*C+s*T if s else C
                    terms.append(np.log(pic)+smvn(muc,C+ridge[i]*np.eye(2)+N).logpdf(Rn[:,l,k]))
                ref=np.logaddexp(*terms)+(lp if i==k else 0.)
                assert np.isclose(loo[l,k,i],ref,rtol=1e-9)

    def test_basic_responses_leave_out_the_mean_response(self):
        """leave-one-out removes the stimulus's mean response (what the fit is made of), not its noisy observed response"""
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',n=2)
        Rb=R+0.5*jnp.asarray(np.random.default_rng(1).standard_normal(R.shape))
        m=ama.Model('mix',nMix=1,mixReg=0.)
        w=unit.stim.weights
        mix=np.asarray(ama.Model._model__mix(Rb,Rm,RVar,noiseCov,None,w,True,m,unit.stim.Y))
        gss=np.asarray(ama.Model._model__gss(Rb,Rm,RVar,noiseCov,None,w,True,ama.Model('gss'),unit.stim.Y))
        valid=np.asarray(w)>0
        assert np.allclose(mix[valid],gss[valid],rtol=1e-9,atol=1e-9)

    def test_training_with_pooling(self):
        x,s,ci,Y,_=bimodal_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y,bContrastNormalize=True),ama.Nrn(),
                      ama.Model('mix','mean',nMix=2,ctgPoolWidth=0.7,bLeaveOneOut=True),ama.Objective('map'),
                      ama.Optimizer(nIterMax=60,lRate0=0.05,bVerbose=False))
        unit.train_new(1)
        h=np.asarray(unit.optimizer.loss_hist)
        assert np.all(np.isfinite(h)) and h[-1]<h[0]


class TestMixturePooledLowRank:
    """pooled 'mix' leave-one-out by rank-one updates for the other categories (Model._mix_loo_pooled_low_rank)"""

    @pytest.mark.parametrize('bLooNoise',[True,False])
    @pytest.mark.parametrize('kw',POOL[:3])
    @pytest.mark.parametrize('gen',[ts.gaussian_ctg,ts.unequal_counts])
    def test_matches_the_general_path(self,kw,bLooNoise,gen,monkeypatch):
        unit,R,Rm,RVar,noiseCov=model_inputs('gss',gen=gen,n=2)
        R=R+0.2*jnp.asarray(np.random.default_rng(0).standard_normal(R.shape))
        m=ama.Model('mix',nMix=2,mixReg=1e-3,bLooNoise=bLooNoise,**kw)
        m._Yperiod=None
        assert ama.Model._mix_loo_low_rank_ok(m)
        w=unit.stim.weights
        fast=np.asarray(ama.Model._model__mix(R,Rm,RVar,noiseCov,None,w,True,m,unit.stim.Y))
        monkeypatch.setattr(ama.Model,'_mix_loo_low_rank_ok',staticmethod(lambda m: False))
        slow=np.asarray(ama.Model._model__mix(R,Rm,RVar,noiseCov,None,w,True,m,unit.stim.Y))
        valid=np.asarray(w)>0
        assert np.all(np.isfinite(fast[valid]))
        assert np.allclose(fast[valid],slow[valid],rtol=1e-9,atol=1e-9)

    def test_general_path_otherwise(self):
        assert not ama.Model._mix_loo_low_rank_ok(ama.Model('mix',ctgPoolWidth=0.5,covRank=1))
        assert not ama.Model._mix_loo_low_rank_ok(ama.Model('mix',ctgPoolWidth=0.5,covShrink=0.2))
