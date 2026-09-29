"""
Latent geometry: several latent dimensions (Y [ nCtg x nDim ]) and circular latent variables (Stim Yperiod). Estimators,
errors, targets, divergences and category pooling are checked against independent computations, and end to end.
"""
import numpy as np
import pytest
import jax.numpy as jnp
from scipy.optimize import linprog

import ama
import stimuli as ts


def grid_ctg(levels=(-1.,0.,1.),nPix=16,nStimPerCtg=30,signal=0.8,seed=0):
    """two latent dimensions on a 3 x 3 grid, each scaling its own direction of the stimulus"""
    rng=np.random.default_rng(seed)
    A=rng.standard_normal((nPix,2))
    d=np.linalg.qr(A-A.mean(0))[0]                                  # zero mean: contrast normalization keeps them
    Y=np.array([(a,b) for a in levels for b in levels])
    ci=np.repeat(np.arange(len(Y)),nStimPerCtg)
    s=signal*(d@Y[ci].T) + rng.standard_normal((nPix,len(ci)))
    s=ts.contrast_normalize(s)
    x=np.arange(nPix)
    return x,s,ci,Y,d


def fixed_unit(stim,modelType='gss',errType='map',n=2,seed=1,**obj_kw):
    unit=ama.Unit(stim,ama.Nrn(),ama.Model(modelType,'mean'),ama.Objective(errType,**obj_kw),ama.Optimizer(nIterMax=1,bVerbose=False))
    unit._finalize(n,np.arange(n),dtype=jnp.float64)
    f=np.random.default_rng(seed).standard_normal(unit.filter._shape)
    unit.filter.out=jnp.asarray(f/np.linalg.norm(f,axis=0))
    return unit


def orientation_stim(**kw):
    x,s,ci,Y,_=ts.image_orientation()
    return ama.Stim(x,s,ci,Y,Yperiod=180.,**kw)


#- helpers and Stim

class TestGeometry:
    def test_wrap(self):
        d=jnp.array([170.,-170.,90.,-90.,10.])
        assert np.allclose(ama._wrap(d,(180.,)),[-10.,10.,-90.,-90.,10.])
        D=jnp.array([[350.,5.],[-200.,-7.]])
        assert np.allclose(ama._wrap(D,(360.,None)),[[-10.,5.],[160.,-7.]])
        assert ama._wrap(d,None) is d

    def test_stim_latent_shapes(self):
        x,s,ci,Y,_=grid_ctg()
        st=ama.Stim(x,s,ci,Y)
        assert st.nDim==2 and st.Y.shape==(9,2) and st.yCtg.shape==(30,9,2)
        assert np.allclose(st.yCtg[3,4],Y[4])
        x,s,ci,Y,_=ts.gaussian_ctg()
        for Yin in (Y[None,:],Y[:,None]):                           # matlab row or column vectors
            st=ama.Stim(x,s,ci,Yin)
            assert st.nDim==1 and st.Y.shape==(5,)

    def test_stim_validation(self):
        x,s,ci,Y,_=grid_ctg()
        with pytest.raises(Exception,match='distinct'):
            ama.Stim(x,s,ci,np.zeros((9,2)))
        with pytest.raises(Exception,match='one value per'):
            ama.Stim(x,s,ci,Y,Yperiod=(360.,None,1.))
        with pytest.raises(Exception,match='positive'):
            ama.Stim(x,s,ci,Y,Yperiod=-1.)
        assert ama.Stim(x,s,ci,Y,Yperiod=(None,None)).Yperiod is None
        assert ama.Stim(x,s,ci,Y,Yperiod=360).Yperiod==(360.,360.)

    def test_subsets_keep_latent_dimensions(self):
        x,s,ci,Y,_=grid_ctg()
        train,test=ama.Stim(x,s,ci,Y).train_test(0.2)
        assert train.yCtg.shape[1:]==(9,2) and test.yCtg.shape[1:]==(9,2)
        w=np.asarray(test.weights)>0
        assert np.allclose(np.asarray(test.yCtg)[w],np.asarray(test.Y)[np.nonzero(w)[1]])


#- estimators and errors

class TestEstimatorsAndErrors:
    def test_circular_mean(self):
        Y=jnp.array([0.,45.,90.,135.])
        p=np.array([0.5,0.,0.,0.5])                                 # 0 and 135 = -45: circular mean -22.5
        est=ama.Objective._est__mean(jnp.log(jnp.asarray(p+1e-300))[None],Y,(180.,))
        assert np.isclose(float(est[0]),-22.5)

    def test_multidimensional_estimators(self):
        rng=np.random.default_rng(0)
        Y=jnp.asarray(rng.standard_normal((6,2)))
        lp=jnp.asarray(rng.standard_normal((4,6)))
        p=np.exp(lp)/np.exp(lp).sum(-1,keepdims=True)
        assert np.allclose(ama.Objective._est__mean(lp,Y),p@np.asarray(Y))
        assert np.allclose(ama.Objective._est__mode(lp,Y),np.asarray(Y)[np.argmax(p,-1)])
        med=np.asarray(ama.Objective._est__median(lp,Y))
        for d in range(2):
            o=np.argsort(np.asarray(Y)[:,d])
            for r in range(4):
                assert np.isclose(med[r,d],np.interp(0.5,np.cumsum(p[r,o]),np.asarray(Y)[o,d]))

    def test_errors_sum_over_dimensions_and_wrap(self):
        yHat=jnp.array([[[170.,1.],[0.,3.]]])
        yCtg=jnp.array([[[-170.,0.],[90.,1.]]])
        Y=jnp.zeros((2,2))
        per=(360.,None)
        assert np.allclose(ama.Objective._err__l2(yHat,yCtg,None,Y,per),[[20.**2+1.,90.**2+4.]])
        assert np.allclose(ama.Objective._err__l1(yHat,yCtg,None,Y,per),[[21.,92.]])
        assert np.allclose(ama.Objective._err__l1(yHat[...,0],yCtg[...,0],None,Y[:,0],(360.,)),[[20.,90.]])

    def test_target_uses_wrapped_multidimensional_distance(self):
        Y=jnp.array([[0.,0.],[170.,1.],[-170.,2.],[90.,0.]])
        obj=ama.Objective('xent',targetSigma=15.)
        obj._Yperiod=(360.,None)
        lQ=np.asarray(obj.log_target(Y))
        D=np.asarray(Y)[None]-np.asarray(Y)[:,None]
        D[...,0]=(D[...,0]+180)%360-180
        ref=-np.sum(D**2,-1)/(2*15.**2)
        ref=ref-np.log(np.exp(ref).sum(-1,keepdims=True))
        assert np.allclose(lQ,ref)


#- divergences on a circle

def circular_w1_lp(p,q,Y,P):
    """1-Wasserstein distance on a circle of circumference P by optimal transport (linear program)"""
    n=len(Y)
    D=np.abs(Y[:,None]-Y[None,:])
    D=np.minimum(D,P-D)
    A_eq=np.vstack([np.kron(np.eye(n),np.ones(n)),np.kron(np.ones(n),np.eye(n))])
    res=linprog(D.ravel(),A_eq=A_eq,b_eq=np.concatenate([p,q]),bounds=(0,None),method='highs')
    return res.fun


class TestCircularDivergences:
    Y=np.array([100.,10.,55.,170.,140.])                           # unsorted, uneven, period 180

    def test_circular_wasserstein_matches_optimal_transport(self):
        rng=np.random.default_rng(0)
        for sigma in (None,20.):
            obj=ama.Objective('wasserstein',targetSigma=sigma)
            obj._Yperiod=(180.,)
            lQ=obj.log_target(jnp.asarray(self.Y))
            lp=jnp.asarray(rng.standard_normal((3,5,5)))
            lp=lp-jnp.log(jnp.sum(jnp.exp(lp),-1,keepdims=True))
            err=np.asarray(ama.Objective._err__wasserstein(lp,None,lQ,jnp.asarray(self.Y),(180.,)))
            P,Q=np.exp(np.asarray(lp)),np.exp(np.asarray(lQ))
            for j in range(3):
                for k in range(5):
                    assert np.isclose(err[j,k],circular_w1_lp(P[j,k],Q[k],self.Y,180.),rtol=1e-6,atol=1e-9)

    def test_circular_fisher_includes_the_wrap_pair(self):
        rng=np.random.default_rng(1)
        obj=ama.Objective('fisher',targetSigma=25.)
        obj._Yperiod=(180.,)
        lQ=np.asarray(obj.log_target(jnp.asarray(self.Y)))
        lp=rng.standard_normal((2,5,5))
        err=np.asarray(ama.Objective._err__fisher(jnp.asarray(lp),None,jnp.asarray(lQ),jnp.asarray(self.Y),(180.,)))
        o=np.argsort(self.Y%180)
        y=np.sort(self.Y%180)
        nxt=np.roll(np.arange(5),-1)
        dY=np.where(nxt>np.arange(5),y[nxt]-y,y[nxt]+180-y)
        for j in range(2):
            for k in range(5):
                a,b=lp[j,k,o],lQ[k,o]
                w=(np.exp(b)+np.exp(b[nxt]))/2
                ref=np.sum(w*((a[nxt]-a)/dY-(b[nxt]-b)/dY)**2)/np.sum(w)
                assert np.isclose(err[j,k],ref)

    def test_linear_wasserstein_is_unchanged_without_period(self):
        lQ=ama.Objective('wasserstein').log_target(jnp.asarray(self.Y))
        lp=jnp.log(jnp.full((1,5,5),0.2))
        lin=np.asarray(ama.Objective._err__wasserstein(lp,None,lQ,jnp.asarray(self.Y)))
        circ=np.asarray(ama.Objective._err__wasserstein(lp,None,lQ,jnp.asarray(self.Y),(180.,)))
        assert np.all(circ<=lin+1e-12)                             # the circle only offers shorter routes


#- category pooling

def test_pooling_kernel_wraps():
    Y=jnp.array([0.,170.,90.])
    K=np.asarray(ama.Model._ctg_kernel(Y,10.,(180.,)))
    assert np.isclose(K[0,1],np.exp(-10.**2/200)) and np.isclose(K[0,2],np.exp(-90.**2/200))
    K2=np.asarray(ama.Model._ctg_kernel(jnp.array([[0.,0.],[3.,4.]]),5.))
    assert np.isclose(K2[0,1],np.exp(-25/50))


#- end to end

class TestTraining:
    def test_circular_orientation(self):
        stim=orientation_stim()
        unit=ama.Unit(stim,ama.Nrn(),ama.Model('gss','mean'),ama.Objective('l2'),
                      ama.Optimizer(nIterMax=100,lRate0=0.05,bVerbose=False))
        unit.train_new(2)
        h=np.asarray(unit.optimizer.loss_hist)
        assert np.all(np.isfinite(h)) and h[-1]<h[0]
        perf=unit.performance('mean')
        assert np.all(np.abs(perf['bias'])<=90) and perf['rmse'].shape==(4,)
        # 0 and 135 degrees are neighbours: an estimate of 170 for 0 is a 10 degree error
        est=unit.estimates('mean')
        assert np.all(est>=-90-1e-9) and np.all(est<=90+1e-9)

    def test_circular_training_differs_from_linear(self):
        x,s,ci,Y,_=ts.image_orientation()
        costs=[]
        for per in (None,180.):
            unit=fixed_unit(ama.Stim(x,s,ci,Y,Yperiod=per),errType='l2')
            costs.append(float(unit.loss))
        assert costs[1]<costs[0]                                    # wrapped errors are never larger

    @pytest.mark.parametrize('errType,obj_kw',[('map',{}),('l2',{}),('l1',{}),('xent',{'targetSigma':0.5})])
    def test_two_latent_dimensions(self,errType,obj_kw):
        x,s,ci,Y,d=grid_ctg()
        train,test=ama.Stim(x,s,ci,Y).train_test(0.25)
        unit=ama.Unit(train,ama.Nrn(),ama.Model('gss','mean'),ama.Objective(errType,**obj_kw),
                      ama.Optimizer(nIterMax=150,lRate0=0.05,bVerbose=False))
        unit.train_new(2)
        h=np.asarray(unit.optimizer.loss_hist)
        assert np.all(np.isfinite(h)) and h[-1]<h[0]
        # the two filters span the two informative directions
        F=np.asarray(unit.out).reshape(len(x),-1)
        assert np.linalg.svd(d.T@np.linalg.qr(F)[0],compute_uv=False).min()>0.8
        perf=unit.performance('mean',stim=test)
        assert perf['rmse'].shape==(9,2) and perf['estimates'].shape[-1]==2
        assert np.isfinite(unit.evaluate(test))

    def test_two_latent_dimensions_with_batches(self):
        x,s,ci,Y,_=grid_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('gss','mean'),ama.Objective('l2'),
                      ama.Optimizer(nIterMax=40,lRate0=0.05,batchSize=90,nBatchMinCtg=4,bVerbose=False))
        unit.train_new(2)
        assert np.all(np.isfinite(np.asarray(unit.optimizer.loss_hist)))

    def test_pooling_with_circular_latent(self):
        stim=orientation_stim()
        unit=ama.Unit(stim,ama.Nrn(),ama.Model('gss','mean',ctgPoolWidth=30.),ama.Objective('map'),
                      ama.Optimizer(nIterMax=30,lRate0=0.05,bVerbose=False))
        unit.train_new(2)
        assert np.isfinite(float(unit.loss))


class TestValidation:
    def test_unsupported_combinations(self):
        x,s,ci,Y,_=grid_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('gss','mean'),ama.Objective('l2',estType='cmean'),
                      ama.Optimizer(nIterMax=1,bVerbose=False))
        with pytest.raises(Exception,match='single latent dimension'):
            unit._finalize(1,[0])
        Yoff=np.asarray(Y,dtype=float).copy()
        Yoff[0]=[-3.,5.]                                            # no longer a grid
        unit=ama.Unit(ama.Stim(x,s,ci,Yoff),ama.Nrn(),ama.Model('gss','mean'),ama.Objective('fisher',targetSigma=1.),
                      ama.Optimizer(nIterMax=1,bVerbose=False))
        with pytest.raises(Exception,match='cartesian grid'):
            unit._finalize(1,[0])
        with pytest.raises(Exception,match='span less than its period'):
            fixed_unit(ama.Stim(x,s,ci,Y*100,Yperiod=(150.,None)))

    def test_held_out_stimuli_need_the_same_geometry(self):
        x,s,ci,Y,_=ts.image_orientation()
        unit=fixed_unit(ama.Stim(x,s,ci,Y,Yperiod=180.))
        with pytest.raises(Exception,match='Yperiod'):
            unit.evaluate(ama.Stim(x,s,ci,Y))

    def test_config_records_the_period(self):
        unit=fixed_unit(orientation_stim())
        assert unit.config()['stim']['Yperiod']==[180.]
        assert unit.objective._key()!=ama.Objective('map')._key()


class TestRegressions:
    def test_shared_objective_keeps_each_units_geometry(self):
        x,s,ci,Y,_=ts.image_orientation()
        obj,m=ama.Objective('l2'),ama.Model('gss','mean')
        a=fixed_unit(ama.Stim(x,s,ci,Y,Yperiod=180.))
        a.objective,a.model=obj,m
        a._set_geometry()
        before=float(a.loss)
        b=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),m,obj,ama.Optimizer(nIterMax=1,bVerbose=False))
        assert float(a.loss)==before
        assert b.objective is not obj and b.objective._Yperiod is None
        c=ama.Unit(ama.Stim(x,s,ci,Y,Yperiod=180.),ama.Nrn(),m,obj,ama.Optimizer(nIterMax=1,bVerbose=False))
        assert c.objective is obj                                   # the same geometry is shared

    def test_circular_mean_gradient_with_a_cancelling_linear_dimension(self):
        import jax
        Y=jnp.array([[10.,0.],[20.,0.5]])                          # exp(2 pi i Y) cancels on the linear dimension
        g=jax.grad(lambda lp: jnp.sum(ama.Objective._est__mean(lp,Y,(360.,None))))(jnp.log(jnp.array([[0.5,0.5]])))
        assert np.all(np.isfinite(np.asarray(g)))
        assert np.allclose(np.asarray(g)[0,1],-np.asarray(g)[0,0])


#- several latent dimensions: wasserstein and fisher

def w1_lp(p,q,D):
    n=len(p)
    A_eq=np.vstack([np.kron(np.eye(n),np.ones(n)),np.kron(np.ones(n),np.eye(n))])
    return linprog(D.ravel(),A_eq=A_eq,b_eq=np.concatenate([p,q]),bounds=(0,None),method='highs').fun


def grid_objective(errType,Y,per,**kw):
    obj=ama.Objective(errType,**kw)
    obj._Yperiod=per
    obj._Ygrid=ama._latent_grid(Y)
    return obj


class TestSeveralDimensions:
    Y=np.array([(a,b) for b in (0.,1.,2.5) for a in (10.,100.,190.,280.)])[np.random.default_rng(0).permutation(12)]
    per=(360.,None)

    def D(self):
        d=self.Y[:,None]-self.Y[None]
        d[...,0]=(d[...,0]+180)%360-180
        return np.sqrt((d**2).sum(-1))

    def test_latent_grid(self):
        shape,order=ama._latent_grid(self.Y)
        assert shape==(4,3)
        g=self.Y[list(order)].reshape(4,3,2)
        assert np.all(np.diff(g[...,0],axis=0)>0) and np.all(np.diff(g[...,1],axis=1)>0)
        assert ama._latent_grid(self.Y[:-1]) is None

    def test_one_hot_wasserstein_is_exact(self):
        obj=grid_objective('wasserstein',self.Y,self.per)
        Yj=jnp.asarray(self.Y)
        lQ=obj.log_target(Yj)
        lp=jnp.asarray(np.random.default_rng(1).standard_normal((2,12,12)))
        lp=lp-jnp.log(jnp.sum(jnp.exp(lp),-1,keepdims=True))
        err=np.asarray(ama.Objective._err__wasserstein(lp,None,lQ,Yj,self.per,obj._err_opts()))
        P,Q=np.exp(np.asarray(lp)),np.exp(np.asarray(lQ))
        for j,k in [(0,0),(1,5),(0,11)]:
            assert np.isclose(err[j,k],w1_lp(P[j,k],Q[k],self.D()),rtol=1e-6)

    def test_sinkhorn_divergence_approaches_the_exact_distance(self):
        Yj=jnp.asarray(self.Y)
        lp=jnp.asarray(np.random.default_rng(2).standard_normal((1,12,12)))
        lp=lp-jnp.log(jnp.sum(jnp.exp(lp),-1,keepdims=True))
        P=np.exp(np.asarray(lp))
        errs={}
        for eps,nIter in [(0.05,300),(0.005,4000)]:
            obj=grid_objective('wasserstein',self.Y,self.per,targetSigma=60.,otEps=eps,nOtIter=nIter)
            lQ=obj.log_target(Yj)
            errs[eps]=np.asarray(ama.Objective._err__wasserstein(lp,None,lQ,Yj,self.per,obj._err_opts()))
            Q=np.exp(np.asarray(lQ))
        exact=np.array([w1_lp(P[0,k],Q[k],self.D()) for k in range(12)])
        assert np.all(np.abs(errs[0.005][0]-exact)<0.02*exact.max())
        assert np.abs(errs[0.005][0]-exact).mean()<np.abs(errs[0.05][0]-exact).mean()
        # zero at the target
        z=np.asarray(ama.Objective._err__wasserstein(lQ[None],None,lQ,Yj,self.per,obj._err_opts()))
        assert np.allclose(z,0,atol=1e-6*exact.max())

    def test_sinkhorn_gradient_matches_finite_differences(self):
        import jax
        Yj=jnp.asarray(self.Y)
        obj=grid_objective('wasserstein',self.Y,self.per,targetSigma=60.,otEps=0.05,nOtIter=2000)
        lQ=obj.log_target(Yj)
        f=lambda z: jnp.sum(ama.Objective._err__wasserstein(z-jax.scipy.special.logsumexp(z,-1,keepdims=True),None,lQ,Yj,
                                                             self.per,obj._err_opts()))
        z=jnp.asarray(np.random.default_rng(3).standard_normal((1,12,12)))
        g=np.asarray(jax.grad(f)(z))
        for idx in [(0,0,0),(0,4,7),(0,11,3)]:
            e=np.zeros(z.shape); e[idx]=1e-5
            assert np.isclose(g[idx],(f(z+e)-f(z-e))/2e-5,rtol=1e-3,atol=1e-6)

    def test_grid_fisher_matches_numpy(self):
        Yj=jnp.asarray(self.Y)
        obj=grid_objective('fisher',self.Y,self.per,targetSigma=80.)
        lQ=np.asarray(obj.log_target(Yj))
        lp=np.random.default_rng(4).standard_normal((2,12,12))
        err=np.asarray(ama.Objective._err__fisher(jnp.asarray(lp),None,jnp.asarray(lQ),Yj,self.per,obj._err_opts()))
        shape,order=ama._latent_grid(self.Y)
        g=self.Y[list(order)].reshape(4,3,2)
        a0,a1=g[:,0,0],g[0,:,1]
        for j,k in [(0,0),(1,7)]:
            A=lp[j,k][list(order)].reshape(4,3)
            B=lQ[k][list(order)].reshape(4,3)
            q=np.exp(B)
            ref=0.
            # axis 0 circular (period 360): 4 pairs including 280 -> 10
            nx=np.roll(np.arange(4),-1)
            dy=np.where(nx>np.arange(4),a0[nx]-a0,a0[nx]+360-a0)[:,None]
            w=(q[nx]+q)/2
            ref+=np.sum(w*((A[nx]-A)/dy-(B[nx]-B)/dy)**2)/w.sum()
            # axis 1 linear
            dy=np.diff(a1)[None]
            w=(q[:,1:]+q[:,:-1])/2
            ref+=np.sum(w*(np.diff(A,axis=1)/dy-np.diff(B,axis=1)/dy)**2)/w.sum()
            assert np.isclose(err[j,k],ref)

    @pytest.mark.parametrize('errType,obj_kw',[('wasserstein',{}),('wasserstein',{'targetSigma':0.8,'nOtIter':100}),
                                               ('fisher',{'targetSigma':0.8})])
    def test_training(self,errType,obj_kw):
        x,s,ci,Y,_=grid_ctg()
        unit=ama.Unit(ama.Stim(x,s,ci,Y),ama.Nrn(),ama.Model('gss','mean'),ama.Objective(errType,**obj_kw),
                      ama.Optimizer(nIterMax=100,lRate0=0.05,bVerbose=False))
        unit.train_new(2)
        h=np.asarray(unit.optimizer.loss_hist)
        assert np.all(np.isfinite(h)) and h[-1]<h[0]


class TestCircularMedian:
    Y=np.array([100.,10.,55.,170.,140.])                           # period 180

    def reference(self,p,Y,P):
        """numpy: the minimizer of the expected wrapped distance among levels and antipodes, then the interpolated
        median of the distribution unwrapped around it"""
        wrap=lambda d: (d+P/2)%P-P/2
        cand=np.concatenate((Y,Y+P/2))
        m0=cand[np.argmin([(p*np.abs(wrap(Y-c))).sum() for c in cand])]
        yu=m0+wrap(Y-m0)
        o=np.argsort(yu)
        return wrap(np.interp(0.5,np.cumsum(p[o]),yu[o]))

    def test_matches_reference(self):
        rng=np.random.default_rng(0)
        lp=rng.standard_normal((20,5))*2
        est=np.asarray(ama.Objective._est__median(jnp.asarray(lp),jnp.asarray(self.Y),(180.,)))
        P=np.exp(lp)/np.exp(lp).sum(-1,keepdims=True)
        for r in range(20):
            assert np.isclose(est[r],self.reference(P[r],self.Y,180.))

    def test_mass_across_the_wrap_point(self):
        Y=jnp.array([0.,45.,90.,135.])
        p=np.array([0.45,0.,0.,0.55])                              # 0 and 135 = -45: the median is between them
        lp=jnp.log(jnp.asarray(p+1e-12))[None]
        est=float(ama.Objective._est__median(lp,Y,(180.,))[0])
        # the same distribution on levels that do not cross the cut (90 = -90, 135 = -45): the ordinary median
        unwrapped=float(ama.Objective._est__median(lp,jnp.array([0.,45.,-90.,-45.]))[0])
        assert np.isclose(est,unwrapped) and -90<est<0
        assert float(ama.Objective._est__median(lp,Y)[0])>45        # the linear median lands on the far side

    def test_gradient_and_l1_training(self):
        import jax
        g=jax.grad(lambda lp: jnp.sum(ama.Objective._est__median(lp,jnp.asarray(self.Y),(180.,))))(
            jnp.asarray(np.random.default_rng(1).standard_normal((3,5))))
        assert np.all(np.isfinite(np.asarray(g))) and np.any(np.asarray(g)!=0)
        unit=ama.Unit(orientation_stim(),ama.Nrn(),ama.Model('gss','mean'),ama.Objective('l1'),
                      ama.Optimizer(nIterMax=60,lRate0=0.05,bVerbose=False))
        unit.train_new(2)
        h=np.asarray(unit.optimizer.loss_hist)
        assert np.all(np.isfinite(h)) and h[-1]<h[0]

    def test_mixed_dimensions(self):
        Y=jnp.array([[10.,0.],[100.,1.],[170.,2.]])
        lp=jnp.asarray(np.random.default_rng(2).standard_normal((4,3)))
        est=np.asarray(ama.Objective._est__median(lp,Y,(180.,None)))
        assert np.allclose(est[:,0],np.asarray(ama.Objective._est__median(lp,Y[:,0],(180.,))))
        assert np.allclose(est[:,1],np.asarray(ama.Objective._est__median(lp,Y[:,1])))
