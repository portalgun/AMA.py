"""Response: the responses of a Unit to its stimuli, with plots of their distributions."""
from ._base import *


class Response():

    def __init__(self,r,rNs,R,RNs,RVar,stim):
        self.r=r
        self.rNs=rNs
        self.R=R
        self.RNs=RNs
        self.RVar=RVar

        self.bFourier=stim.bIsFourier

        self._stim=stim
        self.yCtgInd=stim.yCtgInd
        self.yCtg=stim.yCtg
        self.weights=stim.weights

    @property
    def nCtg(self):
        return self.r.shape[-1]

    @property
    def nF(self):
        return self.r.shape[0]

    @property
    def nStim_Ctg(self):
        return self.r.shape[-2]

    @property
    def nStim(self):
        return self.nStim_Ctg*self.nCtg

    @property
    def shape(self):
        return self.r.shape

    @property
    def bSplit(self):
        return self.r.ndim==4

    @property
    def nSplit(self):
        if not self.bSplit:
            return 0
        else:
            return self.r.shape[1]

    def _component(self,r,iF,iSplit,iComp,iCtg):
        c=np.real if iComp==0 else np.imag
        mask=np.asarray(self.weights[:,iCtg])>0
        if self.bSplit:
            return c(np.asarray(r[iF,iSplit,:,iCtg]))[mask]
        return c(np.asarray(r[iF,:,iCtg]))[mask]

    def _parts(self):
        splits=range(self.nSplit) if self.bSplit else (0,)
        components=range(2 if np.iscomplexobj(self.r) else 1)
        return splits,components

    def plot_marginal(self,fld='RNs',name='plot_marginal_responses'):
        r=getattr(self,fld)
        colors=cm.rainbow(np.linspace(0,1,self.nCtg))
        splits,components=self._parts()

        for iF, iS, iC in product(range(self.nF),splits,components):
            plt.figure(name + '_' + str(iF) + '_' + str(iS) + '_' + str(iC))
            for i in range(self.nCtg):
                R1=self._component(r,iF,iS,iC,i)
                plt.hist(R1,color=colors[i],alpha=.4)

    def plot_joint(self,fld='RNs',name='plot_joint_responses'):
        """
        R    [ nF    x nStim ] -> [ nF    x nStim_Ctg x nCtg]         [ nF x nSplit x nStim_Ctg x nCtg ]
        """
        # TODO plot marginals at left and bottom
        r=getattr(self,fld)
        colors=cm.rainbow(np.linspace(0,1,self.nCtg))
        splits,components=self._parts()

        # (f1,f2), (split1,split2), (component1,component2)
        pairs=[p + s + c for p, s, c in product(combinations(range(self.nF),2), product(splits,splits), product(components,components))]

        for j,p in enumerate(pairs):
            plt.figure(name + str(j))
            for i in range(self.nCtg):
                R1=self._component(r,p[0],p[2],p[4],i)
                R2=self._component(r,p[1],p[3],p[5],i)
                plt.scatter(R1,R2,color=colors[i],marker='.',alpha=.4)

    def plot_tsne(self,fld='RNs',name='plot_tsne',n_components=2,**kwargs):
        """
        n_components - ndims to plot
        perplexity   - number of neighbors to consider (5-50)
        """
        r=np.asarray(_flatten_responses(getattr(self,fld)))    # [ nF' x nStim_Ctg x nCtg ]
        mask=np.asarray(self.weights).ravel()>0
        X=r.reshape(r.shape[0],-1).T[mask]
        yCtgInd=np.asarray(self.yCtgInd).ravel()[mask]

        from sklearn.manifold import TSNE
        colors=cm.rainbow(np.linspace(0,1,self.nCtg))
        t=TSNE(n_components=n_components,**kwargs).fit_transform(X).T

        fig=plt.figure(name)
        if n_components==3:
            ax=fig.add_subplot(projection='3d')
        elif n_components==2:
            ax=fig.add_subplot()
        else:
            raise Exception('n_components must be 2 or 3')
        for i in range(self.nCtg):
            inds=yCtgInd==i
            ax.scatter(*t[:,inds],color=colors[i])


__all__=['Response']
