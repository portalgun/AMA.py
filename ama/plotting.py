"""Figures of a Unit: filters, filter banks, and response embeddings (t-SNE, PaCMAP, PHATE)."""
from ._base import *
from .nrn import Nrn


class _Plotting:
    """Unit methods: figures"""

    #- plot
    def plot_out(self,bFourier=None,name='f_out'):
        self.filter.plot_out(bFourier=bFourier,name=name)

    def plot_last(self,bFourier=None,name='f_last'):
        self.filter.plot_last(bFourier=bFourier,name=name)

    def _bank_info(self,bankInfo=None):
        """filter-bank metadata for the figures: multiscale_values() or parametric_values(), else bankInfo (a dict with any of
        k_peak, orientations, interleaved, mother, log_freq_knots, nMothers, mother_index), else None"""
        info=None
        for attr in ('multiscale_out','param_out'):
            if getattr(self,attr,None) is not None:
                info=dict(getattr(self,attr))
                break
        if info is None and bankInfo is not None:
            info=dict(bankInfo)
        if info is None:
            return None
        n=int(self.filter.n)
        if 'k_peak' in info and np.size(info['k_peak'])!=n:
            info.pop('k_peak')
        M=info.get('mother')
        info.setdefault('nMothers',1 if M is None or np.ndim(M)==1 else np.shape(M)[0])
        info.setdefault('mother_index',np.repeat(np.arange(info['nMothers']),n//info['nMothers']))
        return info

    def _figure_title(self,name,what):
        name=self.name if name is None else name
        return what if not name else str(name) + '\n' + what

    def plot_filter_bank(self,fname=None,name=None,bankInfo=None,perRow=8,dpi=110):
        """
        figure of the learned filter bank. 1D: one row of implied spatial filters per mother filter (real part solid,
        quadrature pair dotted, in that mother's colour), then a log-frequency and a linear-frequency row in which each
        column (scale) overlays the amplitude spectra of all mothers at that scale; side panels with the pooling weights
        (readoutType resultant/resultant_only, bars in the mother colours), the mother filter(s) of a multiscale bank
        (amplitude solid and phase/pi dotted against log(k/k_peak)), and the pooling-weighted sum of the spectra of each
        scale on linear and log frequency axes. 2D: real part and amplitude spectrum (linear frequency axes) of every
        filter, with peak frequency, orientation and interleaving in the panel titles. name (default unit.name) titles the
        figure; bankInfo supplies bank metadata for units without multiscale_values()/parametric_values(). Saves to fname
        when given; returns the figure.
        """
        flt=self.filter
        dims=tuple(flt.pix_dims)
        g=np.asarray(flt.implied_spatial())
        parts=None
        if flt.bSplit:                                                    # one column per sub-filter (e.g. per eye)
            nSp=int(flt.nSplit)
            g=np.reshape(g,tuple(flt.pix_dims)+(-1,))
            parts=np.tile(np.arange(nSp),g.shape[-1]//nSp) if g.shape[-1]%nSp==0 else None
        n=g.shape[-1]
        F=np.asarray(self.out).reshape(dims+(n,)) if flt.bIsFourier and not flt.bSplit else None
        w=None if self.pool_p is None else np.asarray(Nrn.pool_weights(np.asarray(self.pool_p,float).ravel()))
        mv=self._bank_info(bankInfo)
        per_row=perRow if n%perRow==0 else min(n,perRow)
        rows=int(np.ceil(n/per_row))
        kp=lambda j: '' if mv is None or 'k_peak' not in mv else ' k=%.3f'%np.asarray(mv['k_peak'])[j]
        if len(dims)==1:
            N=dims[0]
            x=np.arange(N)-N/2
            f=np.fft.fftshift(np.fft.fftfreq(N))
            pos=f>0
            amp=np.abs(F) if F is not None else np.abs(np.fft.fftshift(np.fft.fft(np.real(g),axis=0),axes=0))
            nM=int(mv['nMothers']) if mv is not None else 1
            nS=max(1,n//nM)
            midx=np.asarray(mv['mother_index']) if mv is not None else np.zeros(n,int)
            colors=[cm.tab10(i%10) for i in range(nM)]
            # rows: one spatial row per mother, then the log- and linear-frequency rows (all mothers of a scale overlaid)
            fig=plt.figure(figsize=(2.2*nS+3.6,2.6*max(nM+2,4)+1.4))
            outer=fig.add_gridspec(1,2,width_ratios=[nS,1.7],wspace=0.18)
            gs=outer[0,0].subgridspec(nM+2,nS)                             # filters: nM spatial rows, then log and linear
            side=outer[0,1].subgridspec(4,1,hspace=0.45)                   # weights, mother(s), weighted sums (linear, log)
            for j in range(n):
                m_,s_=(int(midx[j]),j%nS) if mv is not None else (0,j)
                ax=fig.add_subplot(gs[m_,s_])
                ax.plot(x,np.real(g[:,j]),lw=0.8,color=colors[m_],label='real')
                ax.plot(x,np.imag(g[:,j]),lw=0.8,ls=':',color=colors[m_],label='quadrature')
                ttl='f'+str(j)+(' m'+str(m_) if nM>1 else '')+('' if parts is None else ' part'+str(parts[j]))+kp(j)
                if w is not None and w.size==n:
                    ttl+='\nw=%.2f'%w[j]
                ax.set_title(ttl,fontsize=6)
                ax.tick_params(labelsize=5)
                ax.set_xlabel('px',fontsize=5)
                if j==0:
                    ax.legend(fontsize=5)
            for s_ in range(nS):
                js=[j for j in range(n) if (int(midx[j]),j%nS)[1]==s_] if mv is not None else [s_]
                for row,(scale,xlab) in enumerate(((True,'cycles/px (log)'),(False,'cycles/px (linear)'))):
                    ax=fig.add_subplot(gs[nM+row,s_])
                    for j in js:
                        (ax.semilogx if scale else ax.plot)(f[pos],amp[pos,j],lw=0.8,color=colors[int(midx[j])])
                    ax.set_xlim(f[pos][0] if scale else 0,0.5)
                    ax.tick_params(labelsize=5)
                    ax.set_xlabel(xlab,fontsize=5)
                    if s_==0:
                        ax.set_ylabel('amplitude',fontsize=5)
            ax=fig.add_subplot(side[0])
            if w is not None:
                ax.bar(np.arange(w.size),w,color=[colors[int(midx[j])] for j in range(min(n,w.size))])
                ax.set_title('pooling weights',fontsize=7)
            else:
                ax.text(0.5,0.5,'no pooling readout',ha='center',va='center',fontsize=7)
                ax.axis('off')
            ax.tick_params(labelsize=6)
            ax=fig.add_subplot(side[1])
            if mv is not None and mv.get('mother') is not None and 'log_freq_knots' in mv:
                M=np.asarray(mv['mother'])
                M=M[None] if M.ndim==1 else M
                for im in range(M.shape[0]):
                    ax.plot(mv['log_freq_knots'],np.abs(M[im]),lw=1,color=colors[im])
                    ax.plot(mv['log_freq_knots'],np.angle(M[im])/np.pi*np.abs(M[im]).max(),lw=0.7,ls=':',color=colors[im])
                handles=[plt.Line2D([],[],color='k',lw=1,label='amplitude'),plt.Line2D([],[],color='k',lw=0.7,ls=':',label='phase')]
                handles+=[plt.Line2D([],[],color=colors[im],lw=1,label='mother '+str(im)) for im in range(M.shape[0])]
                ax.legend(handles=handles,fontsize=5)
                ax.set_xlabel('log(k / k_peak)',fontsize=6)
                ax.set_title('mother filter(s)',fontsize=7)
            else:
                for j in range(n):
                    ax.semilogx(f[pos],amp[pos,j],lw=0.7,color=colors[int(midx[j])])
                ax.set_title('all amplitude spectra (log frequency)',fontsize=7)
                ax.set_xlabel('cycles/px (log)',fontsize=6)
            ax.tick_params(labelsize=6)
            # weighted sum of each scale (pooling weights; uniform when there is no readout), linear above log
            ww=np.ones(n)/n if w is None or w.size!=n else w
            scale_sum=np.stack([sum(ww[j]*amp[:,j] for j in range(n) if (j%nS)==s_) for s_ in range(nS)],1)
            scolors=[cm.viridis(i/max(nS-1,1)) for i in range(nS)]
            for row,scale in ((2,False),(3,True)):
                ax=fig.add_subplot(side[row])
                for s_ in range(nS):
                    (ax.semilogx if scale else ax.plot)(f[pos],scale_sum[pos,s_],lw=0.8,color=scolors[s_],
                                                        label=('scale '+str(s_)) if not scale else None)
                (ax.semilogx if scale else ax.plot)(f[pos],scale_sum[pos].sum(1),lw=1.1,color='k',label='total' if not scale else None)
                ax.set_xlim(f[pos][0] if scale else 0,0.5)
                ax.set_title('weighted sum per scale ('+('log' if scale else 'linear')+' frequency)',fontsize=7)
                ax.set_xlabel('cycles/px ('+('log' if scale else 'linear')+')',fontsize=6)
                ax.tick_params(labelsize=6)
                if not scale:
                    ax.legend(fontsize=5,ncol=2)
        elif len(dims)==2:
            fig=plt.figure(figsize=(1.6*per_row+2.5,3.4*rows+1.2))
            gs=fig.add_gridspec(2*rows,per_row+1)
            for j in range(n):
                r_,c_=divmod(j,per_row)
                ax=fig.add_subplot(gs[2*r_,c_])
                v=np.real(g[...,j])
                lim=np.abs(v).max()
                ax.imshow(v,cmap='RdBu_r',vmin=-lim,vmax=lim)
                ori='' if mv is None or 'orientations' not in mv else ' %d deg'%round(np.degrees(np.asarray(mv['orientations'])[j]))
                il='*' if mv is not None and 'interleaved' in mv and np.asarray(mv['interleaved'])[j] else ''
                ax.set_title('f'+str(j)+il+kp(j)+ori,fontsize=5)
                ax.axis('off')
                ax=fig.add_subplot(gs[2*r_+1,c_])
                A=np.abs(F[...,j]) if F is not None else np.abs(np.fft.fftshift(np.fft.fft2(v)))
                ax.imshow(A,cmap='magma',extent=(-0.5,0.5,0.5,-0.5))
                ax.set_xticks([-0.5,0,0.5])
                ax.set_yticks([-0.5,0,0.5])
                ax.tick_params(labelsize=4)
                if j==0:
                    ax.set_xlabel('cycles/px (linear)',fontsize=5)
            ax=fig.add_subplot(gs[:,per_row])
            if w is not None:
                ax.barh(np.arange(w.size),w)
                ax.set_title('pooling weights',fontsize=7)
            else:
                ax.text(0.5,0.5,'no pooling readout\n* = interleaved\nrows: real part,\none-sided |spectrum|',ha='center',va='center',fontsize=7)
                ax.axis('off')
        else:
            raise Exception('plot_filter_bank supports 1D and 2D filters')
        fig.suptitle(self._figure_title(name,'filters'),fontsize=9)
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname,dpi=dpi)
        return fig

    @staticmethod
    def _embed(feats,method,seed,**kw):
        """2D embedding of [ nStim x nFeatures ]: 't-sne' / 'tsne' (sklearn), 'pacmap', or 'phate'"""
        method=method.lower().replace('-','')
        if method=='tsne':
            from sklearn.manifold import TSNE
            kw.setdefault('perplexity',30)
            return TSNE(n_components=2,init='pca',learning_rate='auto',random_state=seed,**kw).fit_transform(feats),'t-SNE'
        if method=='pacmap':
            import pacmap
            kw.setdefault('n_neighbors',10)
            return pacmap.PaCMAP(n_components=2,random_state=seed,**kw).fit_transform(feats,init='pca'),'PaCMAP'
        if method=='phate':
            import phate
            kw.setdefault('knn',10)
            return phate.PHATE(n_components=2,random_state=seed,verbose=0,**kw).fit_transform(feats),'PHATE'
        raise Exception("method must be 'tsne', 'pacmap' or 'phate'")

    def _response_features(self,stim,nMax,seed,cmap=None):
        """(features [ nStim x nFeatures ], responses u [ nF x nStim ], latent [ nStim ], cmap) for the embeddings:
        noise-free responses of every filter to up to nMax category-balanced stimuli, as unit phasors for normalizeType
        'phase'"""
        st=copy.copy(stim)
        if st.bIsFourier:
            st._ifft()
        if self.filter.bSplit and not st.bIsSplit:
            st.nSplit=int(self.filter.nSplit)                              # stimuli built without nSplit: the filters know it
            st.split()
        elif st.bIsSplit and not self.filter.bSplit:
            st.unsplit()
        rng=np.random.default_rng(seed)
        valid=np.asarray(st.weights)>0                                    # [ nStim_Ctg x nCtg ]
        per=max(1,nMax//st.nCtg)
        cols=[]
        for c in range(st.nCtg):
            idx=np.flatnonzero(valid[:,c])
            k=min(per,idx.size)
            cols.append(np.stack([rng.choice(idx,k,replace=False),np.full(k,c)],1))
        sel=np.concatenate(cols)
        Yv=np.asarray(st.Y) if np.ndim(st.Y)==1 else np.asarray(st.Y)[:,0]      # color by the first latent dimension
        lat=Yv[sel[:,1]]
        g=np.asarray(self.filter.implied_spatial())
        if self.filter.bSplit:
            # each sub-filter (e.g. one eye) responds as its own neuron: [ nSplit x nF ] response dimensions
            nS=int(self.filter.nSplit)
            nP=st.nPix//nS
            g=g.reshape(nP,nS,-1)
            S=np.real(np.asarray(st.val).reshape(nP,nS,st.nStim_Ctg,st.nCtg)[:,:,sel[:,0],sel[:,1]])
            r=np.einsum('psj,psn->sjn',g,S).reshape(-1,S.shape[-1])
        else:
            S=np.real(np.asarray(st.val).reshape(st.nPix,st.nStim_Ctg,st.nCtg)[:,sel[:,0],sel[:,1]])
            r=g.reshape(st.nPix,-1).T@S                                   # [ nF x nStim ]
        u=r/np.maximum(np.abs(r),1e-12) if self.nrn.normalizeType=='phase' else r
        feats=(np.concatenate([u.real,u.imag],0) if np.iscomplexobj(u) else u).T
        if cmap is None:
            bCirc=getattr(st,'Yperiod',None) is not None and st.Yperiod[0] is not None
            cmap='twilight' if bCirc or (Yv.min()>=0 and Yv.max()<2*np.pi+1e-9 and np.ptp(Yv)>np.pi) else 'viridis'
        return feats,u,lat,cmap

    def plot_response_embeddings(self,stim,methods=('tsne','pacmap','phate'),fname=None,name=None,nMax=2000,seed=0,cmap=None,
                                 dpi=110,method_kw=None,**embed_kw):
        """
        one figure with every embedding of the same responses side by side (t-SNE, PaCMAP, PHATE by default), and, with
        pooling weights, the pooled resultant as a final panel. The responses (and the colour scale) are computed once, so
        the panels are comparable. method_kw: {method: its own keyword arguments}; embed_kw apply to every method.
        name (default unit.name) titles the figure. Saves to fname when given; returns the figure.
        """
        feats,u,lat,cmap=self._response_features(stim,nMax,seed,cmap)
        w=None if self.pool_p is None else np.asarray(Nrn.pool_weights(np.asarray(self.pool_p,float).ravel()))
        bPooled=w is not None and w.size==u.shape[0] and np.iscomplexobj(u)
        method_kw=method_kw or {}
        embs=[self._embed(feats,m,seed,**{**embed_kw,**method_kw.get(m,{})}) for m in methods]
        nax=len(embs)+(1 if bPooled else 0)
        fig,axes=plt.subplots(1,nax,figsize=(5.2*nax,5.3),squeeze=False)
        for i,(emb,label) in enumerate(embs):
            sc=axes[0,i].scatter(emb[:,0],emb[:,1],c=lat,cmap=cmap,s=4)
            axes[0,i].set_title(label,fontsize=9)
            axes[0,i].set_xticks([])
            axes[0,i].set_yticks([])
            if i==len(embs)-1:
                fig.colorbar(sc,ax=axes[0,i],label='latent')
        if bPooled:
            z=(w[:,None]*u).sum(0)
            axes[0,-1].scatter(z.real,z.imag,c=lat,cmap=cmap,s=4)
            axes[0,-1].set_aspect('equal')
            axes[0,-1].set_title('pooled resultant sum_j p_j r_j',fontsize=9)
            axes[0,-1].tick_params(labelsize=6)
        fig.suptitle(self._figure_title(name,('phase-normalized ' if self.nrn.normalizeType=='phase' else '')
                                        +'responses of '+str(u.shape[0])+' filters, '+str(feats.shape[0])+' stimuli'),fontsize=9)
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname,dpi=dpi)
        return fig

    def plot_response_embedding(self,stim,method='tsne',fname=None,name=None,nMax=2000,seed=0,cmap=None,dpi=110,**embed_kw):
        """
        2D embedding of the noise-free responses of every filter to up to nMax category-balanced stimuli (a Stim, e.g.
        held-out test stimuli; fourier-domain stimuli are transformed back), coloured by the latent value. method: 'tsne'
        (sklearn t-SNE, perplexity 30, PCA initialization), 'pacmap' (PaCMAP, n_neighbors 10, PCA initialization) or 'phate'
        (PHATE, knn 10); embed_kw go to the method. Responses are unit phasors (real and imaginary parts of r_j / |r_j|) for
        normalizeType 'phase', else the responses themselves. With pooling weights (readoutType resultant/resultant_only) a
        second panel shows the pooled resultant sum_j p_j r_j (what the likelihood sees) in the complex plane. cmap defaults
        to 'twilight' (circular latents) when Y spans more than pi within [0, 2 pi), else 'viridis'. name (default
        unit.name) titles the figure. Saves to fname when given; returns the figure.
        """
        feats,u,lat,cmap=self._response_features(stim,nMax,seed,cmap)
        emb,label=self._embed(feats,method,seed,**embed_kw)
        w=None if self.pool_p is None else np.asarray(Nrn.pool_weights(np.asarray(self.pool_p,float).ravel()))
        bPooled=w is not None and w.size==u.shape[0] and np.iscomplexobj(u)
        fig,axes=plt.subplots(1,2 if bPooled else 1,figsize=(11 if bPooled else 6,5.3),squeeze=False)
        sc=axes[0,0].scatter(emb[:,0],emb[:,1],c=lat,cmap=cmap,s=4)
        axes[0,0].set_title(label+' of '+('phase-normalized ' if self.nrn.normalizeType=='phase' else '')+'responses ('
                            +str(u.shape[0])+' filters, '+str(feats.shape[0])+' stimuli)',fontsize=8)
        axes[0,0].set_xticks([])
        axes[0,0].set_yticks([])
        fig.colorbar(sc,ax=axes[0,0],label='latent')
        if bPooled:
            z=(w[:,None]*u).sum(0)
            axes[0,1].scatter(z.real,z.imag,c=lat,cmap=cmap,s=4)
            axes[0,1].set_aspect('equal')
            axes[0,1].set_title('pooled resultant sum_j p_j r_j (what the likelihood sees)',fontsize=8)
            axes[0,1].tick_params(labelsize=6)
        fig.suptitle(self._figure_title(name,'response '+label),fontsize=9)
        fig.tight_layout()
        if fname is not None:
            fig.savefig(fname,dpi=dpi)
        return fig

    def plot_response_tsne(self,stim,fname=None,name=None,**kw):
        """plot_response_embedding with method 'tsne'"""
        return self.plot_response_embedding(stim,'tsne',fname=fname,name=name,**kw)

    def plot_response_pacmap(self,stim,fname=None,name=None,**kw):
        """plot_response_embedding with method 'pacmap'"""
        return self.plot_response_embedding(stim,'pacmap',fname=fname,name=name,**kw)

    def plot_response_phate(self,stim,fname=None,name=None,**kw):
        """plot_response_embedding with method 'phate'"""
        return self.plot_response_embedding(stim,'phate',fname=fname,name=name,**kw)

    def save_figures(self,stem,stim,name=None,bankInfo=None,methods=('tsne','pacmap','phate'),**embed_kw):
        """
        write <stem>_filters.png (plot_filter_bank) and <stem>_<method>.png (plot_response_embedding on stim) for each
        embedding method; closes the figures. With more than one method it also writes <stem>_embeddings.png, all of them in
        one figure (plot_response_embeddings). methods: a sequence of method names, or a dict of method -> its own keyword
        arguments (embed_kw, applying to every method, is for arguments they share, e.g. nMax and seed).
        """
        plt.close(self.plot_filter_bank(str(stem)+'_filters.png',name=name,bankInfo=bankInfo))
        items=dict(methods) if isinstance(methods,dict) else {m:{} for m in methods}
        for m,kw in items.items():
            plt.close(self.plot_response_embedding(stim,m,str(stem)+'_'+m.replace('-','')+'.png',name=name,**{**embed_kw,**kw}))
        if len(items)>1:
            plt.close(self.plot_response_embeddings(stim,tuple(items),str(stem)+'_embeddings.png',name=name,method_kw=items,**embed_kw))


__all__=['_Plotting']
