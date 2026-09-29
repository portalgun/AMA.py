"""
Figures: Unit.plot_filter_bank, plot_response_tsne, save_figures, and the unit name (titles, save/load, config).
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import ama
import stimuli as ts


def trained(gen=ts.sine_frequency,readout='resultant_only',name='test / natural / phase pooled',**train_kw):
    x,s,ci,Y,_=gen()
    stim=ama.Stim(x,s,ci,Y)
    unit=ama.Unit(stim,ama.Nrn(normalizeType='phase',readoutType=readout),ama.Model('circ','mean',covShrink=0.5),
                  ama.Objective('map'),ama.Optimizer(nIterMax=5,bVerbose=False),name=name)
    return unit,stim


def test_filter_bank_and_tsne_1d(tmp_path):
    unit,stim=trained()
    unit.train_multiscale(3,nKnot=9,nMothers=2,edgeTaper=0.2)
    fig=unit.plot_filter_bank(tmp_path/'f.png')
    assert (tmp_path/'f.png').exists() and fig._suptitle.get_text().startswith('test / natural / phase pooled')
    labels=[a.get_title() for a in fig.axes]
    assert 'pooling weights' in labels and any('mother' in t for t in labels)
    assert any('linear' in a.get_xlabel() for a in fig.axes) and any('(log)' in a.get_xlabel() for a in fig.axes)
    fig=unit.plot_response_tsne(stim,tmp_path/'t.png',name='other name',nMax=60,perplexity=5)   # training stim: fourier after finalize
    assert (tmp_path/'t.png').exists() and fig._suptitle.get_text().startswith('other name') and len(fig.axes)>=3


def test_save_figures_2d_and_unit_without_metadata(tmp_path):
    unit,stim=trained(ts.image_orientation,readout='None',name=None)
    unit.train_multiscale(nKnot=9,nKnotTheta=7,kTop=0.3,ratio=1.6,nScales=2,nOrientations=2,bInterleave=True)
    unit.save_figures(tmp_path/'u',stim,name='grid',methods=('tsne',),nMax=60,perplexity=5)
    assert (tmp_path/'u_filters.png').exists() and (tmp_path/'u_tsne.png').exists()
    free,stim=trained(name=None)
    free.train_new(2,fourierType=2)
    fig=free.plot_filter_bank()
    assert fig._suptitle.get_text()=='filters'


def test_name_saved_loaded_and_in_config(tmp_path):
    unit,stim=trained()
    unit.train_multiscale(3,nKnot=9)
    f=str(tmp_path/'u.pkl')
    unit.save(f)
    back=ama.Unit.load(f,stim)
    assert back.name=='test / natural / phase pooled'
    assert np.allclose(back.multiscale_values()['k_peak'],unit.multiscale_values()['k_peak'])
    assert unit.config()['name']==ama.config_from_saved(f)['name']==back.name
    assert ama.Unit.from_config(unit.config(),stim).name==back.name


def test_embedding_methods_and_fourier_stimuli(tmp_path):
    unit,stim=trained()
    unit.train_multiscale(3,nKnot=9)
    x,s,ci,Y,_=ts.sine_frequency(seed=4)
    spatial=ama.Stim(x,s,ci,Y)
    for method,title in (('pacmap','PaCMAP'),('phate','PHATE')):
        fig=unit.plot_response_embedding(spatial,method,tmp_path/(method+'.png'),nMax=90)
        assert (tmp_path/(method+'.png')).exists() and ('response '+title) in fig._suptitle.get_text()
    with pytest.raises(Exception,match='method must be'):
        unit.plot_response_embedding(spatial,'umap')
    # the unit's training stimuli are in the fourier domain after finalize (the caller's stay spatial): the embedding
    # transforms them back
    assert unit.stim.bIsFourier and not stim.bIsFourier
    unit.plot_response_embedding(unit.stim,'tsne',nMax=60,perplexity=5)
    unit.save_figures(tmp_path/'all',spatial,methods={'tsne':{'perplexity':5},'pacmap':{},'phate':{}},nMax=90)
    assert all((tmp_path/('all_'+m+'.png')).exists() for m in ('filters','tsne','pacmap','phate','embeddings'))
    fig=unit.plot_response_embeddings(spatial,('tsne','phate'),tmp_path/'two.png',name='both',nMax=90,
                                      method_kw={'tsne':{'perplexity':5}})
    assert (tmp_path/'two.png').exists() and [a.get_title() for a in fig.axes][:2]==['t-SNE','PHATE']
    assert fig._suptitle.get_text().startswith('both')


def test_split_filters(tmp_path):
    x,st,ci,Y,info=ts.binocular_shift()
    stim=ama.Stim(x,st,ci,Y,nSplit=2)
    unit=ama.Unit(stim,ama.Nrn(),ama.Model('gss','mean',covShrink=0.5),ama.Objective('map'),
                  ama.Optimizer(nIterMax=5,bVerbose=False),name='disparity / gss')
    unit.train_new(2,bSplit=True)
    unit.save_figures(tmp_path/'s',stim,methods=('tsne',),nMax=60,perplexity=5)
    assert (tmp_path/'s_filters.png').exists() and (tmp_path/'s_tsne.png').exists()
    # each sub-filter is its own response dimension: 2 eyes x 2 filters
    feats,u,lat,cmap=unit._response_features(stim,60,0)
    assert u.shape[0]==4 and feats.shape[0]==lat.size
