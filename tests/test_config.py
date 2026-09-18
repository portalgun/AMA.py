"""
yaml configuration: Unit.config / save_config / from_config, config_from_saved, and the training log.
"""
import numpy as np
import pytest
import yaml

import ama
import stimuli as ts


def unit_for(**opt):
    x,s,ci,Y,_=ts.sine_frequency()
    stim=ama.Stim(x,s,ci,Y)
    o=dict(nIterMax=30,lRate0=0.05,bVerbose=False)
    o.update(opt)
    return ama.Unit(stim,ama.Nrn(normalizeType='phase',readoutType='resultant_only'),
                    ama.Model('circ','mean',covShrink=0.5),ama.Objective('map'),ama.Optimizer(**o),seed=3),stim


def test_config_roundtrip_and_replay(tmp_path):
    unit,stim=unit_for()
    unit.train_multiscale(3,nKnot=9,nMothers=2,edgeTaper=0.2,stimVal=stim)
    unit.train_recurse(stimVal=stim)
    f=tmp_path/'u.yaml'
    unit.save_config(f)
    cfg=ama.load_config(f)
    assert cfg==yaml.safe_load(yaml.safe_dump(unit.config()))
    assert [t['method'] for t in cfg['train']]==['train_multiscale','train_recurse']
    assert cfg['train'][0]['args']['nMothers']==2 and cfg['train'][0]['args']['kTop']==0.25   # defaults recorded
    assert cfg['filters']['n']==6 and cfg['filters']['bPoolWeights']
    # rebuilt components equal the originals (the jit keys), and replaying training reproduces the filters
    rebuilt=ama.Unit.from_config(f,stim)
    assert rebuilt.nrn._key()[:16]==ama.Nrn(**cfg['nrn'])._key()[:16]
    assert rebuilt.model==unit.model and rebuilt.objective==unit.objective
    replay=ama.Unit.from_config(str(f),stim,bTrain=True,stimVal=stim)
    assert np.allclose(np.asarray(replay.out),np.asarray(unit.out),atol=1e-6)
    assert [t['method'] for t in replay.train_log]==['train_multiscale','train_recurse']


def test_saved_unit_keeps_log_and_config(tmp_path):
    unit,stim=unit_for()
    unit.train_new(2,fourierType=2)
    fname=str(tmp_path/'u.pkl')
    unit.save(fname)
    back=ama.Unit.load(fname,stim)
    assert back.train_log==unit.train_log
    a,b=ama.config_from_saved(fname),unit.config()
    assert a['ama_source_sha256']==b['ama_source_sha256']==ama.source_sha256() and len(a['ama_source_sha256'])==64
    for key in ('nrn','model','objective','optimizer','filters','train','stim','seed'):
        assert a[key]==b[key], key


def test_nested_training_is_logged_once():
    unit,stim=unit_for(nIterMax=5)
    unit.train_parametric(2,family='loggabor')
    assert [t['method'] for t in unit.train_log]==['train_parametric']


def test_from_config_rejects_newer_version_and_unknown_method():
    unit,stim=unit_for()
    cfg=unit.config()
    cfg['ama_config_version']=ama.CONFIG_VERSION+1
    with pytest.raises(Exception,match='newer'):
        ama.Unit.from_config(cfg,stim)
    cfg=unit.config()
    cfg['train']=[{'method':'fit_everything','args':{}}]
    with pytest.raises(Exception,match='unknown training method'):
        ama.Unit.from_config(cfg,stim,bTrain=True)
