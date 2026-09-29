"""
Held-out comparison of the likelihood models on burgelab's disparity and speed sets (the README table): 4 filters from
train_new(4), the best of several restarts by training cost, 30% of the stimuli held out (stratified, seed 0). Costs
are -log posterior at the correct level; rmse is of the posterior mean over all held-out stimuli.

The headline cost decodes noisy observations of the held-out stimuli (stage-2 noise sampled, averaged over
AMA_NOISE_SAMPLES draws): the expected cost of the modeled neurons. The cost of decoding mean (noise-free) responses,
the paper's approximation, is shown too, but does not compare likelihoods fairly: a category's mean responses are less
dispersed than the likelihood's covariance, which includes the noise. Training decodes mean responses (AMA_TRAIN='mean')
or noisy observations (AMA_TRAIN='noisy', AMA_TRAIN_SAMPLES draws per step).

Run from the repository root (AMAdataSpeed.mat must be downloaded, see tests/conftest.py):

    python examples/heldout_table.py

Environment variables (optional): AMA_RESTARTS (default 8), AMA_ITERS (iterations per restart, default 600),
AMA_NOISE_SAMPLES (default 16), AMA_TRAIN ('mean', the default, or 'noisy'), AMA_TRAIN_SAMPLES (default 8),
AMA_SCHEDULE ('joint', the default; '2+2': train_schedule([2, 2]) with 2 AMA_ITERS of refinement, which on the speed set
reaches a lower cost of mean responses but a higher one of noisy observations; see the README), AMA_DATASETS (default 'Disparity,Speed'), AMA_OUT (json file for the results, default
none).

Besides the table, each row reports how many restarts ended within 0.005 of the best training cost: when few do, the
optimum is hard to find and a row's cost depends on the number of restarts.
"""
import os
import sys
import json

# the GPU settings of tests/conftest.py that matter for float32 results: exact matrix products, no Triton GEMM fusions
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
if '--xla_gpu_enable_triton_gemm' not in os.environ.get('XLA_FLAGS',''):
    os.environ['XLA_FLAGS']=(os.environ.get('XLA_FLAGS','')+' --xla_gpu_enable_triton_gemm=false').strip()

import numpy as np
import jax
jax.config.update('jax_default_matmul_precision','highest')
from scipy.io import loadmat

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
import ama

nRestarts=int(os.environ.get('AMA_RESTARTS',8))
nIter=int(os.environ.get('AMA_ITERS',600))
nNoise=int(os.environ.get('AMA_NOISE_SAMPLES',16))
schedule=os.environ.get('AMA_SCHEDULE','joint')
trainMode=os.environ.get('AMA_TRAIN','mean')
nTrainNoise=int(os.environ.get('AMA_TRAIN_SAMPLES',8))
datasets=os.environ.get('AMA_DATASETS','Disparity,Speed').split(',')
outFile=os.environ.get('AMA_OUT')


def dataset(name):
    path=os.path.join(ROOT,'AMAdata'+name+'.mat')
    prm=loadmat(path)['AMA'][0,0]['paramRSP'][0,0]
    # stage-2 noise is sampled for the noisy held-out cost; training decodes mean responses, which it does not change
    nrn=dict(fano=prm['fano'].item(),var0=prm['var0'].item(),rmax=prm['rMax'].item(),bNoise_2=True)
    stim=ama.Stim.load(path)
    train,test=stim.train_test(testFraction=0.3,seed=0)
    return train,test,nrn,float(np.median(np.diff(np.asarray(stim.Y))))


def rows(spacing):
    # (label, training Model, decoding Model or None for the training one)
    return [("'gss'",ama.Model('gss','mean'),None),
            ("'gss', `bLeaveOneOut`",ama.Model('gss','mean',bLeaveOneOut=True),None),
            ("'gss', `covShrink=0.1`",ama.Model('gss','mean',covShrink=0.1),None),
            ("'gss', `ctgPoolWidth` one level",ama.Model('gss','mean',ctgPoolWidth=spacing),None),
            ("'student', `df=5`",ama.Model('student','mean',df=5.),None),
            ("'gss', decoded with 'student'",ama.Model('gss','mean'),ama.Model('student','mean',df=5.)),
            ("'mix', `nMix=2`",ama.Model('mix','mean',nMix=2),None),
            ("'full'",ama.Model('full','mean'),None),
            ("'full', `bLeaveOneOut`",ama.Model('full','mean',bLeaveOneOut=True),None)]


def responses(model,responseType,nSamples=1):
    # the same likelihood, decoding mean responses ('mean') or noisy observations averaged over nSamples draws ('basic')
    m=model.copy()
    m.responseType=responseType
    m.nNoiseSamples=nSamples
    return m


results={}
trained={}
for name in datasets:
    train,test,nrn,spacing=dataset(name)
    for label,model,decoder in rows(spacing):
        key=(name,model._key())
        if key not in trained:
            trainModel=responses(model,'basic',nTrainNoise) if trainMode=='noisy' else model
            unit=ama.Unit(train,ama.Nrn(**nrn),trainModel,ama.Objective('map'),
                          ama.Optimizer(nIterMax=nIter,lRate0=0.02,bVerbose=False))
            if schedule=='joint':
                unit.train_new(4,nRestarts=nRestarts)
            else:
                unit.train_schedule([2,2],nRestarts=nRestarts,refineIter=2*nIter)
            trained[key]=unit
        unit=trained[key]
        costs=np.asarray(unit.restart_costs if unit.restart_costs is not None else [float(unit.loss)])
        dec=decoder or model
        perf=unit.performance(estType='mean',stim=test,model=responses(dec,'mean'))
        res=dict(train=float(unit.loss) if decoder is None else unit.performance(model=responses(dec,'mean'))['cost'],
                 test=perf['cost'],rmse=float(np.ravel(perf['rmseAll'])[0]),
                 testNoisy=unit.evaluate(test,model=responses(dec,'basic',nNoise)),
                 nBest=int(np.sum(costs<costs.min()+0.005)),restartCosts=costs.tolist())
        results.setdefault(label,{})[name]=res
        print(f"{name:10s} {label:34s} noisy held-out {res['testNoisy']:.3f}  held-out (mean responses) {res['test']:.3f}  "
              f"rmse {res['rmse']:.2f}  train {res['train']:.3f}  restarts at the best {res['nBest']}/{len(costs)}",flush=True)

print()
print('| model | '+' | '.join(f'{d.lower()}: noisy held-out / held-out (mean responses) / rmse / train (restarts at best)'
                             for d in datasets)+' |')
print('|---|'+'---|'*len(datasets))
for label,r in results.items():
    print(f'| {label} | '+' | '.join(f"**{r[d]['testNoisy']:.3f}** / {r[d]['test']:.3f} / {r[d]['rmse']:.2f} / {r[d]['train']:.3f} ({r[d]['nBest']}/{len(r[d]['restartCosts'])})"
                                      for d in datasets)+' |')
if outFile:
    with open(outFile,'w') as fh:
        json.dump(results,fh,indent=1)
