"""
Learn AMA-Gauss filters for binocular disparity estimation on the burgelab training set (AMAdataDisparity.mat),
evaluate them on held-out stimuli, save them, and plot them.

Run from the repository root:

    python examples/quickstart.py

Environment variables (optional): AMA_EXAMPLE_ITERS (iterations per training call, default 300) and
AMA_EXAMPLE_OUT (directory for the saved unit, default the current directory).
"""
import os
import sys

import numpy as np

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
import ama

nIter=int(os.environ.get('AMA_EXAMPLE_ITERS',300))
outDir=os.environ.get('AMA_EXAMPLE_OUT','.')

# 7600 contrast-normalized binocular stimuli, 19 disparity levels (arcmin)
stim=ama.Stim.load(os.path.join(ROOT,'AMAdataDisparity.mat'))
train,test=stim.train_test(testFraction=0.2,seed=0)

unit=ama.Unit(train,
              ama.Nrn(fano=0.5,var0=0.23,rmax=5.7),       # response model of the burgelab reference run
              ama.Model('gss','mean'),                    # AMA-Gauss, likelihoods of mean responses
              ama.Objective('map'),                       # 0,1 cost: -log posterior at the correct level
              ama.Optimizer(nIterMax=nIter,lRate0=0.02,bVerbose=False))

unit.train_new(2)          # learn two filters jointly
unit.train_append(2)       # learn two more with the first two fixed
unit.train_recurse()       # refine all four

print(f'training cost {float(unit.loss):.3f}   held-out cost {unit.evaluate(test):.3f}')

perf=unit.performance(estType='mean',stim=test)          # MMSE estimates of held-out stimuli
print(' disparity    bias    rmse  pCorrect')
for Y,bias,rmse,pc in zip(perf['Y'],perf['bias'],perf['rmse'],perf['pCorrect']):
    print(f'{Y:10.3f} {bias:7.2f} {rmse:7.2f} {pc:9.2f}')

path=os.path.join(outDir,'disparity_unit.pkl')
unit.save(path)
restored=ama.Unit.load(path,train)
assert np.allclose(np.asarray(restored.out),np.asarray(unit.out))
print('saved to',path)

unit.plot_out()
ama.plt.show()
