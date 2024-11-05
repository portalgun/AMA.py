# AMA.py
Accuracy Maximization Analysis (AMA) in python with JAX

AMA is a task-oriented dimensionality reduction technique that is image computable and physiologically plausible.
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005281

```python
import ama

nrn=ama.Nrn( fano=1.36,              # fano factor
             sigma0=0.23,            # base noise variance
             rmax=5.7                # max response,
             normalizeType='None',   # type of normalization
             bRectify=False,         # whether to recify responses
             bNoise=False,           # whether to use noise
             rho=0,                  # noise correlation
             eps=0.001               # normalization eps
)

model=ama.Model(modelType='gss')
objective=ama.Objective(errType='map',
                        lossType='mean'
)

stim=ama.Stim.load('./AMAdataDisparity.mat')

trn=ama.Trn(nrn,model,objective,stim)
trnIter=trn.finalize(3,              # number of filters
        ind_lrn=(0,1,2),             # index of filters to learn
        fourierType=0,
)
optimizer=a.Optimizer(optimizerType='adam',
                      projectionType=['l2_ball',1], # constrained optimization
                      lRate0=1e-1,                  # initial learn rate
                      nIterMax=200,                 # max number of iterations
                      f0_jxrand_fun=['ball',1])     # how to generate initial filters

trnIter.minimize(optimizer)
```

Normalization types:
    None
    broad
    Narrow

fourierType
    0 - spatio-temporal
    1 - fourier
    2 - fourier + analytic fitlers


