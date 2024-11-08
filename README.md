# AMA.py
Accuracy Maximization Analysis (AMA) in python with JAX and Optax

AMA is a task-oriented dimensionality reduction technique that is image computable and physiologically plausible.
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005281

JAX support provides:
- autodiff
- jit compliation
- gpu support
- auto-vectorization (no for loops)

```python
import ama
import numpy as np

# set basic neural response properties
nrn=ama.Nrn(
             fano=1.36,              # fano factor
             var0=0.23,              # base noise variance
             rmax=5.7                # max response,
             normalizeType='None',   # type of normalization
             bRectify=False,         # whether to recify responses
             bNoise=False,           # whether to use noise
             rho=0,                  # noise correlation
             eps=0.001               # normalization eps
)

# set algorithm that determines likelihoods from responses
model=ama.Model(
                 modelType='gss'      # main algorithm
                 responseType='basic' # how responses are read
        )

# set ojbective for learning
objective=ama.Objective(errType='map',
                        lossType='mean'
)

# wrap these into a unit
unit=ama.Unit(stim,nrn,model,objective)

# set optimizer properties for filter learning
optimizer=a.Optimizer(
                      optimizerType='adam',         # optax algorithm
                      projectionType=['l2_ball',1], # constrained optimization
                      lRate0=1e-1,                  # initial learn rate
                      nIterMax=200,                 # max number of iterations
                      f0_jxrand_fun=['ball',1])     # how to generate initial filters


# learn 3 new filters
unit.train_new(
          3,                      # number of filters
          fourierType=0,          # whether to learn filters in fourier domain
          dtype=float32,          # data type of filters and stimuli
          optimizer=optimzer
)


# recurse upon what already has been learned
unit.train_recurse()

# fix previous filters and learn 3 more
unit.train_append(3)

# split unit for batch learning
unit1=unit.split(stimInd=np.arange(100))
unit2=unit.split(stimInd=np.arange(100,201))

# plot learned filters
unit.plot_out()

# plot joint responses
unit.plot_spinner()

trnIter.minimize(optimizer)
```

## Options
rho
- None = no noise
- 0    = no noise correlation:w
- !=0  = noise correlation

responsType
- 'basic' -

Normalization types:
- None
- 'broadband'
- 'narrowband'

fourierType
- 0 = learning in spatio/temporal domain
- 1 = learning in fourier domain
- 2 = learns quadradrature pair in fourier domain

modelTypes
- 'gss'
- 'full' (coming soon)

## TODO
- full ama
- fourier learning
- t-sne
- option for splitting binocular filters
    - nSplit in Stim
- option for splitting quadrature pairs
- phase parameter for quadrature pair learning
- fmincon like options
