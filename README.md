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

stim=Stim(
        x,                      # meshgrid for stimuli
        stimuli,                # data
        yCtgInd,                # label indeces
        Y,                      # labels
        bStimIsFourier=False,   # whether stimuli is in fourier domain
        nSplit                  # number of natural splits data has (see below)
        bStimIsSplit=False      # whether stimuli is already split
)

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
          bSplit,                 # whether to split filters (and stimuil into multiple parts
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

```
## Models
Model specifies how likelihoods are estimated
`full`is the original, full AMA model
`gss` is AMA-gauss assumes that class-conditional distributions are normally distributed (see https://jov.arvojournals.org/article.aspx?articleid=2659576)

## normalization


## Constraints
Constrained optimization is a requirement for ama to work properly.
Otherwise filters magnitude will continue to grow infinitely.
AMA.py uses projections to perform constrained otpimizations, and is able to use
any projection specified in optax (see https://optax.readthedocs.io/en/latest/api/projections.html).
Note also that AMA.py also applies the specified constraint to each filter.

When learning in the spatial domain, L2 constraints `L1_ball` or ` L1_sphere` is recommended.

When learning in the fourier domain, a sparsity constraint is  `l1_ball` or ` l1_sphere` is recommended.

### Examples
`['l2_ball',1]` specifies optax `projection_l2_ball(_,1)`, where `1` is the scale paremter.

`['box',-1, 1]` specifies optax `projection_box(_,-1,1)`, where `-1` and `1` are the `lower` and `upper` parameters.

## fourierType
Stimuli can be loaded in the fourier domain and learned in spatio/temororal domain and vice versa.
Whether or not data is learned in the fourier domain is specified by`fourierType` in the `Unit` class.

## Splitting
Stereo/binocular data a single natural split (`nSplit=1`) between each stereo-half.
Thus, when learning filters for stereo images, one can treat each stereo-half as its
own sub-filter. This can be a more efficient way to learn filters as it
effectively allows sub-filters to be 'reused' in context with other filters.

In order to use this splitting feature, the number of splitsnSplit must be specified
in `Stim` by `nSplit` and `bSplit=True` set in `Unit.`

## f0_jx_rand_fun
This simply specifies the distribution from which to pull initial filter values from.
These can be any of the random samplers specified in `jax.random` (see https://jax.readthedocs.io/en/latest/jax.random.html#random-samplers)
These are specified the same way as constraints above
e.g. `['ball',1]` for `ball(_,1)` where `1` specified the dimension parameter `d`.

## optimizerType
TODO

## Options
rho
- None = no noise
- 0    = no noise correlation:w
- !=0  = noise correlation

responsType
- 'basic' = likelihoods based on mean responses
- 'basic' = likelihoods based on noisey responses

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
- dictionary options for projections
- full ama
- fourier learning
- t-sne
- option for splitting binocular filters
    - nSplit in Stim
- option for splitting quadrature pairs
- phase parameter for quadrature pair learning
- fmincon like options
