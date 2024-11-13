# AMA.py
Accuracy Maximization Analysis (AMA) in python with JAX and Optax

AMA is a supervised learning algorithm that learns image-filters that best encode features for a given task, given the constraints of the visual system.
See [Burge and Jaini 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005281) (1).


JAX support provides:
- autodiff
- jit compliation
- gpu support
- auto-vectorization (no for loops)

## Use
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
## Installation

## Models
Model specifies how likelihoods are computed.
`full`is the original, full AMA model
`gss` is AMA-gauss assumes that class-conditional distributions are normally distributed (see https://jov.arvojournals.org/article.aspx?articleid=2659576)

## Optimizaton algorithm 
ama.py uses optax for optimization.  The list of supported optimizers are listed in the optax documentation (see https://optax.readthedocs.io/en/latest/api/optimizers.html)

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

## Initial filter vaues
This simply specifies the distribution from which to pull initial filter values from.
These can be any of the random samplers specified in `jax.random` (see https://jax.readthedocs.io/en/latest/jax.random.html#random-samplers)
  * [ ] These are specified by `f0_jx_rand_fun` in the same way as constraints above,
e.g. `['ball',1]` for `ball(_,1)` where `1` specified the dimension parameter `d`.

## Fourier-domain learning 
Stimuli can be loaded in the fourier domain and learned in spatio/temororal domain and vice versa.
Whether or not data is learned in the fourier domain is specified by`fourierType` in the `Unit` class.

## Splitting
Stereo/binocular data a single natural split (`nSplit=1`) between each stereo-half.
When learning filters for stereo images, one can treat each stereo-half as its
own sub-filter. This can be a more efficient way to learn filters, as it
effectively allows sub-filters to be 'reused' in context with other filters.

In order to use this splitting feature, the number of splits must be specified
in `Stim` by `nSplit` and `bSplit=True` set in `Unit.` 

## Response Normalization
ama.py currently supports two forms of neural response normalization---broadband <img src="https://latex.codecogs.com/svg.image?N_{brd}" title="N_{brd}" />
narrowband <img src="https://latex.codecogs.com/svg.image?N_{brd}" title="N_{nrw}" />.
A good explanation of these two types of normalization and how they differ can be found in [Burge & Iyer 2019(2)]
(https://jov.arvojournals.org/article.aspx?articleid=2755285) (2).

### Broadband
Broadband normalization is simply the L2 norm of the stimulus contrast energy:

<img src="https://latex.codecogs.com/svg.image?N_{brd}=||\mathbf&space;A_c||_2" title="N_{brd}=||\mathbf A_c||_2" />

Broadband normalization is stimulus specific but feature independent (2).

### Narrowband
Narrowband normalization is the dot product between stimulus contrast energy and filter contrast energy:

<img src="https://latex.codecogs.com/svg.image?N_{nrw}=\mathbf&space;A_c^\intercal\mathbf&space;A_f&space;" title="N_{nrw}=\mathbf A_c^\intercal\mathbf A_f" />

Narrowband normalization is stimulus specific but feature independent (2).

## Options
rho
- None = no noise
- 0    = no noise correlation:w
- !=0  = noise correlation

responseType
- 'mean' = likelihoods based on mean responses
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

## Data
Binocular data from [White & Burge 2024](https://www.biorxiv.org/content/10.1101/2024.02.27.582383v3.full)
- TODO full
- TODO flattened
Motion-in-depth data from [dheerrera1911/3D-motion_ideal_observer](https://github.com/dherrera1911/3D_motion_ideal_observer)
- [OSF repository](https://osf.io/w9mpe/) 
Disparity and speed data from (burgelab/AMA)[https://github.com/burgelab/AMA]
- [AMAdataDisparity.mat](https://github.com/burgelab/AMA/raw/refs/heads/master/AMAdataDisparity.mat)
- [AMAdataSpeed.mat](https://github.com/burgelab/AMA/raw/refs/heads/master/AMAdataSpeed.mat)

## Other ama implementations
[burgelab/AMA](https://github.com/burgelab/AMA) - the original matlab implementation
[dherrera/amatorch](https://github.com/dherrera1911/amatorch) - written in python with pytorch, features learning based on noise-covariance

## TODO
- finish split
- full ama
- fourier learning
- t-sne
- option for splitting quadrature pairs
- phase parameter for quadrature pair learning
- fmincon like options

# Citations
(1) Burge J, Jaini P (2017). Accuracy Maximization Analysis for sensory-perceptual tasks: Computational improvements, filter robustness, and coding advantages for scaled additive noise.  PLoS Computational

   Biology, 13(2): e1005281. doi:10.1371/journal.pcbi.1005281
(2) Iyer AV, Burge J (2019). The statistics of how natural images drive the responses of neurons. Journal of Vision, 19(13): 4, 1-25, doi: https://doi.org/10.1167/19.13.4
(3) DN White, J Burge. How distinct sources of nuisance variability in natural images and scenes limit human stereopsis. Preprint. (582383). https://doi.org/10.1101/2024.02.27.582383
