# AMA.py
Accuracy Maximization Analysis (AMA) in python with JAX and Optax

AMA is a supervised learning algorithm that learns image-filters that best encode features for a given task, given the constraints of the visual system.
See [Burge and Jaini 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005281) (1).

NOTE: this is under-development. Most features listed below have been implemented, those that have not are mentioned in 'TODO'


## Ratinonale
Previous implementations of AMA are limited in that they struggle to handle 2D images due to performance bottlenecks.
This implementation aims to make AMA as fast as possible through JAX. JAX provides a huge increase in performance through:
- autodiff 
- jit compliation 
- gpu support 
- auto-vectorization 

This implementation is also able to gain some performance increases by optionally learning in the fourier domain with/without learning Quadrature/Hilbert filter pairs.
Learning filters in fourier domain allows one to take advantage of a sparsity constraint, and learning Quadrature/Hilbert pairs effectively learns two filters simultaneously with 1/4th the number of parameters.
These features also allow efficient computation of narrow-band normalization---a feature exclusive to this implementation.

For binocular images, additional performance can be gained by the "splitting" feature which splits the filter into two sub-filters.

## Installation
```
git clone https://github.com/portalgun/AMA.py
cd AMA.py
pip install .
```

## Example Data
Binocular data from [White & Burge 2024](https://www.biorxiv.org/content/10.1101/2024.02.27.582383v3.full)
- TODO full
- TODO flattened

Motion-in-depth data from [dheerrera1911/3D-motion_ideal_observer](https://github.com/dherrera1911/3D_motion_ideal_observer)
- [OSF repository](https://osf.io/w9mpe/)

Disparity and speed data from [burgelab/AMA](https://github.com/burgelab/AMA)
- [AMAdataDisparity.mat](https://github.com/burgelab/AMA/raw/refs/heads/master/AMAdataDisparity.mat)
- [AMAdataSpeed.mat](https://github.com/burgelab/AMA/raw/refs/heads/master/AMAdataSpeed.mat)

## Example Use
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
             eps=0.001               # normalization epsilon
             bRectify=False,         # whether to recify responses
             bNoise=False,           # whether to use noise
             rho=0,                  # noise correlation
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
                      projectionType=['l2_sphere',1], # constrained optimization (||f||=1)
                      lRate0=1e-1,                  # initial learning rate
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
## End-User Classes and Options
The main classes of objects are
- Stim
- Nrn
- Model
- Objective
- Unit
  For most use-cases, all but Unit should be treated as containers for the various parameters.
  Once they have been wrapped into a Unit, all learning and plotting routines are to be access through the Unit.

### Stim
```python
stim=Stim(
        x,                      # meshgrid for stimuli
        stimuli,                # data
        yCtgInd,                # label indeces
        Y,                      # labels
        bStimIsFourier=False,   # whether stimuli is in fourier domain
        nSplit                  # number of natural splits data has (see below)
        bStimIsSplit=False      # whether stimuli is already split
)
```

x - meshgird for stimulus, used for plotting TODO

stimuli - data

yCtgInd - label indeces

labels - labels

bStimIsFourier - whether stimuli is in fourier domain

nSplit - number of natural splits data has

bStimIsSplit - whether stimuli is already split

### Nrn
The neural response model
```python
nrn=ama.Nrn(
             fano=1.36,              # fano factor
             var0=0.23,              # base noise variance
             rmax=5.7                # max response,
             normalizeType='None',   # type of normalization
             eps=0.001               # normalization epsilon
             bRectify=False,         # whether to recify responses
             bNoise=False,           # whether to use noise
             rho=0,                  # noise correlation
)
```


fano - fano factor. `1.36` default

var0 - base noise variance. `0.23` default

rmax - max response. `5.7` default. NOTE default subject to change.

eps = normalization epsilon, i.e. minimum normalization factor to prevent division by 0. default `0.001`.

bRectify - whether to rectify responses. `False` default.

bNoise   - whether to rectify responses. `False` default.

rho - noise correlation
- None = no noise
- 0    = no noise correlation:w
- !=0  = noise correlation; correlates the noise of all response dimensions (filters, sub-filters, and real/imaginary components) equally.
  modelType='full' requires noise (rho not None).

normalizationType - type of response normalization to perform. A good explanation of these two types of normalization and how they differ can be found in [Burge & Iyer 2019](https://jov.arvojournals.org/article.aspx?articleid=2755285) (2).
- 'None'
- 'broadband' - Broadband normalization is simply the L2 norm of the stimulus contrast energy: <img src="https://latex.codecogs.com/svg.image?{\color{Gray}N_{brd}=||\mathbf&space;A_c||_2}" title="{\color{Gray} N_{brd}=||\mathbf A_c||_2}" />.

- 'narrowband' - Narrowband normalization is the dot product between stimulus contrast energy and filter contrast energy: <img src="https://latex.codecogs.com/svg.image?{\color{Gray}N_{nrw}=\mathbf&space;A_c^\intercal\mathbf&space;A_f&space;}" title="{\color{Gray} N_{nrw}=\mathbf A_c^\intercal\mathbf A_f}" />.
   Narrowband normalization is stimulus specific but feature independent (2).

whitenType - population whitening of the mean responses, applied before noise (intended for full AMA, whose likelihood assumes noise that is independent across response dimensions).
- 'None'
- 'gram' - whitens by the Gram matrix of the filters: the inner products of the real spatial filters behind each response dimension, including quadrature pairs and split sub-filters. Equivalent to orthonormalizing the filters; does not depend on the stimuli.
- 'response' - whitens by the covariance of the mean responses over the stimuli, which decorrelates the responses themselves. The two differ unless the stimuli are white.

whitenMethod - 'zca' (symmetric, `D^1/2 M^-1/2`, default) or 'chol' (Gram-Schmidt order, `D^1/2 L^-1`: earlier filters are not changed by later ones, which suits `train_append`).
Both keep each response dimension's variance (filter norm), so rmax, fano, and var0 keep their meaning.

whitenEps - ridge added to the matrix being whitened, relative to its mean diagonal. Default `1e-5`.

With 'response' whitening, the whitening matrix is estimated from the stimuli being evaluated (each batch, when training with batches).
To evaluate other stimuli with the training-set whitening, call `unit.freeze_whitening()` before `unit.split(...)`; training (or `unit.unfreeze_whitening()`) clears it.
Whitening can not be combined with `normalizeType='narrow'`.

### Model
The likelihood model: determines likelihoods of response data, given the parameters
```python
model=ama.Model(
                 modelType='gss'      # main algorithm
                 responseType='basic' # how responses are read
        )
```


modelType - algorithm for likelihoods are computed.
- 'gss'  = AMA-gauss assumes that class-conditional distributions are normally distributed (see [Jaini & Burge 2017](https://jov.arvojournals.org/article.aspx?articleid=2659576))
- 'full' = the original, full AMA model

responseType
- 'mean' = likelihoods based on mean responses
- 'basic' = likelihoods based on noisey responses

### Objective
The objective function for optimization
```python
objective=ama.Objective(errType='map',
                        lossType='mean',
)
```
  * [ ] The objective function is composed as `loss(error(estimation(posterior(likelihood))` when using a posterior or
`loss(error(estimation(likelihood)` otherwise

bPosterior - whether to estimate based on posterior instead of the likelihood.

estType - how to estimate 
- None = for map and mle errTypes TODO
- 'median' = TODO rename interp?
- 'mean'   = TODO rename
- 'mode'   = TODO rename

errType  -  how errors are computed/weighted
- 'mle' = maximimum likelihood  log(Likelihood at correct) TODO
- 'map' = maximum a posteriori -log(Posterior at correct) TODO
- 'l1'  = L1 deviation TODO
- 'l2'  = L2 deviation TODO

lossType - how to transform individual errors into a scalar loss value
- 'mean' = e.g. mean with `errType=l2` is MSE
- 'median'
- 'mode'
- 'cmean' = circular mean
    
### Optimizer
```python
optimizer=a.Optimizer(
                      optimizerType='adam',         # optax algorithm
                      projectionType=['l2_sphere',1], # constrained optimization (||f||=1)
                      lRate0=1e-1,                  # initial learning rate
                      nIterMax=200,                 # max number of iterations
                      f0_jxrand_fun=['ball',1],     # how to generate initial filters
                      batchSize=None,               # stimuli per iteration (None = full batch)
                      nStepsPerChunk=100,           # iterations compiled together; loss printed once per chunk
                      bVerbose=True                 # print the loss after each chunk
)  
```

lRate0 - initial learning default

optimizerType - optax optimzation algorithm.
ama.py uses optax for optimization.  The list of supported optimizers are listed in the [optax documentation](https://optax.readthedocs.io/en/latest/api/optimizers.html).


projectionType - optax projeciton type
Constrained optimization is a requirement for ama to work properly.
Otherwise filters magnitude will continue to grow infinitely.
With `l2_sphere`, each filter's gradient is projected onto the tangent plane of the sphere before the optimizer update (Burge & Jaini 2017, Eq 19);
without this, adaptive optimizers such as adam drift away from good filters.
AMA.py uses projections to perform constrained optimizations, and is able to use
any projection specified in [optax documentation](https://optax.readthedocs.io/en/latest/api/projections.html).
Note also that AMA.py also applies the specified constraint to each filter.
When learning in the spatial domain, the L2 constraint `l2_sphere` (unit-length filters, as in Burge & Jaini 2017) is the default and recommended.
When learning in the fourier domain, a sparsity constraint is  `l1_ball` or ` l1_sphere` is recommended.
`['l2_sphere',1]` specifies optax `projection_l2_sphere(_,1)`, where `1` is the scale paremter.
`['box',-1, 1]` specifies optax `projection_box(_,-1,1)`, where `-1` and `1` are the `lower` and `upper` parameters.

f0_jxrand_fun - jax.random function to generate initial filters
This simply specifies the distribution from which to pull initial filter values from.
These can be any of the random samplers specified in `jax.random` [jax documentation](https://jax.readthedocs.io/en/latest/jax.random.html#random-samplers).
These are specified by `f0_jx_rand_fun` in the same way as constraints above,
e.g. `['ball',1]` for `ball(_,1)` where `1` specified the dimension parameter `d`.

batchSize - stimuli per iteration for stochastic mini-batch learning (AMA-SGD, Burge & Jaini 2017). `None` (default) uses the full training set.
Each iteration draws a new random batch, stratified so every category keeps its share of the training set (i.e. the prior is preserved) with at least 2 stimuli per category.
Posteriors are computed against the batch, so the full AMA model costs O(batchSize²) per iteration instead of O(N²).
With batches, `optimizer.loss_hist` holds batch costs, which are noisy; use `unit.loss` for the cost on the whole training set.

nStepsPerChunk - number of iterations compiled into a single `lax.scan`. Larger chunks reduce per-iteration overhead; the loss is reported once per chunk.

bVerbose - print the loss after each chunk.

### Unit
Container for all the various parts and interface for the end-user.
```python
unit=ama.Unit(stim,nrn,model,objective)
```
Constructor

```python
unit.split(stimInd)
```

#### Learning methods 
```python
unit.train_new(n,...)
```
Discard current filters (if any) and learn `n` new filters.

``` python
unit.train_append(n,...)
```
Learn `n` new filters while fixing the others in place. 

``` python
unit.train_recurse(...)
```
Continue iterating over current filters.

#### Learning parameters
All learning methods optionally contain the following parameters:

fourierType
- 0 = learning in spatio/temporal domain
- 1 = learning in fourier domain: one real filter per learned filter
- 2 = learns quadradrature pair in fourier domain: each learned filter is a real filter and its hilbert pair, which respond as two neurons

In the fourier domain each filter is parameterized by half of the frequencies (the half-space whose first nonzero frequency coordinate is positive;
in 1D the positive frequencies, in 2D a half-plane). The other half is the complex conjugate, so the implied spatial filters are real.
DC and nyquist-only frequencies are not learned; stimuli should be contrast normalized (zero mean) anyway.
Stimuli are transformed with an orthonormal fourier transform and responses are scaled so that a unit-norm fourier filter (`['l2_sphere',1]`)
implies unit-norm real spatial filters. fourierType=1 gives exactly the same responses as the equivalent spatial filter, and with fourierType=2
both members of the pair have unit norm.

dtype - datatype of filters to be learned
- float32   - default and recommended for non-fourier-domain learning
- float64
- float128
- complex64 - default and only sane choice for fourier-domain learning 

optimizer = ama.optimizer instance.

bSplit - whether to split filters into sub-filters
Stereo/binocular data a single natural split (`nSplit=1`) between each stereo-half.
When learning filters for stereo images, one can treat each stereo-half as its
own sub-filter. This can be a more efficient way to learn filters, as it
effectively allows sub-filters to be 'reused' in context with other filters.
In order to use this splitting feature, the number of splits must be specified
in `Stim` by `nSplit` and `bSplit=True` set in `Unit.`

#### properties
- out
- last

- loss
- response
- error

#### Plotting
```python
unit.plot_out()
```

```python
unit.plot_joint_responses()
```

```python
unit.plot_joint_responses_tsne() 
```
## Notes on hacking
In order to aid with autograd, learning routines do not contain any 'if' statements.
Instead, objective functions are composed dynamically before execution.
The _TypeFunc descriptor parses and fetches functions as properties are set using meta programming.
Together, these allow for fast optimization, parameter flexibility, and straightforward extensibility, but at the cost of readable code. 

## Other AMA implementations
[burgelab/AMA](https://github.com/burgelab/AMA) - the original matlab implementation

[dherrera/amatorch](https://github.com/dherrera1911/amatorch) - written in python with pytorch, features learning based on noise-covariance

[portalgun/AMA.DNW.mat](https://github.com/portalgun?tab=repositories) - matlab prototype for ama.py

## TODO
- test split
- test fourier learning
- test joint responses
- test t-sne response plotting 
    no split no fourier
    split
    fourier
    split and fouier
- test marginal
    no split no fourier
    split
    fourier
    split and fouier
- test est types (mean, median,mode, cmean)
- test loss mode

- bias term
- split negatives
- softmax activation

- activation functions/saturation
  ramp

Redundant
- rho = None
- responseType = mean
- bNoise = False

- normalization
- finish full ama
- binocular data
- normalization indeces 
- phase parameter for quadrature pair learning
- fmincon-like options
- unit clear
- unit save

- test setup
- test cases
- move to src

jupyter notebooks
- different data

- f0 to last?
- better rmax default? ~60?
- different eps value?
- merge Objective and Model?
- other parameter based learning?


V2
- learn normalization indeces?
- responseType n - mean over n, gen n iterations of noise
- weights and combination learning (no likelihoods?)
- specified covaraince/correlation
- layers


# Works cited
(1) Burge J, Jaini P (2017). Accuracy Maximization Analysis for sensory-perceptual tasks: Computational improvements, filter robustness, and coding advantages for scaled additive noise.  PLoS Computational Biology, 13(2): e1005281. doi:10.1371/journal.pcbi.1005281

(2) Iyer AV, Burge J (2019). The statistics of how natural images drive the responses of neurons. Journal of Vision, 19(13): 4, 1-25, doi: https://doi.org/10.1167/19.13.4

(3) DN White, J Burge. How distinct sources of nuisance variability in natural images and scenes limit human stereopsis. Preprint. (582383). https://doi.org/10.1101/2024.02.27.582383

(4) Jaini P, Burge J (2017). Linking normative models of natural tasks with descriptive models of neural response. Journal of Vision, 17(12):16, 1-26
    
