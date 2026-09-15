# AMA.py
Accuracy Maximization Analysis (AMA) in python with JAX and Optax

AMA is a supervised learning algorithm that learns the stimulus filters (receptive fields) that best encode the features needed to estimate a latent variable, given a model of noisy neural responses.
See [Burge and Jaini 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005281) (1), and [Jaini & Burge 2017](https://jov.arvojournals.org/article.aspx?articleid=2659576) (4) for AMA-Gauss.

NOTE: this is under development. Planned features are listed under 'TODO'.


## Rationale
Previous implementations of AMA are limited in that they struggle to handle 2D images due to performance bottlenecks.
This implementation aims to make AMA as fast as possible through JAX. JAX provides a huge increase in performance through:
- autodiff
- jit compilation
- gpu support
- auto-vectorization

Filters can also be learned in the fourier domain, with or without learning quadrature (hilbert) filter pairs.
Learning in the fourier domain allows a sparsity constraint, and a quadrature pair is learned with the parameters of a single filter.
The fourier domain also allows efficient narrowband normalization.

For binocular images, filters can be split into sub-filters (one per eye).

## Installation
```
git clone https://github.com/portalgun/AMA.py
git clone https://github.com/portalgun/Filter.py
cd AMA.py
pip install jax optax numpy scipy matplotlib scikit-learn pytest
export PYTHONPATH=$PWD/../Filter.py        # Filter.py is not pip-installable yet
```
For NVIDIA GPUs, install `jax[cuda12]` instead of `jax`.

Run the tests from the repository root with `python -m pytest tests` (about a minute on a CPU).

## Example Data
Binocular data from [White & Burge 2024](https://www.biorxiv.org/content/10.1101/2024.02.27.582383v3.full)
- TODO full
- TODO flattened

Motion-in-depth data from [dheerrera1911/3D-motion_ideal_observer](https://github.com/dherrera1911/3D_motion_ideal_observer)
- [OSF repository](https://osf.io/w9mpe/)

Disparity and speed data from [burgelab/AMA](https://github.com/burgelab/AMA)
- [AMAdataDisparity.mat](https://github.com/burgelab/AMA/raw/refs/heads/master/AMAdataDisparity.mat) (included in this repository)
- [AMAdataSpeed.mat](https://github.com/burgelab/AMA/raw/refs/heads/master/AMAdataSpeed.mat)

## Example Use
The runnable version of this example is [examples/quickstart.py](examples/quickstart.py).
```python
import ama

stim=ama.Stim.load('AMAdataDisparity.mat')               # 7600 stimuli, 19 disparity levels
train,test=stim.train_test(testFraction=0.2,seed=0)     # stratified by category

unit=ama.Unit(train,
              ama.Nrn(fano=0.5,var0=0.23,rmax=5.7),     # response model
              ama.Model('gss','mean'),                  # AMA-Gauss likelihood
              ama.Objective('map'),                     # 0,1 cost
              ama.Optimizer(nIterMax=300,lRate0=0.02))  # optax adam on unit-norm filters

unit.train_new(2)          # learn two filters
unit.train_append(2)       # learn two more with the first two fixed
unit.train_recurse()       # refine all four

unit.loss                                # cost on the training stimuli
unit.evaluate(test)                      # cost on held-out stimuli, decoded with the training set
perf=unit.performance(estType='mean',stim=test)   # bias, sd, rmse, pCorrect, confusion per level

unit.save('disparity_unit.pkl')
unit=ama.Unit.load('disparity_unit.pkl',train)

unit.plot_out()
```

## End-User Classes and Options
- `Stim` - labeled training stimuli
- `Nrn` - response model: filter responses, normalization, whitening, noise
- `Model` - likelihood of the responses given each level of the latent variable
- `Objective` - cost to minimize
- `Optimizer` - filter learning
- `Unit` - wraps the above; all learning, evaluation, and plotting is done through the Unit

The cost is composed as `loss(error(estimate(posterior(likelihood))))`.
Stimuli are grouped by category and padded to the largest category; `Stim.weights` marks the valid stimuli.
Arrays with a stimulus axis (e.g. `unit.estimates()`) are grouped the same way, `[ nStim_Ctg x nCtg ]`.

### Stim
```python
stim=ama.Stim(x,                        # Filter.X meshgrid of the stimulus (e.g. Filter.X(ndim=1,n=64,totS=1))
              stimuli,                  # [ *dims x nStim ]
              yCtgInd,                  # [ nStim ] category label of each stimulus (any integer coding)
              Y,                        # [ nCtg ] latent variable value of each category, in sorted label order
              bStimIsFourier=False,     # whether stimuli are in the fourier domain
              nSplit=0,                 # number of parts for split filters (e.g. 2 for the two eyes)
              bStimIsSplit=False,       # whether stimuli are already split
              bContrastNormalize=False) # contrast normalize the stimuli
```
AMA assumes contrast-normalized stimuli (zero mean, unit norm); `Stim` warns otherwise. Use `bContrastNormalize=True` or `ama.contrast_normalize(stimuli)`.
For split stimuli, the parts are contiguous blocks of the first stimulus dimension (e.g. left eye then right eye).

- `ama.Stim.load('file.mat')` - burgelab `.mat` files (`s`, `ctgInd`, `X`)
- `stim.train_test(testFraction=0.2,seed=0)` - `(train, test)`, stratified by category
- `stim.folds(k=5,seed=0)` - `[(train, test), ...]` for cross-validation, stratified by category
- `stim.plot(index=(0,0),bFourier=False)` - plot stimulus `index=(stimulus within category, category)`

### Nrn
The response model (Burge & Jaini 2017, Eq 1): mean response `r = rmax * f^T s`, noise variance `fano*|r| + var0`.
```python
nrn=ama.Nrn(fano=1.36,             # fano factor
            var0=0.23,             # base noise variance
            rmax=5.7,              # response scale
            normalizeType='None',  # response normalization
            eps=0.001,             # normalization constant
            activationType='None', # response nonlinearity
            bNoise_1=False,        # sample noise before normalization
            bNoise_2=False,        # sample noise after normalization
            rho=0,                 # noise correlation
            averageType='full',    # how noise samples are combined
            nSamples=1,            # noise samples per response
            whitenType='None',     # population whitening before noise
            whitenMethod='zca',
            whitenEps=1e-5)
```

normalizeType - response normalization. See [Iyer & Burge 2019](https://jov.arvojournals.org/article.aspx?articleid=2755285) (2).
- 'None'
- 'broad' - broadband: divide by the stimulus contrast energy, `||s||`
- 'narrow' - narrowband: divide by the dot product of the stimulus and filter amplitude spectra, `A_s^T A_f`. Stimulus specific but feature independent (2). Requires fourier-domain learning.
- 'gen' - divide by `eps + sum(|r|)` over the population

activationType - 'None', 'relu', 'softplus', 'abs', 'logistic', 'swish', 'swish2', 'tanh', 'gauss', 'igauss' (applied to real and imaginary parts separately)

Noise - the likelihood always uses noise of variance `fano*|R| + var0` on the final responses, also when no noise is sampled (`responseType='mean'`, the mean-response approximation of Burge & Jaini 2017).
Noise sampled before normalization (`bNoise_1`) is carried through the normalization: exactly for 'broad' and 'narrow', to first order for 'gen'.

rho - noise correlation
- None = no noise (only with `modelType='gss'`)
- 0 = independent noise
- otherwise, correlates the noise of all response dimensions (filters, sub-filters, and real/imaginary components) equally

averageType - with `nSamples > 1`: 'full' (a single sample), 'mean', 'log_mean', or 'median' of the samples

whitenType - population whitening of the mean responses, applied before noise (intended for full AMA, whose likelihood assumes noise that is independent across response dimensions).
- 'None'
- 'gram' - whitens by the Gram matrix of the filters: the inner products of the real spatial filters behind each response dimension, including quadrature pairs and split sub-filters. Equivalent to orthonormalizing the filters; does not depend on the stimuli.
- 'response' - whitens by the covariance of the mean responses over the stimuli, which decorrelates the responses themselves. The two differ unless the stimuli are white.

whitenMethod - 'zca' (symmetric, `D^1/2 M^-1/2`, default) or 'chol' (Gram-Schmidt order, `D^1/2 L^-1`: earlier filters are not changed by later ones, which suits `train_append`).
Both keep each response dimension's variance (filter norm), so rmax, fano, and var0 keep their meaning.

whitenEps - ridge added to the matrix being whitened, relative to its mean diagonal.

With 'response' whitening, the whitening matrix is estimated from the stimuli being evaluated (each batch, when training with batches); `unit.evaluate` uses the training stimuli.
Whitening can not be combined with `normalizeType='narrow'`.

### Model
The likelihood of the responses given each level of the latent variable
```python
model=ama.Model(modelType='gss',       # likelihood model
                responseType='basic',  # which responses are decoded
                bLeaveOneOut=False)    # full AMA: leave each decoded stimulus out of its own category
```

modelType
- 'gss' - AMA-Gauss: the responses to each category are gaussian, with the mean and covariance of the category's mean responses plus the noise covariance (Jaini & Burge 2017)
- 'full' - the original AMA: each category's likelihood is the average over its stimuli of the noisy response distribution to each stimulus (Burge & Jaini 2017, Eq 5)

responseType
- 'mean' - decode mean responses (noise enters through the likelihood)
- 'basic' - decode noisy responses

bLeaveOneOut - with 'full', each decoded stimulus otherwise matches its own mean response, which makes the cost optimistic when noise is low or categories (or batches) are small.
With `bLeaveOneOut=True`, the posterior is Eq 5 with the decoded stimulus removed from the training set. Not available with `errType='mle'`.

### Objective
```python
objective=ama.Objective(errType='map',   # error
                        bPosterior=None, # decode the posterior (default for all but 'mle')
                        estType=None,    # estimator for 'l1' and 'l2'
                        lossType='mean') # how errors are combined
```

errType
- 'map' - `-log posterior` at the correct level: the 0,1 cost (Burge & Jaini 2017, Eq 9)
- 'mle' - `-log likelihood` at the correct level
- 'l2' (or 2) - squared error of the estimate; estType defaults to 'mean' (MMSE)
- 'l1' (or 1) - absolute error of the estimate; estType defaults to 'median'

estType - 'mean', 'median', 'mode' (MAP; no gradient), or 'cmean' (circular mean, Y in radians)

lossType - 'mean' or 'median' over the valid stimuli

### Optimizer
```python
optimizer=ama.Optimizer(optimizerType='adam',            # optax algorithm
                        projectionType=['l2_sphere',1],  # constraint on each filter
                        lRate0=1e-1,                     # learning rate (or an optax schedule)
                        nIterMax=1000,                   # iterations per training call
                        f0_jxrand_fun=['ball',1],        # distribution of initial filters
                        batchSize=None,                  # stimuli per iteration (None = all)
                        nStepsPerChunk=100,              # iterations compiled together
                        bVerbose=True,                   # print the loss after each chunk
                        nBatchMinCtg=2)                  # minimum stimuli per category in a batch
```

optimizerType - any optimizer in the [optax documentation](https://optax.readthedocs.io/en/latest/api/optimizers.html).

projectionType - any optax [projection](https://optax.readthedocs.io/en/latest/api/projections.html), applied to each filter.
Constrained optimization is a requirement: otherwise filter magnitude grows without bound.
`['l2_sphere',1]` (unit-norm filters, as in Burge & Jaini 2017) is the default and recommended; `['box',-1,1]` specifies `projection_box(_,-1,1)`.
With `l2_sphere`, each filter's gradient is projected onto the tangent plane of the sphere before the update (Burge & Jaini 2017, Eq 19); without this, adaptive optimizers such as adam drift away from good filters.

f0_jxrand_fun - a [jax.random sampler](https://jax.readthedocs.io/en/latest/jax.random.html#random-samplers) and its parameters, e.g. `['ball',1]` for `ball(_,1)`.

batchSize - stimuli per iteration for stochastic mini-batch learning (AMA-SGD, Burge & Jaini 2017).
Each iteration draws a new random batch, stratified so that every category keeps its share of the training set (the prior), with at least `nBatchMinCtg` stimuli per category.
Posteriors are computed against the batch, so full AMA costs O(batchSize²) per iteration instead of O(N²).
AMA-Gauss needs more stimuli per category than response dimensions; a warning is given otherwise.
With batches, `optimizer.loss_hist` holds batch costs, which are noisy; use `unit.loss` for the cost on all training stimuli.

nStepsPerChunk - iterations compiled into a single `lax.scan`.

### Unit
```python
unit=ama.Unit(stim,nrn,model,objective,optimizer=None,seed=None)
```

#### Learning
```python
unit.train_new(n,fourierType=None,bSplit=None,stimInd=None,dtype=None,optimizer=None,nRestarts=1)
```
Discard current filters (if any) and learn `n` new filters. With `nRestarts=k`, learn from `k` random initial filter sets and keep the one with the lowest cost (costs in `unit.restart_costs`).

```python
unit.train_append(n,...,nRestarts=1)
```
Learn `n` new filters while keeping the others fixed.

```python
unit.train_recurse(ind_rec=None,...)
```
Continue learning the current filters (or only those in `ind_rec`).

fourierType
- 0 = spatial (or spatio-temporal) domain
- 1 = fourier domain: one real filter per learned filter
- 2 = fourier domain quadrature pairs: each learned filter is a real filter and its hilbert pair, which respond as two neurons

In the fourier domain each filter is parameterized by half of the frequencies (the half-space whose first nonzero frequency coordinate is positive: the positive frequencies in 1D, a half-plane in 2D).
The other half is the complex conjugate, so the implied spatial filters are real (`unit.filter.implied_spatial()`).
DC and nyquist-only frequencies are not learned.
Stimuli are transformed with an orthonormal fourier transform and responses are scaled so that unit-norm fourier filters imply unit-norm real spatial filters:
fourierType=1 gives exactly the responses of the equivalent spatial filter, and with fourierType=2 both members of the pair have unit norm.

dtype - float32 (default) or float64; complex64 (default) or complex128 in the fourier domain

bSplit - split filters into one sub-filter per part of the stimulus (set `nSplit` in `Stim`, e.g. 2 for the two eyes). Each sub-filter responds as its own neuron.

#### Evaluation
- `unit.loss` - cost on the training stimuli
- `unit.evaluate(stim)` - cost on other stimuli, decoded with the training set: category response distributions (AMA-Gauss), reference stimuli (full AMA), prior, and whitening all come from the training stimuli
- `unit.estimates(estType='mode',stim=None)` - estimates of the latent variable: 'mode' (MAP), 'mean', 'median', 'cmean'
- `unit.performance(estType='mode',stim=None)` - per level: bias, sd, and rmse of the estimates; pCorrect and confusion of the MAP category; and cost
- `unit.cross_validate(n,k=5,seed=0,**train_kw)` - train `n` filters on each of `k` stratified folds and evaluate on the held-out fold
- `unit.responses`, `unit.likelihoods`, `unit.posterior`, `unit.error` - intermediate results for the training stimuli
- `unit.split(stimInd)` - a unit with a subset of the stimuli (evaluated with their own statistics; use `evaluate` for held-out data)
- `unit.freeze_whitening()` / `unit.unfreeze_whitening()` - fix the whitening matrix at its value for the training stimuli, e.g. before `split`

#### Saving
```python
unit.save('unit.pkl')                       # settings, filters, optimizer state, random keys (not the stimuli)
unit=ama.Unit.load('unit.pkl',train_stim)   # training continues identically
```

#### Properties
- `unit.out` - learned filters
- `unit.last` - filters before the last `train_append`/`train_recurse`
- `unit.filter.implied_spatial()` - spatial filters implied by fourier-domain filters

#### Plotting
```python
unit.plot_out()                    # learned filters
unit.plot_last()                   # previous filters
unit.responses.plot_marginal()     # response distributions per category
unit.responses.plot_joint()        # joint responses of filter pairs
unit.responses.plot_tsne()         # t-SNE of the population responses
stim.plot()                        # a stimulus
```

## Notes on hacking
To aid autograd and jit, learning routines do not contain `if` statements on options.
Instead, the objective is composed before execution: the `_TypeFunc` descriptor binds the function for an option when the option is set (e.g. setting `normalizeType='broad'` binds `_normalize__broad`).
The configuration objects are static arguments of jitted functions, keyed by their settings (`_key()`), so a changed setting compiles a new trace; add new settings that change the math to `_key()`.
See the tests in `tests/` for reference implementations of the math.

## Other AMA implementations
[burgelab/AMA](https://github.com/burgelab/AMA) - the original matlab implementation

[dherrera/amatorch](https://github.com/dherrera1911/amatorch) - written in python with pytorch, features learning based on noise-covariance

[portalgun/AMA.DNW.mat](https://github.com/portalgun?tab=repositories) - matlab prototype for ama.py

## TODO
- bias term
- split negatives
- softmax activation, saturating activations (ramp)
- normalization indices
- phase parameter for quadrature pair learning
- fmincon-like options
- installable Filter dependency; move to src
- jupyter notebooks with different data
- better rmax and eps defaults?
- merge Objective and Model?

V2
- learn normalization indices?
- weights and combination learning (no likelihoods?)
- specified noise covariance
- layers


# Works cited
(1) Burge J, Jaini P (2017). Accuracy Maximization Analysis for sensory-perceptual tasks: Computational improvements, filter robustness, and coding advantages for scaled additive noise.  PLoS Computational Biology, 13(2): e1005281. doi:10.1371/journal.pcbi.1005281

(2) Iyer AV, Burge J (2019). The statistics of how natural images drive the responses of neurons. Journal of Vision, 19(13): 4, 1-25, doi: https://doi.org/10.1167/19.13.4

(3) DN White, J Burge. How distinct sources of nuisance variability in natural images and scenes limit human stereopsis. Preprint. (582383). https://doi.org/10.1101/2024.02.27.582383

(4) Jaini P, Burge J (2017). Linking normative models of natural tasks with descriptive models of neural response. Journal of Vision, 17(12):16, 1-26
