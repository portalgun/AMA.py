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
cd AMA.py
pip install jax optax numpy scipy matplotlib scikit-learn pyyaml pytest
pip install pacmap phate                   # optional: PaCMAP and PHATE response embeddings
```
`Filter.py` in this repository holds the parts of [Filter](https://github.com/portalgun/Filter.py) that ama uses (the sample grid `Filter.X` and `plotFT`), so the full package is not needed.
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
              Y,                        # [ nCtg ] or [ nCtg x nDim ] latent value(s) of each category, in sorted label order
              bStimIsFourier=False,     # whether stimuli are in the fourier domain
              nSplit=0,                 # number of parts for split filters (e.g. 2 for the two eyes)
              bStimIsSplit=False,       # whether stimuli are already split
              bContrastNormalize=False, # contrast normalize the stimuli
              Yperiod=None)             # period of circular latent variables
```
Several latent dimensions (`Y [ nCtg x nDim ]`, e.g. disparity and speed) are estimated jointly: each category is one combination of values. Estimates then carry a last axis of length `nDim`, `'l1'`/`'l2'` sum over the dimensions, `'mean'` and `'median'` estimate each dimension, and `performance` reports bias, sd and rmse per dimension.

`Yperiod` makes latent variables circular (e.g. `180` for orientation in degrees, `2*np.pi` for phase): one value for every dimension, or one per dimension with `None` for the linear ones. All distances between latent values wrap: errors, `'mean'` (the circular mean), the divergence targets, 'wasserstein' and 'fisher' (on the circle), and `ctgPoolWidth` pooling. `'median'` is the circular median there: the diameter through it halves the posterior probability, and it minimizes the expected wrapped distance.
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
            rho=0,                 # noise correlation: a number, or a correlation matrix
            averageType='full',    # how noise samples are combined
            nSamples=1,            # noise samples per response
            whitenType='None',     # population whitening before noise
            whitenMethod='zca',
            whitenEps=1e-5,
            readoutType='None')   # pooled readout with learned weights
```

normalizeType - response normalization. See [Iyer & Burge 2019](https://jov.arvojournals.org/article.aspx?articleid=2755285) (2).
- 'None'
- 'broad' - broadband: divide by the stimulus contrast energy, `||s||`
- 'narrow' - narrowband: divide by the dot product of the stimulus and filter amplitude spectra, `A_s^T A_f`. Stimulus specific but feature independent (2). Requires fourier-domain learning.
- 'gen' - divide by `eps + sum(|r|)` over the population
- 'phase' - unit phasors `r/(|r|+eps)` (signs for real responses): contrast- and gain-invariant responses, the inputs of phase congruency. Most useful with quadrature pairs (fourierType=2).

readoutType - pooling of the (normalized) responses across filters with learned nonnegative weights (softplus of `unit.pool_p`, normalized to sum to 1), before stage-2 noise:
- 'None'
- 'resultant' - the filter responses plus their weighted resultant `sum_j p_j R_j`
- 'resultant_only' - only the weighted resultant; with `normalizeType='phase'` this is weighted phase congruency, decoded by the likelihood

activationType - 'None', 'relu', 'softplus', 'abs', 'logistic', 'swish', 'swish2', 'tanh', 'gauss', 'igauss' (applied to real and imaginary parts separately)

Noise - the likelihood always uses noise of variance `fano*|R| + var0` on the final responses, also when no noise is sampled (`responseType='mean'`, the mean-response approximation of Burge & Jaini 2017). With `responseType='basic'` the decoded responses are noisy samples, so the cost is a Monte Carlo estimate of the expected cost over the noise (more samples per step with `nSamples` and `averageType`).
Noise sampled before normalization (`bNoise_1`) is carried through the normalization: exactly for 'broad' and 'narrow', to first order for 'gen', and not at all through 'phase', which is not linear (rejected).
`var0` must be positive whenever noise is modeled (`rho` not None): responses of 0 (padding, or zeroed by an activation) would otherwise have zero variance.

rho - noise correlation
- None = no noise (only with `modelType='gss'`)
- 0 = independent noise
- a number: correlates the noise of all response dimensions (filters, sub-filters, and real/imaginary components) equally
- a correlation matrix `[ nDim x nDim ]` (symmetric, unit diagonal, positive definite): the correlation of each pair of response dimensions, in sampling and in the likelihood. Dimensions are ordered as the flattened responses: the real parts of all filters (sub-filters within each filter), then the imaginary parts (`fourierType=2`)

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
                bLeaveOneOut=False,    # leave each decoded stimulus out of its own category
                covShrink=0.,          # shrink category covariances toward covTarget (0-1)
                covTarget='diag',      # 'diag' or 'pooled'
                df=5.,                 # degrees of freedom for modelType='student'
                ctgPoolWidth=None,     # pool category statistics over neighbouring Y (kernel width in units of Y)
                bPoolMeans=False,      # also pool the category means
                circMean='estimate',   # modelType='circ': 'estimate' or 'zero' (unknown complex gain)
                nMix=2,                # modelType='mix': gaussian components per category
                nEM=20,                #   EM iterations
                mixReg=1e-3,           #   ridge on the component covariances (relative to the category variance)
                bWarmEM=False,         #   while training, start EM from the previous iteration's fit
                nEMWarm=3,             #   and run this many EM iterations
                nRef=None,             # modelType='full': reference stimuli per category while training (None = all)
                covRank=None,          # low-rank (factor analysis) category covariances: number of factors
                nFA=50)                #   EM iterations of the factor analysis fit
```

modelType
- 'gss' - AMA-Gauss: the responses to each category are gaussian, with the mean and covariance of the category's mean responses plus the noise covariance (Jaini & Burge 2017)
- 'full' - the original AMA: each category's likelihood is the average over its stimuli of the noisy response distribution to each stimulus (Burge & Jaini 2017, Eq 5)
- 'student' - as 'gss', with a multivariate student t of `df` degrees of freedom (df > 2) whose covariance matches the category covariance plus noise: heavy tails, as natural-image responses have
- 'circ' - a circular (proper) complex gaussian on quadrature-pair responses (fourierType=2, no whitening): hermitian category covariances, so half the parameters of 'gss' on the real and imaginary parts, and circular symmetry built in. With `circMean='zero'` the category mean is fixed at zero and the covariance is the second moment `E[R R^H]`: the likelihood when each stimulus's complex gain (contrast and phase, e.g. edge polarity or feature type) is unknown, `R = G m + noise`

- 'mix' - a gaussian mixture per category with `nMix` components, `p(R|X_i) = sum_c pi_ic N(R; mu_ic, Sigma_ic + noise)`: for responses that are multimodal within a category (e.g. sign or phase flips from nuisance variables), between the single gaussian of 'gss' and the O(N^2) cost of 'full'. The mixture is fit to each category's mean responses by `nEM` EM iterations inside the cost, initialized by quantiles along the category's first principal axis, and differentiated through, so the gradient includes how the fit moves with the filters. `nMix=1` with `mixReg=0` is 'gss'. With `bWarmEM=True`, each training iteration starts EM from the previous iteration's (maximum likelihood) fit and runs only `nEMWarm` iterations; the fit carries across batches, and `unit.loss`/`evaluate` still fit from scratch with `nEM`. `bLeaveOneOut` removes the stimulus's responsibility-weighted share from its own category's components, holding the responsibilities of the full fit fixed. `covRank` applies to the component covariances. Needs at least `2*nMix` stimuli per category (also per batch); not with `ctgPoolWidth` or `covShrink`.

covRank ('gss', 'student', 'mix') - each category (or mixture component) covariance is a factor analysis model `L L^T + Psi`, with `covRank` factors and a diagonal `Psi`, fit to its sample covariance by `nFA` EM iterations (Ghahramani & Hinton 1996) inside the cost and differentiated through. With many response dimensions and few stimuli per category, it generalizes much better than the sample covariance (which needs more stimuli than dimensions). `covRank=0` is a diagonal covariance. It applies before `covShrink`, and to the left-out covariances of `bLeaveOneOut`.

nRef ('full') - while training, each stimulus is decoded against `nRef` reference stimuli per category, drawn at random every iteration, instead of all of them: O(N nRef) instead of O(N^2) per iteration, for large training sets. The training cost is then a stochastic estimate of full AMA's (slightly pessimistic, the log of a sample mean); `unit.loss`, `evaluate` and `performance` still decode against all training stimuli. Combines with `batchSize` (references are drawn from the batch) and `bLeaveOneOut`.

Category statistics ('gss', 'student', 'circ'):
- covShrink, covTarget - each category covariance becomes `(1-covShrink) Sigma + covShrink T`, with T its own diagonal ('diag') or the count-weighted mean covariance over categories ('pooled'), before the noise covariance is added. Stabilizes covariances estimated from few stimuli per response dimension.
- ctgPoolWidth - each category's covariance is the scatter pooled over categories with gaussian weights `exp(-(Y_i-Y_k)^2/(2 w^2))` (times the counts): for latent variables whose response statistics change smoothly (e.g. position, disparity). `bPoolMeans` pools the means too, which biases them toward their neighbours.

responseType
- 'mean' - decode mean responses (noise enters through the likelihood)
- 'basic' - decode noisy responses

bLeaveOneOut - the likelihoods are fit to the stimuli they score, so each stimulus is otherwise decoded against statistics that include it. That makes the training cost optimistic when noise is low, filters are many, or categories (or batches) are small. With `bLeaveOneOut=True`:
- 'full' - the posterior is Eq 5 with the decoded stimulus removed from the training set
- 'gss', 'student', 'circ' - the category statistics are recomputed without the stimulus, exactly: its own category's mean and covariance by a rank-one downdate, and with `ctgPoolWidth` (and `bPoolMeans`) or `covTarget='pooled'` shrinkage every category's pooled covariance (and mean), which the stimulus also enters. Needs at least 3 stimuli per category (2 with `ctgPoolWidth` or `circMean='zero'`), also in every batch (`Optimizer nBatchMinCtg`).

The stimulus is also left out of its category's prior, `(N_k-1)/(N-1)`, and of its category's noise covariance (the mean noise variance). Full AMA's form already has the left-out prior. Not available with `errType='mle'`. Held-out evaluation (`unit.evaluate`, `cross_validate`) is unaffected.

### Objective
```python
objective=ama.Objective(errType='map',   # error
                        bPosterior=None, # decode the posterior (default for all but 'mle')
                        estType=None,    # estimator for 'l1' and 'l2'
                        lossType='mean', # how errors are combined
                        regType='None',  # filter penalty while training
                        regWeight=0.,    # its weight
                        targetSigma=None,# target for the divergence errTypes
                        otEps=0.05,      # 'wasserstein', several latent dimensions, gaussian target: sinkhorn regularization
                        nOtIter=200)     # and iterations
```

errType
- 'map' - `-log posterior` at the correct level: the 0,1 cost (Burge & Jaini 2017, Eq 9)
- 'mle' - `-log likelihood` at the correct level
- 'l2' (or 2) - squared error of the estimate; estType defaults to 'mean' (MMSE)
- 'l1' (or 1) - absolute error of the estimate; estType defaults to 'median'

Divergences between the posterior over the levels and a target distribution around the correct level:
- 'xent' - cross-entropy; the same as 'map' with the one-hot target
- 'js' - Jensen-Shannon divergence (natural log, at most log 2)
- 'wasserstein' - 1-Wasserstein (earth mover's) distance over `Y`; with the one-hot target, the posterior mean of `|Y - X_k|`
- 'fisher' - Fisher divergence: the target-weighted squared difference of the scores `d log p / dY`, taken as finite differences between neighbouring levels. It doesn't depend on the posterior's normalization, and needs `targetSigma`

targetSigma - `None` for a one-hot target, or the standard deviation, in units of `Y`, of a gaussian target `q_i ~ exp(-(Y_i - X_k)^2 / 2 targetSigma^2)`. On circular dimensions (`Stim Yperiod`) the target is a wrapped gaussian (nearest image). 'wasserstein' and 'fisher' use the distances between levels, so their scale follows `Y`.

With several latent dimensions (euclidean distance between levels, wrapped on circular dimensions):
- 'wasserstein' with the one-hot target is exact (the posterior mean distance to the correct level). With a gaussian target it is the debiased entropic (Sinkhorn) divergence `OT(p,q) - OT(p,p)/2 - OT(q,q)/2`, zero at the target, with regularization `otEps` times the mean distance between levels and `nOtIter` iterations: a smaller `otEps` approaches the exact distance but needs more iterations. Its gradient comes from the Sinkhorn potentials (no backpropagation through the iterations).
- 'fisher' needs the levels on a cartesian grid (every combination of the per-dimension values): scores are finite differences along each grid axis, and the divergence sums the per-axis terms.
- Circular dimensions must span less than their period.

estType - 'mean' (circular on circular dimensions), 'median' (per dimension; on circular dimensions the circular median), 'mode' (MAP; no gradient), or 'cmean' (circular mean, Y in radians; superseded by `Stim Yperiod` with 'mean')

lossType - 'mean' or 'median' over the valid stimuli

regType - a penalty on the learned filter coefficients, added to the cost during training as `regWeight` times its mean over filters (not included in `unit.loss`; see `unit.penalty`). It acts in the learning domain:
- 'None'
- 'l1' - `sum |f|`: sparse filters, or sparse spectra when learning in the fourier domain
- 'smooth' - summed squared differences between neighbouring learned coefficients: smooth spatial filters, or smooth spectra (compact spatial support) in the fourier domain

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
                        nBatchMinCtg=2,                  # minimum stimuli per category in a batch
                        patience=None,                   # early stopping (with stimVal)
                        stepMin=0.,                      # 'ama_sgd': smallest step
                        stepDecay=0.01)                  # 'ama_sgd': fraction the step shrinks each iteration
```

optimizerType - any optimizer in the [optax documentation](https://optax.readthedocs.io/en/latest/api/optimizers.html), or `'ama_sgd'`: the update of Burge & Jaini (2017) and [burgelab/AMA](https://github.com/burgelab/AMA) (`gradSGD.m`, `updateSGD.m`). Each filter moves a distance `lRate0*(1-stepDecay)^t` (at least `stepMin`) along its unit-normalized tangent-plane gradient and is renormalized, and a step is kept only if it does not raise the cost of the batch it was computed on (evaluated again with the same noise, one extra forward pass per iteration). With `batchSize` this is AMA-SGD; without it, the full-batch cost never increases. It needs `projectionType=['l2_sphere',1]` and applies to `train_new`/`train_append`/`train_recurse` (not the generated filters of `train_parametric`/`train_multiscale`). Unlike burgelab/AMA, which walks each permutation of the training set in consecutive batches, every iteration draws a new batch stratified by category.

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

patience - early stopping. When training is given validation stimuli (`unit.train_new(n,stimVal=val)`, also `train_append`, `train_recurse`, `train_parametric`), the cost on them is computed after every chunk (`optimizer.val_hist`), training stops after `patience` chunks without improvement (None: never stops early), and the filters with the lowest validation cost are kept (`optimizer.best_step`).

### Unit
```python
unit=ama.Unit(stim,nrn,model,objective,optimizer=None,seed=None,name=None)
```
`name` labels the unit (used as the title of its figures, kept by `save`/`load` and written to the yaml configuration).

#### Learning
```python
unit.train_new(n,fourierType=None,bSplit=None,stimInd=None,dtype=None,optimizer=None,nRestarts=1)
```
Discard current filters (if any) and learn `n` new filters. With `nRestarts=k`, learn from `k` random initial filter sets and keep the one with the lowest cost: on the validation stimuli when `stimVal` is given, else on the training stimuli (costs in `unit.restart_costs`).

```python
unit.train_append(n,...,nRestarts=1)
```
Learn `n` new filters while keeping the others fixed.

```python
unit.train_recurse(ind_rec=None,...)
```
Continue learning the current filters (or only those in `ind_rec`).

All three accept `stimVal` for early stopping (see Optimizer `patience`).

```python
unit.train_parametric(n,family='morse',fourierType=2,bTied=True,orientations=None,init=None,stimVal=None)
```
Learn a parametric filter bank instead of free filters, in the fourier domain (1D or 2D stimuli, not split):
- family 'morse' - generalized Morse spectra `(k/kp)^b exp((b/gamma)(1-(k/kp)^gamma))`, whose low-frequency power-law rise (b) and high-frequency fall (gamma) are learned
- family 'loggabor' - `exp(-log(k/kp)^2/(2 sigma_u^2))`
- 2D stimuli: times a one-sided gaussian angular profile (width sigma_theta) around each filter's orientation (`orientations`, radians from the first stimulus axis)
- bTied - filters are dilations of one mother filter, peaks `kTop/ratio^j`, sharing its shape; otherwise each filter has its own peak and shape
- init - starting values: kTop (cycles per sample), ratio, gamma, b, sigma_u, sigma_theta

The learned filters become ordinary filters (`unit.out`), so evaluation, saving, and `train_recurse` (free refinement from the parametric solution) work as usual. `unit.parametric_values()` returns the parameters (peak frequencies, ratio, shapes, orientations).

```python
unit.train_multiscale(n,fourierType=2,nKnot=25,uWidth=2.5,scaleType='tied',kTop=0.25,ratio=2.0,kPeak=None,
                      orientations=None,nKnotTheta=13,knotSmooth=0.,init='loggabor',stimVal=None,
                      nScales=None,nOrientations=None,bInterleave=False,nMothers=1,motherInitNoise=0.1,edgeTaper=0.)
```
Learn a non-parametric multiscale filter bank: every filter is a copy of one learned mother filter, dilated (shifted in log frequency to its peak frequency) and, for 2D stimuli, rotated (shifted in angle to its orientation). The mother filter has no functional form: complex spectrum values at `nKnot` points uniform over `log(k/k_peak)` in `[-uWidth, uWidth]` (2D: a `nKnot x nKnotTheta` grid over log radius and angle in `[-pi/2, pi/2]`), linearly interpolated and zero outside. Filters are unit norm, in the fourier domain (1D or 2D stimuli, not split).
- scaleType - 'tied' (peaks `kTop/ratio^j`, kTop and ratio learned), 'fixed' (peaks `kPeak`, not learned), or 'free' (a learned peak per filter)
- knotSmooth - weight of a penalty on squared second differences of the mother's knot values, relative to its energy
- init - initial mother: 'loggabor' (a log-gaussian bump) or 'random'
- nScales, nOrientations - 2D: build the bank as a grid (n need not be given): peaks `kTop/ratio^j`, j = 0..nScales-1, at orientations `o*pi/nOrientations`, o = 0..nOrientations-1 (wrapped to (-pi/2, pi/2]), `n = nScales*nOrientations`; `kPeak` and `orientations` are then set by the grid
- bInterleave - with the grid: add an interleaved grid of the same size (`n = 2*nScales*nOrientations`) at the tritones, half a scale step (peaks `/sqrt(ratio)`) and half a rotation step (`+pi/(2*nOrientations)`) from the primary grid, sharing the same mother filter. With scaleType='tied' the interleaved scales stay halfway in log frequency as the ratio is learned. Filter order: the primary grid (scale first), then the interleaved grid; `multiscale_values()` adds `scale_step`, `nScales`, `nOrientations`, `bInterleave`, and an `interleaved` mask
- nMothers - learn this many mother filters simultaneously. Every mother gets the same scale (and orientation) filters, so the bank has `nMothers * n` filters, ordered mother first; scales and orientations are shared. `n` (or the grid) counts the filters per mother. `multiscale_values()` then returns `mother` as `[ nMothers x nKnot (x nKnotTheta) ]`, per-filter `k_peak`, `scale_step` and `orientations`, and `mother_index`
- edgeTaper - force each mother to vanish at the ends of its log-frequency support: its knots are multiplied by a fixed window rising (cosine) from 0 at the outermost knots to 1 over this fraction of the knot range at each end (0 = none; 0.2 tapers 5 of 25 knots per end). Without it a mother can move its energy to the edge of the support, so its dilated copies become narrowband filters cut off at the boundary. The reported mother, the filters and `knotSmooth` use the tapered knots; in 2D the taper is along log radius only
- motherInitNoise - with `nMothers > 1`, independent complex gaussian noise of this size (relative to the initial bump) added to each mother's initial knots; identical mothers would get identical gradients and never diverge

As with `train_parametric`, the filters become ordinary filters; `unit.multiscale_values()` returns the mother filter (`mother`, `log_freq_knots`, `angle_knots`) and the scales (`k_peak`, `ratio`). This learns the shape of a dilation-invariant bank from the task, without assuming log-Gabor, Morse, or any other family.

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

#### Configuration files (yaml)
Every call to `train_new`, `train_recurse`, `train_append`, `train_parametric` and `train_multiscale` is recorded (method and all arguments, defaults included) in `unit.train_log`, which `save` / `load` keep.
- `unit.config()` - the unit's options and settings as a plain dict: `nrn`, `model`, `objective` and `optimizer` settings, `seed`, the filter layout (`filters`: n, fourierType, dtype, shape, whether it has pooling weights), a summary of the training stimuli (`stim`), and the training calls (`train`). Learned filters are not included (use `save`).
- `unit.save_config('unit.yaml')` - write it as yaml
- `ama.Unit.from_config('unit.yaml', stim, bTrain=False, stimVal=None)` - build a unit from a yaml file (or dict) with these stimuli; `bTrain=True` replays the recorded training calls, passing `stimVal` to the calls that had validation stimuli. Unknown top-level keys (e.g. project metadata) are ignored
- `ama.config_from_saved('unit.pkl')` - the configuration of a saved unit, without loading stimuli
- `ama.load_config('unit.yaml')` - read a configuration file
- `ama.source_sha256()` - hash of the `ama.py` source; configs and saved units record it (`ama_source_sha256`), since optimization paths can differ numerically between code versions

```python
unit.save_config('unit.yaml')                                   # after training
same=ama.Unit.from_config('unit.yaml', stim, bTrain=True, stimVal=stimVal)
```

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
unit.save_config('unit.yaml')               # the settings and training calls alone (see Configuration files)
```
`save` also keeps the unit's `name`, its `train_log` (the training calls), the bank metadata of `train_multiscale` /
`train_parametric` (so `multiscale_values()` / `parametric_values()` work after `load`), the pooling weights, and
`ama.source_sha256()`, the hash of the `ama.py` that trained it.

#### Properties
- `unit.out` - learned filters
- `unit.last` - filters before the last `train_append`/`train_recurse`
- `unit.pool_p` - pooling-weight parameters of a readout (weights: `ama.Nrn.pool_weights(unit.pool_p)`)
- `unit.penalty` - the current filters' regType penalty
- `unit.filter.implied_spatial()` - spatial filters implied by fourier-domain filters
- `unit.multiscale_values()` / `unit.parametric_values()` - the learned bank: mother filter(s), peak frequencies, ratio,
  orientations, and (with a grid) `nScales`, `nOrientations`, `bInterleave`, `nMothers`, `edgeTaper`
- `unit.train_log` - the training calls made so far (method and all arguments), the basis of `config()`
- `unit.name` - the unit's label

#### Plotting
```python
unit.plot_out()                    # learned filters
unit.plot_last()                   # previous filters
unit.responses.plot_marginal()     # response distributions per category
unit.responses.plot_joint()        # joint responses of filter pairs
unit.responses.plot_tsne()         # t-SNE of the population responses
unit.plot_filter_bank('bank.png')                  # filters, log and linear amplitude spectra, pooling weights, mother filter(s)
unit.plot_response_embedding(test_stim, 'tsne', 'tsne.png')   # t-SNE of the filter responses to (held-out) stimuli + pooled resultant
unit.plot_response_embedding(test_stim, 'pacmap')  # or PaCMAP / PHATE (plot_response_tsne / _pacmap / _phate are shortcuts)
unit.plot_response_embeddings(test_stim)           # all three in one figure (+ the pooled resultant), same responses
unit.save_figures('run/stage', test_stim)          # run/stage_filters.png, _{tsne,pacmap,phate}.png and _embeddings.png
#   methods=('tsne',) or methods={'tsne': {'perplexity': 20}, 'phate': {}} for per-method options
```
Embeddings need scikit-learn (t-SNE), `pacmap` (PaCMAP) and `phate` (PHATE).
`ama.Unit(..., name='mixed amplitude / natural / phase pooled')` labels a unit: the name titles its figures (or pass `name=` to a
plotting call), and is kept by `save` / `load` and in the yaml configuration. `save` / `load` also keep the bank metadata of
`train_multiscale` / `train_parametric`, so `multiscale_values()` works on loaded units. `plot_filter_bank(bankInfo=...)` takes
that metadata for units saved without it.
```python
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
- move to src
- jupyter notebooks with different data
- better rmax and eps defaults?
- merge Objective and Model?

Known limitations
- likelihoods: 'mix' has no pooling or shrinkage; its leave-one-out holds the responsibilities fixed, and an EM
  component that empties stays empty (also across warm-started iterations)
- noise: the default mean-response approximation is optimistic at high noise (responseType='basic' samples the noise
  instead); stage-1 noise is carried through 'gen' normalization only to first order; noise correlations are specified (rho), with
  stimulus-dependent variances but a fixed correlation structure
- scaling: without covRank, AMA-Gauss needs more stimuli per category than response dimensions
- encoder: linear unit-norm filters with a fixed nonlinearity; `train_append` is greedy and the objective is nonconvex
  (restarts help); no learned nonlinearity or hierarchy (see 'layers')
- latent variables: 'fisher' with several dimensions needs a grid of levels; the entropic 'wasserstein' (several
  dimensions, gaussian target) is approximate (otEps); the circular target is a nearest-image wrapped gaussian (fine for targetSigma well below
  the period); latent values are still a discrete set of levels (estimates between levels only through 'mean'/'median')
- engineering: ama.py is a single ~4000 line module; optax is pinned to a git commit in setup.py

V2
- learn normalization indices?
- weights and combination learning beyond the pooled resultant (readoutType)
- low-rank (signal plus noise) category covariances for 'circ'
- layers


Possible optimizations
- bLeaveOneOut ('gss', 'student', 'circ', 'mix'): one Cholesky per stimulus of its left-out covariance. The left-out
  category covariance is a rank-one downdate, but the left-out noise covariance rescales its diagonal per stimulus, so
  the matrix determinant lemma and Sherman-Morrison apply only with independent noise of equal variances (or with the
  noise term kept fixed); then each left-out log-likelihood would take O(nF^2) instead of O(nF^3). The full lAll is
  also computed before its diagonal is replaced. With covRank, leave-one-out runs the factor analysis EM per stimulus. With pooling (or the pooled shrinkage
  target) every category's covariance differs per stimulus: one Cholesky per stimulus and category, and tensors of
  [ nStim x nCtg x nF x nF ] per category (O(nCtg^2 nF^2) work per stimulus); a low-rank update of the shared pooled
  statistics would avoid materializing them.

# Works cited
(1) Burge J, Jaini P (2017). Accuracy Maximization Analysis for sensory-perceptual tasks: Computational improvements, filter robustness, and coding advantages for scaled additive noise.  PLoS Computational Biology, 13(2): e1005281. doi:10.1371/journal.pcbi.1005281

(2) Iyer AV, Burge J (2019). The statistics of how natural images drive the responses of neurons. Journal of Vision, 19(13): 4, 1-25, doi: https://doi.org/10.1167/19.13.4

(3) DN White, J Burge. How distinct sources of nuisance variability in natural images and scenes limit human stereopsis. Preprint. (582383). https://doi.org/10.1101/2024.02.27.582383

(4) Jaini P, Burge J (2017). Linking normative models of natural tasks with descriptive models of neural response. Journal of Vision, 17(12):16, 1-26
