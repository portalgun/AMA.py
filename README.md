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
- [AMAdataSpeed.mat](https://github.com/burgelab/AMA/raw/refs/heads/master/AMAdataSpeed.mat) (42 MB, not included; `tests/test_speed.py` uses it from the repository root when present)

On both sets, the AMA-Gauss cost of burgelab's filters matches their reported cost (to 0.001-0.03), and training reaches a lower cost than their filters. Held-out comparison of the likelihood models (4 filters from `train_new(4)`, 600 iterations, the best of 2 seeds by training cost, 30% of the stimuli held out; costs are -log posterior at the correct level, rmse from the posterior mean):

| model | disparity: train / held-out / rmse (arcmin) | speed: train / held-out / rmse (deg/s) |
|---|---|---|
| 'gss' | 1.965 / 1.968 / 7.04 | 1.849 / 1.855 / 2.16 |
| 'gss', `bLeaveOneOut` | 1.981 / 1.968 / 7.04 | 1.858 / 1.855 / 2.16 |
| 'gss', `covShrink=0.1` | 2.008 / 2.010 / 7.11 | 1.901 / 1.912 / 2.12 |
| 'gss', `ctgPoolWidth` one level | 2.038 / 2.039 / 6.83 | 1.937 / 1.943 / 2.15 |
| 'student', `df=5` | **1.815 / 1.830 / 6.79** | **1.676 / 1.698 / 1.83** |
| 'mix', `nMix=2` | 1.949 / 1.966 / 6.99 | 1.838 / 1.850 / 2.15 |
| 'full' | 1.895 / 1.984 / 7.15 | 1.810 / 1.850 / 1.95 |

The heavy-tailed 'student' likelihood decodes both sets best; full AMA fits its training set most closely and generalizes worse than 'gss' on disparity; shrinkage and pooling cost accuracy at these sizes (500 or more stimuli per level).

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
              Yperiod=None,             # period of circular latent variables
              y=None)                   # [ nStim ] or [ nStim x nDim ] each stimulus's own latent value (continuous)
```
Several latent dimensions (`Y [ nCtg x nDim ]`, e.g. disparity and speed) are estimated jointly: each category is one combination of values. Estimates then carry a last axis of length `nDim`, `'l1'`/`'l2'` sum over the dimensions, `'mean'` and `'median'` estimate each dimension, and `performance` reports bias, sd and rmse per dimension.

`Yperiod` makes latent variables circular (e.g. `180` for orientation in degrees, `2*np.pi` for phase): one value for every dimension, or one per dimension with `None` for the linear ones. All distances between latent values wrap: errors, `'mean'` (the circular mean), the divergence targets, 'wasserstein' and 'fisher' (on the circle), and `ctgPoolWidth` pooling. `'median'` is the circular median there: the diameter through it halves the posterior probability, and it minimizes the expected wrapped distance.
AMA assumes contrast-normalized stimuli (zero mean, unit norm); `Stim` warns otherwise. Use `bContrastNormalize=True` or `ama.contrast_normalize(stimuli)`.
For split stimuli, the parts are contiguous blocks of the first stimulus dimension (e.g. left eye then right eye).

Continuous latent variables: when every stimulus has its own latent value (e.g. a measured depth or orientation), pass it as `y`, or let `Stim.binned` make the categories. The categories are then bins, and `Y` holds each bin's level (by default the mean value of its stimuli). Everything that compares an estimate with the truth uses each stimulus's own value: the `'l1'`/`'l2'` errors, the gaussian divergence targets (`targetSigma`, centered at the stimulus's value), and `performance` (which also reports `rmseAll` over all stimuli). With `Model(bWithin=True)` the estimates resolve values within the bins too (see Model).
```python
stim=ama.Stim.binned(x,stimuli,y,       # y [ nStim ] or [ nStim x nDim ]
                     nBins=10,          # bins per latent dimension
                     binType='quantile',# 'quantile' (equal counts) or 'uniform' (equal widths; circular dimensions tile the period)
                     edges=None,        # explicit bin edges instead
                     like=None,         # another binned Stim whose edges and levels to reuse
                     Yperiod=None,**stim_kw)
```

- `ama.Stim.load('file.mat',dims=None,keys=None,**kw)` - burgelab `.mat` files (`s`, `ctgInd`, `X`) or `.npz` files with the same variables; `dims` reshapes each stimulus (e.g. `(32,32)` for images, column-major from matlab), `keys` renames variables, and a file with per-stimulus values `y` instead of `ctgInd`/`X` is binned with `Stim.binned`
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
            readoutType='None',    # pooled readout with learned weights, or a learned second layer ('linear')
            bBias=False,           # learned offsets before the activation (and of the second layer)
            bSplitNegatives=False, # ON/OFF channels: max(r,0) and max(-r,0)
            rsat=None,             # activationType='ramp': saturation (default rmax/2)
            c50=None,              # activationType='nakarushton': semi-saturation (default rmax/4)
            nNR=2.,                #   and exponent
            softmaxT=1.,           # activationType='softmax': temperature
            normPool=None,         # normalizeType='gen': normalization pool weights [ nChan x nChan ]
            bLearnNormPool=False,  #   learned
            nReadout=None,         # readoutType='linear': output neurons (default: the number of channels), or a tuple of layer widths
            readoutActivation='None', # readoutType='linear': the layers' activation
            respBudget=None)       # fixed root mean square of each final response dimension over the training stimuli
```

normalizeType - response normalization. See [Iyer & Burge 2019](https://jov.arvojournals.org/article.aspx?articleid=2755285) (2).
- 'None'
- 'broad' - broadband: divide by the stimulus contrast energy, `||s||`
- 'narrow' - narrowband: divide by the dot product of the stimulus and filter amplitude spectra, `A_s^T A_f`. Stimulus specific but feature independent (2). Requires fourier-domain learning.
- 'gen' - divide by `eps + sum(|r|)` over the population, or by `eps + sum_j normPool_ij |r_j|` with a normalization pool (`normPool`, a `[ nChan x nChan ]` matrix of nonnegative weights over the channels entering normalization: the filters, twice as many with `bSplitNegatives`, or the flattened dimensions when whitened; sub-filters and real/imaginary components are summed per channel). `bLearnNormPool` learns the pool's weights (starting at `normPool`, or all ones); each row keeps its initial total weight, so only the relative weights are learned (with fixed additive noise, a smaller denominator would otherwise always raise the signal-to-noise ratio). Stage-1 noise is carried through the pool to first order.
- 'phase' - unit phasors `r/(|r|+eps)` (signs for real responses): contrast- and gain-invariant responses, the inputs of phase congruency. Most useful with quadrature pairs (fourierType=2).

readoutType - pooling of the (normalized) responses across filters with learned nonnegative weights (softplus of `unit.pool_p`, normalized to sum to 1), before stage-2 noise:
- 'None'
- 'resultant' - the filter responses plus their weighted resultant `sum_j p_j R_j`
- 'resultant_only' - only the weighted resultant; with `normalizeType='phase'` this is weighted phase congruency, decoded by the likelihood
- 'linear' - a second layer of `nReadout` output neurons, each a learned unit-norm combination of all the (normalized) responses (filters, sub-filters and ON/OFF channels; real and imaginary components separately), plus a learned offset with `bBias`, followed by `readoutActivation` (any activationType). Stage-2 noise is on the output neurons, so their scale is fixed by the unit-norm weights (as the filters' is by their norm). Not with stage-1 noise or whitening. Learned weights are `unit.nrn_p['A']` (normalized by `Nrn.readout_matrix`). `nReadout` as a tuple of widths, e.g. `(8, 4)`, stacks layers, each a unit-norm combination of the previous one followed by `readoutActivation` (and offsets with `bBias`); further layers' weights are `unit.nrn_p['A_1']`, `['A_2']`, ... Without an activation the stack is a single linear map.

activationType - 'None', 'relu', 'softplus', 'abs', 'logistic', 'swish', 'swish2', 'tanh', 'gauss', 'igauss' (applied to real and imaginary parts separately), and
- 'ramp' - `min(max(r,0), rsat)`, rectified and saturating
- 'nakarushton' - `rmax r^nNR / (r^nNR + c50^nNR)` for r > 0: the hyperbolic ratio (Naka-Rushton) contrast response function
- 'softmax' - `rmax softmax(r/softmaxT)` over the filters and sub-filters of each stimulus: a population nonlinearity whose responses sum to rmax (real responses only)

Responses are in units of rmax (`r = rmax f^T s`, at most rmax for unit-norm filters and stimuli).

bBias - learned offsets (per filter, sub-filter and component; per flattened dimension when whitened), added to the linear responses before the activation. With a linear response they only shift the category means, but they set the threshold of a rectifying or saturating activation, and change the noise variance `fano*|r|`. Learned values are in `unit.nrn_p['bias']`; `train_append` keeps the previous filters' offsets.

respBudget - a response budget: each final response dimension (after the readout, before stage-2 noise) is scaled so that its root mean square over the training stimuli is `respBudget`. Against the fixed additive noise `var0`, a larger response always has a higher signal-to-noise ratio, so any learned scale (a normalization pool, second-layer weights) can improve the cost by growing the responses instead of by coding better. With a budget only the shape of the responses matters. On the disparity set (4 filters, 'gen' normalization, held-out cost), learning the pool without a budget "improved" the cost from 2.782 to 2.401 while the response RMS grew fourfold (0.39 to 1.52); with `respBudget=1` the learned pool improved it from 2.207 to 2.168. Costs are comparable only at the same budget, which itself sets the signal-to-noise ratio. Stage-1 noise carried through is scaled with the responses. Held-out stimuli (`evaluate`, `estimates`, `performance`) are scaled by the training stimuli's gains, `unit.nrn.gain(...)`; with `batchSize`, each batch uses its own. Training `bLearnNormPool` or `readoutType='linear'` with modeled noise and without a budget warns.

bSplitNegatives - ON/OFF channels: after the activation, each response dimension becomes two rectified channels, `max(r,0)` and `max(-r,0)` (real and imaginary components separately), each with its own noise `fano*|r| + var0`. Firing rates are not negative; a signed response with scaled noise is otherwise an idealization. Doubles the response dimensions; not with 'narrow' normalization or the pooled readouts.

Noise - the likelihood always uses noise of variance `fano*|R| + var0` on the final responses, also when no noise is sampled (`responseType='mean'`, the mean-response approximation of Burge & Jaini 2017). With `responseType='basic'` the decoded responses are noisy samples, so the cost is a one-sample Monte Carlo estimate of the expected cost over the noise; `Model(nNoiseSamples=S)` averages the cost over S noisy observations of each stimulus. (`nSamples` with `averageType` averages the noisy *responses*, which reduces the noise rather than accounting for it.)
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
                nFA=50,                #   EM iterations of the factor analysis fit
                nNeighbors=None,       # modelType='full': nearest references per category while training (None = all)
                nTail=8,               #   random references per category that estimate the rest
                bWithin=False,         # continuous estimates within the categories (Stim y)
                bLooNoise=True,        # bLeaveOneOut: also leave the stimulus out of its category's noise covariance
                nNoiseSamples=1,       # responseType='basic': average the cost over this many noisy observations
                bFixedNoise=False,     #   while training, the same noise draws every iteration (deterministic cost)
                noiseSeed=0)           #   their seed
```

modelType
- 'gss' - AMA-Gauss: the responses to each category are gaussian, with the mean and covariance of the category's mean responses plus the noise covariance (Jaini & Burge 2017)
- 'full' - the original AMA: each category's likelihood is the average over its stimuli of the noisy response distribution to each stimulus (Burge & Jaini 2017, Eq 5)
- 'student' - as 'gss', with a multivariate student t of `df` degrees of freedom (df > 2) whose covariance matches the category covariance plus noise: heavy tails, as natural-image responses have
- 'circ' - a circular (proper) complex gaussian on quadrature-pair responses (fourierType=2, no whitening): hermitian category covariances, so half the parameters of 'gss' on the real and imaginary parts, and circular symmetry built in. With `circMean='zero'` the category mean is fixed at zero and the covariance is the second moment `E[R R^H]`: the likelihood when each stimulus's complex gain (contrast and phase, e.g. edge polarity or feature type) is unknown, `R = G m + noise`

- 'mix' - a gaussian mixture per category with `nMix` components, `p(R|X_i) = sum_c pi_ic N(R; mu_ic, Sigma_ic + noise)`: for responses that are multimodal within a category (e.g. sign or phase flips from nuisance variables), between the single gaussian of 'gss' and the O(N^2) cost of 'full'. The mixture is fit to each category's mean responses by `nEM` EM iterations inside the cost, initialized by quantiles along the category's first principal axis, and differentiated through, so the gradient includes how the fit moves with the filters. `nMix=1` with `mixReg=0` is 'gss'. With `bWarmEM=True`, each training iteration starts EM from the previous iteration's (maximum likelihood) fit and runs only `nEMWarm` iterations; the fit carries across batches, and `unit.loss`/`evaluate` still fit from scratch with `nEM`. `bLeaveOneOut` removes the stimulus's responsibility-weighted share (of its mean response) from its own category's components, holding the responsibilities of the full fit fixed. `covRank` applies to the component covariances. `covShrink` shrinks each component covariance toward its diagonal (`covTarget='diag'`) or the pooled covariance of all categories (`'pooled'`). With `ctgPoolWidth`, component c of category i also borrows the kernel-weighted scatter of the other categories in proportion to its weight, `Sigma_ic = (S_ic + pi_ic P_i)/(dof_ic + pi_ic Q_i)` with `P_i = sum_{j!=i} K_ij S_j` and `Q_i = sum_{j!=i} K_ij (N_j-1)` over their whole-category scatters, so that `nMix=1` is 'gss' with the same pooling (`bPoolMeans` is not used). With pooling, `bLeaveOneOut` also removes the stimulus from the other categories' pooled statistics. A warm start from a fit in which EM emptied a component starts again from the initialization, so the component can recover. Needs at least `2*nMix` stimuli per category (also per batch).

covRank ('gss', 'student', 'mix', 'circ') - each category (or mixture component) covariance is a factor analysis model `L L^T + Psi` (hermitian, `L L^H + Psi`, for 'circ'), with `covRank` factors and a diagonal `Psi`, fit to its sample covariance by `nFA` EM iterations (Ghahramani & Hinton 1996) inside the cost and differentiated through. With many response dimensions and few stimuli per category, it generalizes much better than the sample covariance (which needs more stimuli than dimensions). `covRank=0` is a diagonal covariance. It applies before `covShrink`, and to the left-out covariances of `bLeaveOneOut`.

nRef ('full') - while training, each stimulus is decoded against `nRef` reference stimuli per category, drawn at random every iteration, instead of all of them: O(N nRef) instead of O(N^2) per iteration, for large training sets. The training cost is then a stochastic estimate of full AMA's (slightly pessimistic, the log of a sample mean); `unit.loss`, `evaluate` and `performance` still decode against all training stimuli. Combines with `batchSize` (references are drawn from the batch) and `bLeaveOneOut`.

nNeighbors, nTail ('full') - while training, each stimulus is decoded exactly against the `nNeighbors` references of each category with the largest likelihood terms, and the rest of each category is estimated from `nTail` of its references drawn at random every iteration (their mean term times their number: unbiased for the likelihood, and, since each of them is below the neighbours, with bounded variance). The neighbours are searched again with the current filters at the start of every `Optimizer` chunk (`nStepsPerChunk` iterations): an O(N^2) search, while each iteration is O(N nCtg (nNeighbors + nTail)). `unit.loss`, `evaluate` and `performance` decode against all training stimuli. Combines with `bLeaveOneOut` and `bWithin`; not with `nRef`. With `batchSize`, each iteration decodes a batch of stimuli (stratified by category) against all training stimuli, through their neighbours and the tail, with the prior of all training stimuli: O(batchSize nCtg (nNeighbors + nTail)) per iteration plus the responses of all stimuli (unlike full AMA's batches, which decode the batch against itself).
Where it pays off: with several filters most of each likelihood comes from few references, so random subsets (`nRef`) fail while neighbours track full AMA; on the burgelab disparity set (7,600 stimuli, 200 iterations from random filters) the exact cost after training was

| filters | full AMA | `nNeighbors=16, nTail=16` | `nNeighbors=8, nTail=8` | `nRef=32` |
|---|---|---|---|---|
| 2 | 2.316 | 2.320 | 2.325 | 2.319 |
| 4 | 1.916 | 1.928 | 2.060 | 1.976 |
| 8 | 1.100 | 1.095 | 1.099 | 1.624 |

On the disparity set with 4 filters (RTX 5070, float32, 400 iterations from random filters; exact cost after training): full AMA 4.9 ms per iteration (1.915), batches of 760 decoded against themselves 0.9 ms (2.198), `nNeighbors=16, nTail=8` 4.8 ms (1.920), the same with `batchSize=760` 1.5 ms (2.095). These times are without `--xla_gpu_deterministic_ops` (which `tests/conftest.py` sets): with it the gradient of the neighbours' gathers (a scatter-add) is about 7x slower, 35 and 3.9 ms.
Full AMA's iterations are fast on a GPU up to ~10,000 stimuli (4 ms per iteration at 7,600), so neighbours are for larger sets: on an RTX 5070, with 8 filters, an iteration took 95 ms (full) vs 11 ms (neighbours) at 22,800 stimuli, and 360 ms vs 23 ms at 45,600 (plus a search of 0.14 s per chunk); 47 ms at 91,200.

bWithin - for stimuli with their own latent values (Stim `y`, e.g. `Stim.binned`): each candidate category stands for the expected value of the latent variable given the response and that category, instead of its level `Y_i`, so estimates are not limited to the levels. `'mean'`, `'median'` and `'mode'` (the `'l1'`/`'l2'` errors, `estimates`, `performance`) combine these values with the posterior over categories. For 'full' it is the likelihood-weighted mean of the category's reference stimuli's values (kernel regression; without the stimulus itself under `bLeaveOneOut`); for the other models the linear regression of the value on the noisy responses within the category, `ybar_i + c_i^T (Sigma_i + Lambda_i)^-1 (R - mu_i)` from the sample statistics (no shrinkage or pooling; with `bLeaveOneOut`, the own category's statistics and noise covariance without the stimulus). Training with `'l2'` then learns filters for fine discrimination within the bins as well as between them. Offsets wrap on circular dimensions.

Category statistics ('gss', 'student', 'circ'):
- covShrink, covTarget - each category covariance becomes `(1-covShrink) Sigma + covShrink T`, with T its own diagonal ('diag') or the count-weighted mean covariance over categories ('pooled'), before the noise covariance is added. Stabilizes covariances estimated from few stimuli per response dimension.
- ctgPoolWidth - each category's covariance is the scatter pooled over categories with gaussian weights `exp(-(Y_i-Y_k)^2/(2 w^2))` (times the counts): for latent variables whose response statistics change smoothly (e.g. position, disparity). `bPoolMeans` pools the means too, which biases them toward their neighbours.

responseType
- 'mean' - decode mean responses (noise enters through the likelihood)
- 'basic' - decode noisy responses

nNoiseSamples, bFixedNoise - the mean-response approximation decodes noise-free responses, so its cost underestimates the expected cost of decoding noisy ones: on the disparity set (4 filters, held out) by 0.38 at `fano=0.5, var0=0.23`, 0.19 at `fano=2, var0=1` and 0.08 at `fano=4, var0=2`. With `responseType='basic'`, `nNoiseSamples=S` makes the cost (training, `unit.loss`, `evaluate`) the mean over S independent noisy observations of each stimulus, decoded against the same statistics or references: a Monte Carlo estimate of the expected cost. With `bFixedNoise=True`, training draws the same noise every iteration (from `noiseSeed`), so its cost is a deterministic function of the filters, e.g. for `Optimizer('lbfgs')` or `tolFun`; reference subsets and batches are still drawn anew. Memory grows with S, except for 'full', whose samples run one after another. Needs sampled noise (`Nrn bNoise_1` or `bNoise_2`). On the disparity set, filters trained on the expected cost (S=16, fixed noise) were no better under it than filters trained with the mean-response approximation (2.347 vs 2.348, 2.702 vs 2.702, 2.783 vs 2.782 at the three noise levels): use it to measure the cost honestly; whether it changes the filters depends on the task.

bLeaveOneOut - the likelihoods are fit to the stimuli they score, so each stimulus is otherwise decoded against statistics that include it. That makes the training cost optimistic when noise is low, filters are many, or categories (or batches) are small. With `bLeaveOneOut=True`:
- 'full' - the posterior is Eq 5 with the decoded stimulus removed from the training set
- 'gss', 'student', 'circ' - the category statistics are recomputed without the stimulus, exactly: its own category's mean and covariance by a rank-one downdate, and with `ctgPoolWidth` (and `bPoolMeans`) or `covTarget='pooled'` shrinkage every category's pooled covariance (and mean), which the stimulus also enters. Without `bPoolMeans`, `covRank` or diagonal shrinkage, every left-out pooled covariance is one matrix per pair of categories minus a rank-one term, so the other categories' likelihoods cost O(nF^2) per stimulus after nCtg^2 Cholesky factorizations (the own category's needs one per stimulus with `bLooNoise`): on the disparity set ('student', `ctgPoolWidth=1`, GPU) an iteration took 7.4 ms instead of 21.0 ms with 16 filters (6.9 ms instead of 19.7 ms with `bLooNoise=False`; 5.2 ms without leave-one-out), and 6.7 ms instead of 6.9 ms with 4 (5.2 ms instead of 6.4 ms with `bLooNoise=False`; 3.8 ms without leave-one-out). Needs at least 3 stimuli per category (2 with `ctgPoolWidth` or `circMean='zero'`), also in every batch (`Optimizer nBatchMinCtg`).

The stimulus is also left out of its category's prior, `(N_k-1)/(N-1)`, and of its category's noise covariance (the mean noise variance), unless `bLooNoise=False`, which keeps the category's noise covariance (an O(1/N_k) change; none with `fano=0`). Then, without `covRank`, `covShrink` or pooling, the left-out covariance of 'gss', 'student' and 'circ' is a rank-one downdate of one matrix per category, and each left-out likelihood costs O(nF^2) by the matrix determinant lemma and Sherman-Morrison instead of a Cholesky factorization per stimulus: on the disparity set (GPU, float32) an iteration took 0.47 ms instead of 0.84 ms with 8 filters (0.37 ms without leave-one-out), and 0.43 ms instead of 0.56 ms with 4. Full AMA's form already has the left-out prior. Not available with `errType='mle'`. Held-out evaluation (`unit.evaluate`, `cross_validate`) is unaffected.

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
                        stepDecay=0.01,                  # 'ama_sgd': fraction the step shrinks each iteration
                        tolFun=None,                     # stop when the cost changes less than this (relative) over a chunk
                        tolX=None,                       # stop when no parameter moves more than this over a chunk
                        lbfgsMemory=10)                  # 'lbfgs': stored curvature pairs
```

optimizerType - any optimizer in the [optax documentation](https://optax.readthedocs.io/en/latest/api/optimizers.html), or `'ama_sgd'`: the update of Burge & Jaini (2017) and [burgelab/AMA](https://github.com/burgelab/AMA) (`gradSGD.m`, `updateSGD.m`). Each filter moves a distance `lRate0*(1-stepDecay)^t` (at least `stepMin`) along its unit-normalized tangent-plane gradient and is renormalized, and a step is kept only if it does not raise the cost of the batch it was computed on (evaluated again with the same noise, one extra forward pass per iteration). With `batchSize` this is AMA-SGD; without it, the full-batch cost never increases. It needs `projectionType=['l2_sphere',1]` and applies to `train_new`/`train_append`/`train_recurse` (not the generated filters of `train_parametric`/`train_multiscale`). Unlike burgelab/AMA, which walks each permutation of the training set in consecutive batches, every iteration draws a new batch stratified by category.

optimizerType='lbfgs' - limited-memory BFGS with a zoom line search (`optax.lbfgs`), the quasi-Newton method of matlab's `fminunc`/`fmincon` that the original AMA used: full batch only, and `lRate0` is not used. The cost is evaluated at the normalized filters, which makes the unit-norm constraint exact and keeps the line search on the sphere's tangent space. On the burgelab disparity set it reaches the cost of 400 adam iterations in about 60 iterations. Also for `train_parametric`/`train_multiscale`.

tolFun, tolX - fmincon-like stopping tolerances, checked after every chunk: training stops when the cost changed by less than `tolFun*(1+|cost|)` between the chunk's first and last iterations, or when no parameter moved by more than `tolX` over the chunk. `optimizer.stop_reason` is 'nIterMax', 'patience', 'tolFun' or 'tolX'. Batch costs and costs of sampled noise fluctuate, so `tolFun` must allow for that.

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
unit.train_parametric(n,family='morse',fourierType=2,bTied=True,orientations=None,init=None,stimVal=None,bPhase=False)
```
Learn a parametric filter bank instead of free filters, in the fourier domain (1D or 2D stimuli, not split):
- family 'morse' - generalized Morse spectra `(k/kp)^b exp((b/gamma)(1-(k/kp)^gamma))`, whose low-frequency power-law rise (b) and high-frequency fall (gamma) are learned
- family 'loggabor' - `exp(-log(k/kp)^2/(2 sigma_u^2))`
- 2D stimuli: times a one-sided gaussian angular profile (width sigma_theta) around each filter's orientation (`orientations`, radians from the first stimulus axis)
- bTied - filters are dilations of one mother filter, peaks `kTop/ratio^j`, sharing its shape; otherwise each filter has its own peak and shape
- init - starting values: kTop (cycles per sample), ratio, gamma, b, sigma_u, sigma_theta
- bPhase - also learn a phase per filter, multiplying its spectrum by `exp(i phase)`. The profiles are otherwise zero-phase: even filters with fourierType=1, and cosine/sine quadrature pairs with fourierType=2. A learned phase gives odd or intermediate filters, or rotates each quadrature pair, which matters for 'phase' normalization and pooled readouts, and through the response-dependent noise. `parametric_values()['phase']`

The learned filters become ordinary filters (`unit.out`), so evaluation, saving, and `train_recurse` (free refinement from the parametric solution) work as usual. `unit.parametric_values()` returns the parameters (peak frequencies, ratio, shapes, orientations).

```python
unit.train_multiscale(n,fourierType=2,nKnot=25,uWidth=2.5,scaleType='tied',kTop=0.25,ratio=2.0,kPeak=None,
                      orientations=None,nKnotTheta=13,knotSmooth=0.,init='loggabor',stimVal=None,
                      nScales=None,nOrientations=None,bInterleave=False,nMothers=1,motherInitNoise=0.1,edgeTaper=0.,
                      bPhase=False)
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
- bPhase - a learned phase per filter (`exp(i phase)` on its spectrum), so the dilated copies of a mother need not share its phase, e.g. to align phases across scales for a pooled readout; `multiscale_values()['phase']`

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
- `ama.source_sha256()` - hash of the package's source files; configs and saved units record it (`ama_source_sha256`), since optimization paths can differ numerically between code versions

```python
unit.save_config('unit.yaml')                                   # after training
same=ama.Unit.from_config('unit.yaml', stim, bTrain=True, stimVal=stimVal)
```

#### Evaluation
- `unit.loss` - cost on the training stimuli
- `unit.evaluate(stim)` - cost on other stimuli, decoded with the training set: category response distributions (AMA-Gauss), reference stimuli (full AMA), prior, and whitening all come from the training stimuli
- `unit.estimates(estType='mode',stim=None)` - estimates of the latent variable: 'mode' (MAP), 'mean', 'median', 'cmean'
- `unit.performance(estType='mode',stim=None)` - per level: bias, sd, and rmse of the estimates (from each stimulus's own value with Stim `y`), rmseAll over all stimuli; pCorrect and confusion of the MAP category; and cost
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
`ama.source_sha256()`, the hash of the package source that trained it.

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
The package `ama/` has one module per stage: `_base.py` (shared imports, densities, latent geometry, `_Static`, `_TypeFunc`, configuration helpers), `stim.py` (`Stim`, `_Index`, `Filter`), `nrn.py`, `model.py`, `objective.py`, `optimizer.py`, and `unit.py` (`Unit`, `Response`). Each module starts with `from ._base import *`, whose `__all__` includes the private helpers, and `import ama` re-exports every class and helper. Module constants live where they are used (e.g. `ama.model._FULL_CHUNK`), so patch them there.
To aid autograd and jit, learning routines do not contain `if` statements on options.
Instead, the objective is composed before execution: the `_TypeFunc` descriptor binds the function for an option when the option is set (e.g. setting `normalizeType='broad'` binds `_normalize__broad`).
The configuration objects are static arguments of jitted functions, keyed by their settings (`_key()`), so a changed setting compiles a new trace; add new settings that change the math to `_key()`.
See the tests in `tests/` for reference implementations of the math.

## Other AMA implementations
[burgelab/AMA](https://github.com/burgelab/AMA) - the original matlab implementation

[dherrera/amatorch](https://github.com/dherrera1911/amatorch) - written in python with pytorch, features learning based on noise-covariance

[portalgun/AMA.DNW.mat](https://github.com/portalgun?tab=repositories) - matlab prototype for ama.py

## TODO
- move to src
- jupyter notebooks with different data
- better rmax and eps defaults?
- merge Objective and Model?

Known limitations
- likelihoods: 'mix' leave-one-out holds the responsibilities fixed
- noise: the default mean-response approximation underestimates the expected cost (Model nNoiseSamples estimates it;
  on the disparity set training on it did not change the filters); stage-1 noise is carried through 'gen' normalization only to first order; noise correlations are specified (rho), with
  stimulus-dependent variances but a fixed correlation structure
- scaling: without covRank, AMA-Gauss needs more stimuli per category than response dimensions
- encoder: linear unit-norm filters with a fixed-shape nonlinearity (learned offsets with bBias) and learned
  unit-norm layers after them (readoutType 'linear', nReadout a tuple of widths) with one activation; `train_append` is greedy and the objective is nonconvex (restarts help)
- learned normalization pools and second-layer weights: without Nrn respBudget, the fixed additive noise rewards larger
  responses, which the unit-norm/row-sum constraints do not prevent (on the disparity set most of a learned pool's gain);
  training them without a budget warns
- latent variables: 'fisher' with several dimensions needs a grid of levels; the entropic 'wasserstein' (several
  dimensions, gaussian target) is approximate (otEps); the circular target is a nearest-image wrapped gaussian (fine for targetSigma well below
  the period); continuous latent values are binned for the likelihood (bWithin estimates within the bins: linear
  regression for the parametric models)
- full AMA: nNeighbors still searches all pairs once per chunk (O(N^2), chunked), and without batches is no faster than
  exact full AMA below ~10,000 stimuli on a GPU; with batchSize every stimulus's responses are still computed each
  iteration

V2
- deeper encoders beyond stacked readout layers (e.g. learned nonlinearities, convolutional structure)


Possible optimizations
- bLeaveOneOut ('gss', 'student', 'circ', 'mix'): with the default bLooNoise=True (and with covRank or covShrink),
  one Cholesky per stimulus of its left-out covariance: the left-out noise covariance rescales its diagonal per stimulus,
  which is not a low-rank update (bLooNoise=False uses the rank-one downdate; pooling without bPoolMeans uses rank-one
  downdates for the other categories). The full lAll is also computed before its
  diagonal is replaced. With covRank, leave-one-out runs the factor analysis EM per stimulus. With pooling (or the pooled shrinkage
  target) every category's covariance differs per stimulus: one Cholesky per stimulus and category, and tensors of
  [ nStim x nCtg x nF x nF ] per category (O(nCtg^2 nF^2) work per stimulus); a low-rank update of the shared pooled
  statistics would avoid materializing them.

# Works cited
(1) Burge J, Jaini P (2017). Accuracy Maximization Analysis for sensory-perceptual tasks: Computational improvements, filter robustness, and coding advantages for scaled additive noise.  PLoS Computational Biology, 13(2): e1005281. doi:10.1371/journal.pcbi.1005281

(2) Iyer AV, Burge J (2019). The statistics of how natural images drive the responses of neurons. Journal of Vision, 19(13): 4, 1-25, doi: https://doi.org/10.1167/19.13.4

(3) DN White, J Burge. How distinct sources of nuisance variability in natural images and scenes limit human stereopsis. Preprint. (582383). https://doi.org/10.1101/2024.02.27.582383

(4) Jaini P, Burge J (2017). Linking normative models of natural tasks with descriptive models of neural response. Journal of Vision, 17(12):16, 1-26
