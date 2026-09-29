"""
Accuracy Maximization Analysis (AMA) with JAX and optax (Burge & Jaini 2017; AMA-Gauss: Jaini & Burge 2017).

Pipeline (Unit._loss_fun*):
    Stim -> Nrn (responses, whitening, noise, normalization) -> Model (log-likelihoods)
         -> Objective (posterior, estimate, error, loss) -> Optimizer (optax, unit-norm filters)

Shapes (stimuli are grouped by category and padded with zeros; Stim.weights marks the valid stimuli):
    stim  [ nPix x nStim_Ctg x nCtg ]          split: [ nPix/nSplit x nSplit x nStim_Ctg x nCtg ]
    f     [ nPix x nF ]                        split: [ nPix/nSplit x nSplit x nF ]
    R     [ nF x nStim_Ctg x nCtg ]            split: [ nF x nSplit x nStim_Ctg x nCtg ]
          flattened to real dimensions [ nDim x nStim_Ctg x nCtg ] for the likelihood (_flatten_responses)
    lAll  [ nStim_Ctg x nCtg(true) x nCtg(candidate) ]   log-likelihoods; log-posteriors after Objective
"""
from ._base import *
from .stim import *
from .nrn import *
from .model import *
from .objective import *
from .optimizer import *
from .unit import *
from . import _base, stim, nrn, model, objective, optimizer, unit
