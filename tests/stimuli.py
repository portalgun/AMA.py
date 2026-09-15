"""
Synthetic, labeled stimulus sets with known structure for testing AMA.

Every generator returns (x, stimuli, yCtgInd, Y, info):
    x        Filter.X meshgrid for the stimulus
    stimuli  [ *dims x nStim ] contrast-normalized (zero mean, unit L2 norm), as AMA assumes
    yCtgInd  [ nStim ] integer category label of each stimulus
    Y        [ nCtg ] latent variable value of each category
    info     dict with ground truth useful for assertions (e.g. the informative direction)

Pass everything but info to ama.Stim:  ama.Stim(x, stimuli, yCtgInd, Y)
"""
import numpy as np
import Filter as filt


def contrast_normalize(stimuli):
    """mean-subtract and scale each stimulus (last axis) to unit L2 norm"""
    dims=stimuli.shape[:-1]
    s=np.reshape(stimuli,(-1,stimuli.shape[-1]))
    s=s-s.mean(axis=0,keepdims=True)
    s=s/np.linalg.norm(s,axis=0,keepdims=True)
    return np.reshape(s,dims+(s.shape[-1],))


def _labels(nCtg,nStimPerCtg):
    counts=np.broadcast_to(np.asarray(nStimPerCtg),(nCtg,)).astype(int)
    return np.repeat(np.arange(nCtg),counts)


def gaussian_ctg(nPix=16,nCtg=5,nStimPerCtg=40,signal=0.8,seed=0):
    """
    Categories are separated along a single known direction (info['direction']); all other directions are
    isotropic nuisance noise. The single optimal filter is +/- that direction.
    """
    rng=np.random.default_rng(seed)
    yCtgInd=_labels(nCtg,nStimPerCtg)
    Y=np.linspace(-1,1,nCtg)

    basis,_=np.linalg.qr(rng.standard_normal((nPix,nPix)))
    # direction orthogonal to the constant vector so contrast normalization preserves it
    d=basis[:,0]-basis[:,0].mean()
    d=d/np.linalg.norm(d)

    noise=rng.standard_normal((nPix,len(yCtgInd)))
    noise=noise-np.outer(d,d@noise)
    stimuli=signal*np.sqrt(nPix)*np.outer(d,Y[yCtgInd]) + noise*(1-signal)
    x=filt.X(ndim=1,n=nPix,totS=1)
    return x,contrast_normalize(stimuli),yCtgInd,Y,{'direction':d}


def sine_frequency(nPix=32,freqs=(2,4,6),nStimPerCtg=30,seed=0):
    """
    Random-phase sinusoids whose frequency (cycles per stimulus) is the latent variable.
    Optimal filters are quadrature pairs at the category frequencies; useful for fourier-domain learning.
    """
    rng=np.random.default_rng(seed)
    nCtg=len(freqs)
    yCtgInd=_labels(nCtg,nStimPerCtg)
    Y=np.asarray(freqs,dtype=float)

    t=np.arange(nPix)/nPix
    phase=rng.uniform(0,2*np.pi,len(yCtgInd))
    stimuli=np.cos(2*np.pi*Y[yCtgInd][None,:]*t[:,None] + phase[None,:])
    stimuli=stimuli + 0.05*rng.standard_normal(stimuli.shape)
    x=filt.X(ndim=1,n=nPix,totS=1)
    return x,contrast_normalize(stimuli),yCtgInd,Y,{'t':t}


def binocular_shift(nPixEye=16,disparities=(-2,0,2),nStimPerCtg=40,smooth=2.0,seed=0):
    """
    1D binocular stimuli: a smoothed random texture seen by the left eye, and by the right eye shifted
    by the category's disparity (pixels). Eyes are concatenated [left; right], so nSplit=2.
    """
    rng=np.random.default_rng(seed)
    nCtg=len(disparities)
    yCtgInd=_labels(nCtg,nStimPerCtg)
    Y=np.asarray(disparities,dtype=float)

    pad=int(np.max(np.abs(Y)))+1
    n=nPixEye+2*pad
    k=np.exp(-0.5*(np.arange(-3*smooth,3*smooth+1)/smooth)**2)
    stimuli=np.empty((2*nPixEye,len(yCtgInd)))
    for i,c in enumerate(yCtgInd):
        tex=np.convolve(rng.standard_normal(n),k,mode='same')
        dsp=int(Y[c])
        stimuli[:nPixEye,i]=tex[pad:pad+nPixEye]
        stimuli[nPixEye:,i]=tex[pad+dsp:pad+dsp+nPixEye]
    x=filt.X(ndim=1,n=2*nPixEye,totS=1)
    return x,contrast_normalize(stimuli),yCtgInd,Y,{'nSplit':2,'nPixEye':nPixEye}


def image_orientation(size=(8,8),angles=(0,45,90,135),nStimPerCtg=25,freq=2,seed=0):
    """
    2D random-phase gratings whose orientation (degrees) is the latent variable. Tests 2D stimulus handling.
    """
    rng=np.random.default_rng(seed)
    nCtg=len(angles)
    yCtgInd=_labels(nCtg,nStimPerCtg)
    Y=np.asarray(angles,dtype=float)

    yy,xx=np.meshgrid(np.arange(size[0])/size[0],np.arange(size[1])/size[1],indexing='ij')
    stimuli=np.empty(tuple(size)+(len(yCtgInd),))
    for i,c in enumerate(yCtgInd):
        th=np.deg2rad(Y[c])
        u=np.cos(th)*xx+np.sin(th)*yy
        stimuli[...,i]=np.cos(2*np.pi*freq*u+rng.uniform(0,2*np.pi)) + 0.05*rng.standard_normal(size)
    x=filt.X(n=tuple(size),totS=1)
    return x,contrast_normalize(stimuli),yCtgInd,Y,{}


def unequal_counts(nPix=12,counts=(10,30,20),seed=0):
    """
    Like gaussian_ctg but with a different number of stimuli per category (non-flat prior, padding in Stim),
    and 1-based labels as produced by matlab.
    """
    rng=np.random.default_rng(seed)
    nCtg=len(counts)
    yCtgInd=_labels(nCtg,counts)
    Y=np.arange(nCtg,dtype=float)
    stimuli=rng.standard_normal((nPix,len(yCtgInd)))
    stimuli[0]+=2*Y[yCtgInd]
    x=filt.X(ndim=1,n=nPix,totS=1)
    return x,contrast_normalize(stimuli),yCtgInd+1,Y,{'counts':np.asarray(counts)}
