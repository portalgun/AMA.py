"""
Stim.load: burgelab .mat files, .npz files, 2-D stimuli (matlab column-major pixel order), renamed variables, and
continuous latent values.
"""
import numpy as np
import pytest
from scipy.io import savemat

import ama
import stimuli as ts


def test_burgelab_mat(reference_mat):
    from conftest import REFERENCE_MAT
    st=ama.Stim.load(REFERENCE_MAT)
    D=reference_mat
    assert st.nCtg==len(np.unique(D['ctgInd'])) and st.dims==(D['s'].shape[0],)
    assert np.allclose(np.sort(np.asarray(st.Y)),np.sort(np.ravel(D['X'])))


def test_2d_mat_is_column_major(tmp_path):
    rng=np.random.default_rng(0)
    imgs=ama.contrast_normalize(rng.standard_normal((4,6,30)))             # [ rows x cols x nStim ]
    ci=np.repeat([1,2,3],10)
    fname=str(tmp_path/'img.mat')
    savemat(fname,{'s':np.reshape(imgs,(24,30),order='F'),'ctgInd':ci[:,None],'X':np.array([[0.,1.,2.]])})
    st=ama.Stim.load(fname,dims=(4,6))
    assert st.dims==(4,6) and st.nCtg==3
    first=np.reshape(np.asarray(st.val[:,0,0]),(4,6))                      # category 1, first stimulus
    assert np.allclose(first,imgs[:,:,0])


def test_npz_with_keys_and_continuous_values(tmp_path):
    x,s,y,_=ts.continuous_gaussian(nStim=120)
    fname=str(tmp_path/'c.npz')
    np.savez(fname,stimuli=s,depth=y)
    st=ama.Stim.load(fname,keys={'s':'stimuli','y':'depth'},nBins=4)
    assert st.bContinuous and st.nCtg==4
    ref=ama.Stim.binned(x,s,y,nBins=4)
    assert np.allclose(np.asarray(st.yCtg),np.asarray(ref.yCtg)) and np.allclose(np.asarray(st.Y),np.asarray(ref.Y))


def test_errors(tmp_path):
    fname=str(tmp_path/'bad.npz')
    np.savez(fname,s=np.zeros((5,4)))
    with pytest.raises(Exception,match='no labels'):
        ama.Stim.load(fname)
    with pytest.raises(Exception,match='dims'):
        ama.Stim.load(fname,dims=(2,2))
    with pytest.raises(Exception,match='.mat and .npz'):
        ama.Stim.load(str(tmp_path/'x.txt'))


def test_one_category_with_several_latent_dimensions():
    """Y [ 1 x nDim ] is one category's point, not a row vector of levels"""
    x,s,ci,Y,_=ts.gaussian_ctg()
    keep=np.asarray(ci)==np.asarray(ci)[0]
    st=ama.Stim(x,np.asarray(s)[:,keep],np.asarray(ci)[keep],np.array([[1.,2.]]))
    assert np.asarray(st.Y).shape==(1,2)
