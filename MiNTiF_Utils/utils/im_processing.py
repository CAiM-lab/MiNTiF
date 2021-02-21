# (c) 2019-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
import numpy as np
from skimage import feature
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve
from scipy.ndimage.measurements import center_of_mass, label

def normalize_data(X):
    X_out = np.zeros(shape=X.shape, dtype=np.uint8)
    for cc in range(0, X.shape[3]):
        maxval = np.percentile(X[..., cc], 99)
        for zz in range(0, X.shape[0]):
            im_aux = X[zz, ..., cc].astype(np.float32)
            imnorm_aux = im_aux / maxval * 255
            imnorm_aux[imnorm_aux > 255] = 255
            X_out[zz, :, :, cc] = imnorm_aux
    return X_out


def convert_uint8(im, norm=None):
    if not norm:
        norm = False if im.max() < 257 else True
    if im.dtype == np.uint8 or im.max() <= 1:
        return im
    elif im.dtype == np.uint16 or im.dtype == np.float16:
        if norm:
            im = np.around(im * ((2 ** 16 - 1) / im.max())).astype(np.uint16)
        if im.max() > 257:
            return (im / 257).round().astype(np.uint8)
        else:
            return np.around(im).astype(np.uint8)
    elif im.max() <= 255:
        return np.around(im).astype(np.uint8)
    else:
        # raise Exception("Add function to consider the data type of this image")
        return np.around(im*(255/im.max())).astype(np.uint8)

def get_train_test_data(X_data, Y_data, ratio):
    # randomly assign indices for training and test data
    num_data = X_data.shape[0]
    all_ind = range(num_data)
    train_ind = np.random.choice(all_ind, size=int(ratio * num_data), replace=False)
    test_ind = [i for i in all_ind if i not in train_ind]

    # generate array of training data/labels
    X_train = X_data[train_ind]
    Y_train = Y_data[train_ind]

    # generate array of test data/labels
    X_test = X_data[test_ind]
    Y_test = Y_data[test_ind]

    return (X_train, Y_train), (X_test, Y_test)



def interp3(in_vol, out_size, interp_method):
    in_size = in_vol.shape
    x1 = np.linspace(1, in_vol.shape[0], in_size[0])
    y1 = np.linspace(1, in_vol.shape[1], in_size[1])
    z1 = np.linspace(1, in_vol.shape[2], in_size[2])
    x2 = np.linspace(1, in_vol.shape[0], out_size[0])
    y2 = np.linspace(1, in_vol.shape[1], out_size[1])
    z2 = np.linspace(1, in_vol.shape[2], out_size[2])
    int_func = RegularGridInterpolator((x1, y1, z1), in_vol, method=interp_method)
    x3, y3, z3 = np.meshgrid(x2, y2, z2, indexing='ij')
    out_vol = int_func(np.array([x3, y3, z3]).T)
    return out_vol.T


def patch_resize(vol_orig, re_shape, dim_mode=3):
    vol_re = np.empty(shape=re_shape, dtype=vol_orig.dtype)
    if dim_mode == 3:
        vol_re = interp3(vol_orig, re_shape, interp_method='linear')
    # elif dim_mode==2:
    #     for nc in range(lPatch_orig.shape[-1]):
    #         for zz in range(lPatch_orig.shape[0]):
    #             lPatch[zz,..., nc] = resize(vol_orig[zz,..., nc],[self.patchsize_vox[1], self.patchsize_vox[2]], order=1, mode='reflect', preserve_range=True)
    return vol_re

def detect_spots(vol, mindist=None):
    """
    Detect spots in a density map
    :param vol:
    :param mindist:
    :return:
    """
    volg = gaussian_filter(vol, sigma=2, mode='constant')
    is_peak = feature.peak_local_max(volg, min_distance=mindist,
                                     threshold_abs=30,
                                     exclude_border=int(mindist),
                                     indices=False)
    return labels2coordinates(is_peak)

def labels2coordinates(mask):
    # Correct peaks that correspond to the same label (https://stackoverflow.com/questions/51672327/skimage-peak-local-max-finds-multiple-spots-in-close-proximity-due-to-image-impu)
    labels = label(mask)[0]
    merged_peaks = center_of_mass(mask, labels, range(1, np.max(labels) + 1))
    merged_peaks = np.array(merged_peaks)
    return merged_peaks


def g1dkernel(klen, ksigma):
    ax = np.arange(-klen // 2 + 1., klen // 2 + 1.)
    Kx = np.exp(-(ax ** 2 / (2 * (ksigma ** 2))))
    return Kx


def g3dkernel(klen, ksigma):
    if not isinstance(klen, (list, tuple, np.ndarray)):
        klen = [klen] * 3
    if not isinstance(ksigma, (list, tuple, np.ndarray)):
        ksigma = [ksigma] * 3
    K = [None] * 3
    for dd in range(3):
        K[dd] = g1dkernel(klen[dd], ksigma[dd])
    Kout = convolve(K[2][None, None, :], convolve(K[0][:, None, None], K[1][None, :, None]))
    return Kout

def gndkernel(klen, ksigma, ndims):
    """
    Generates a nD Gaussian image

    Parameters
    ----------
    klen : int or list (of int)
        length of the kernel for each of the dimensions
    ksigma : float or list (of float)
        sigma of the Gaussian for each of the dimensions
    ndims : int
        Number of dimensions where the Gaussian function is generated

    Returns
    -------

    """
    if ndims == 1:
        return g1dkernel(klen, ksigma)
    if not isinstance(klen, (list, tuple, np.ndarray)):
        klen = [klen] * ndims
    if not isinstance(ksigma, (list, tuple, np.ndarray)):
        ksigma = [ksigma] * ndims
    K = [None] * ndims
    for dd in range(ndims):
        K[dd] = g1dkernel(klen[dd], ksigma[dd])
    if ndims==3:
        Kout = convolve(K[2][None, None, :], convolve(K[0][:, None, None], K[1][None, :, None]))
    elif ndims==2:
        Kout = convolve(K[0][:, None], K[1][None, :])
    else:
        raise Exception("Function not implemented for this dimensionality")
    return Kout

