# (c) 2019-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
import numpy as np
import tensorflow as tf
import os
import logging
import re
import matplotlib.pyplot as plt
import io
import time

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def calc_imcrop(imsize, nlevels, mp_size, ksize=None, nconv=(2, 2, 2)):
    """Calculates the size of the deepest image (deep_im) and output image (out_im) in the CNN"""
    if ksize is None:
        ksize = np.repeat(3, len(mp_size))
    kdrop = np.asarray(ksize, dtype=np.uint16) - 1
    deep_im = np.empty(shape=len(mp_size), dtype=np.uint16)
    out_im = np.empty(shape=len(mp_size), dtype=np.uint16)
    for ndim, mpdim in enumerate(mp_size):
        asum = 0
        asum2 = 0
        ds = mpdim
        for i in range(nlevels):
            asum += ds ** i
        for j in range(nlevels - 1):
            asum2 += ds ** j
        deep_im[ndim] = (imsize[ndim] - kdrop[ndim] * nconv[ndim] * asum) / (ds ** (nlevels - 1))
        out_im[ndim] = deep_im[ndim] * ds ** (nlevels - 1) - kdrop[ndim] * nconv[ndim] * asum2
        if deep_im[ndim] * ds ** (nlevels - 1) < kdrop[ndim] * nconv[ndim] * asum2:
            raise Exception("The output patch is smaller than 0, check input parameters. Consider lowering the number of levels")
    return out_im, deep_im


def calc_imsize_min(deep_im, nlevels, mp_size, ksize=None, nconv=(2, 2, 2)):
    "Calculates the imsize required for a given size in the image at the deepest layer"
    if ksize is None:
        ksize = np.repeat(3, len(mp_size))
    kdrop = np.asarray(ksize, dtype=np.uint16) - 1
    imsize = np.empty(shape=len(mp_size), dtype=np.uint16)
    for ndim, mpdim in enumerate(mp_size):
        asum = 0
        ds = mpdim
        for i in range(nlevels):
            asum += ds ** i
        imsize[ndim] = (ds ** (nlevels - 1)) * deep_im[ndim] + kdrop[ndim] * nconv[ndim] * asum
    return imsize


def calc_layercrop(level, nlevels, mp_size, ksize=None, nconv=(2, 2, 2)):
    "Calculates the crop required for a layer to be concatenated in the decoder part when using valid padding"
    nsteps = nlevels - level - 1
    if ksize is None:
        ksize = np.repeat(3, len(mp_size))
    kdrop = np.asarray(ksize, dtype=np.uint16) - 1
    dcrop = np.empty(shape=len(mp_size), dtype=np.uint16)
    for ndim, mpdim in enumerate(mp_size):
        ds = mpdim
        asum = 0
        for i in range(1, nsteps):
            asum += 2 * (ds ** i)
        dcrop[ndim] = int(kdrop[ndim] * nconv[ndim] * (asum + ds ** nsteps) / 2)
    return dcrop


def crop_output(vol, *args):
    ndim = len(vol.shape) - 1
    sorig = vol.shape[0:-1]
    scrop = calc_imcrop(sorig, *args)[0]
    lcrop = np.around((sorig - scrop) / 2).astype(np.int32)
    if ndim == 2:
        return vol[lcrop[0]:-lcrop[0], lcrop[1]:-lcrop[1], :]
    elif ndim == 3:
        return vol[lcrop[0]:-lcrop[0] or None, lcrop[1]:-lcrop[1], lcrop[2]:-lcrop[2]]
    else:
        raise Exception('Function need to be adapted for this dimensionality')







def get_lastmodel(model_path, mode='best'):
    if mode == 'best':
        fname = 'model_best.tf'
    elif mode == 'last':
        fname = 'model_last.tf'
        # fname = 'model.tf'
    else:
        raise Exception("Mode unknown")
    if os.path.exists(model_path):
        last_model = os.path.join(os.path.split(model_path)[0], fname)
        with open(model_path) as f:
            fvals = f.read()
            if mode == 'best':
                start_epoch = int(re.search("epoch (.*) --", fvals).group(1))
                best_val = float(re.search(" -- value (.*)", fvals).group(1))
            else:
                start_epoch = int(fvals)
                best_val = -np.inf
        logger.info("Model found: {}\nLoading from epoch {}".format(last_model, start_epoch))
    else:
        start_epoch = 0
        best_val = None
        last_model = None
        logger.info("No MINTIF model found, starting from scratch")
    return last_model, start_epoch, best_val


def image_augmentation(augm_dict, x, y=None):
    np.random.seed(15)

    if 'flip' in augm_dict and augm_dict['flip']:
        if np.random.uniform() < augm_dict['flip']:
            x = x[::-1, ...]
            y = y if y is None else y[::-1, ...]
        if np.random.uniform() < augm_dict['flip']:
            x = x[:, ::-1, ...]
            y = y if y is None else y[:, ::-1, ...]
    if 'zoom' in augm_dict and augm_dict['zoom']:
        logging.warning("not implemented")
        pass
    if 'rotate' in augm_dict and augm_dict['rotate']:
        logging.warning("not implemented")
        pass
    return x, y


def dataset_summary(dataset, log_dir, **kwargs):
    itime = time.time()
    keys = ['train', 'val', 'test'] if len(dataset) == 3 else ['test']
    for k, data in zip(keys, dataset):
        data_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'data_vis', k))
        with data_writer.as_default():
            input_summary(data, **kwargs)
    logger.debug("Input visualization took {} seconds".format(time.time() - itime))


def image_grid(X, nims=16):
    ncols = 4
    nrows = int(np.around(np.ceil(nims / ncols)))
    fig = plt.figure(figsize=(10, 10))
    for i, x in enumerate(X):
        # In case we have a tf dataset
        if i == nims:
            break
        plt.subplot(nrows, ncols, i + 1)
        plt.grid(False)
        xaux = x[x.shape[0] // 2] if x.ndim == 3 else x
        plt.imshow(xaux)
    return fig


def input_summary(dataset, nims=16, model=None, maxfigs=4):
    def plot_images(x, y, pred, nfig, aux_title=''):
        for cx in range(x.shape[-1]):
            fig = image_grid(x[..., cx], nims)
            tf.summary.image("Input channel " + str(cx) + aux_title, plot_to_image(fig), step=nfig)
        for cy in range(y.shape[-1]):
            fig = image_grid(y[..., cy], nims)
            tf.summary.image("Output label " + str(cy) + aux_title, plot_to_image(fig), step=nfig)
        if model is not None:
            for cp in range(pred.shape[-1]):
                fig = image_grid(pred[..., cp], nims)
                tf.summary.image("Predicted label " + str(cp) + aux_title, plot_to_image(fig), step=nfig)

    for nfig, (x_aux, y_aux) in enumerate(dataset.batch(nims).take(maxfigs)):
        x = x_aux[:, 0, ...]
        y = y_aux[:, 0, ...]
        pred = np.zeros(shape=([x.shape[0]] + model.output.shape[1:]))
        for b in range(x.shape[0]):
            xb = tf.expand_dims(tf.gather(x, b), 0)
            pred_b = model(xb, training=False)
            if len(pred_b) > 1:
                pred_b = pred_b[0]
            pred[b] = pred_b.numpy()
        if x.ndim == 5:
            plot_images(
                x[:, x.shape[1] // 2, ...],
                y[:, y.shape[1] // 2, ...],
                pred[:, pred.shape[1] // 2, ...],
                nfig=nfig, aux_title='_xy')
            plot_images(
                x[:, :, x.shape[2] // 2, ...],
                y[:, :, y.shape[2] // 2, ...],
                pred[:, :, pred.shape[2] // 2, ...],
                nfig=nfig, aux_title='_xz')
            plot_images(
                x[:, :, :, x.shape[3] // 2, ...],
                y[:, :, :, y.shape[3] // 2, ...],
                pred[:, :, :, pred.shape[3] // 2, ...],
                nfig=nfig, aux_title='_yz')
        else:
            plot_images(x, y, pred, nfig=nfig)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call.
  from: https://www.tensorflow.org/tensorboard/r2/image_summaries"""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def check_weights_nan(weights, grads):
    [print("Layer " + str(n) + " name: " + w.name + "-- isnan: " + str(np.isnan(g).any())) for n, (w, g) in
     enumerate(zip(weights, grads))]


def get_overlap(patch_size, mp_size, nlevels, ksize=None, nconv=None):
    """Calculates the minimum overlap between patches so that the output from the neural network can be stitch to
    reconstruct the original image
            Parameters
            ----------
            patch_size: tuple
                a tuple with the size of the patches into which the image will be decomposed
            mp_size: tuple
                a tuple with the downsampling factor applied when changing the level in the network for each dimension
            nlevels: int
                number of downsampling steps in the network
            ksize: tuple
                kernel size applied in the convolutional layers for each dimension
            nconv: tuple
                number of convolutional layers at each layer of the network for each dimension
            Returns
            -------
            list
                a list with the amount of voxel which overlap among patches for each dimension
    """
    nconv = nconv or [2] * len(mp_size)
    ksize = ksize or [3] * len(mp_size)
    kdrop = np.asarray(ksize, dtype=np.uint16) - 1
    deep_im = np.empty(shape=len(mp_size), dtype=np.uint16)
    out_im = np.empty(shape=len(mp_size), dtype=np.uint16)
    for ndim, mpdim in enumerate(mp_size):
        asum = 0
        asum2 = 0
        ds = mpdim
        for i in range(nlevels):
            asum += ds ** i
        for j in range(nlevels - 1):
            asum2 += ds ** j
        deep_im[ndim] = (patch_size[ndim] - kdrop[ndim] * nconv[ndim] * asum) / (ds ** (nlevels - 1))
        out_im[ndim] = deep_im[ndim] * ds ** (nlevels - 1) - kdrop[ndim] * nconv[ndim] * asum2
        if deep_im[ndim] * ds ** (nlevels - 1) < kdrop[ndim] * nconv[ndim] * asum2:
            raise Exception("The output patch is smaller than 0, check input parameters")
    overlap = np.asarray(patch_size, dtype=np.uint16) - out_im
    return overlap


def get_patches_indices(image_shape, patchsize_patch, voxelsize_im, voxelsize_patch, overlap_patch):
    """Create an array of indices to decompose the image. Note that every parameters may have to dimensions.
    Suffix "_patch" means that it is in the patch dimensions defined by the user, and suffix "_im" means that it is in
    the original image dimensions, which should be accessible from the image metadata.
            Parameters
            ----------
            image_shape: tuple
                the shape of the image we want to decompose
            patchsize_patch: tuple
                a tuple with the size of the patches (in the patch frame) into which the image will be decomposed
            voxelsize_im
                the resolution of the original image, should be accessible from the image metadata
            voxelsize_patch
                the resolution of the patch, should be defined by the user
            overlap_patch
                the amount of voxels which overlap in the patch frame
            Returns
            -------
            list of lists
                a list of the indices we employ to decompose the image
                the first dimension encodes the id of the patch
                the second dimension encodes the indices as [indx1, indx2, indy1, indy2, indz1, indz2]
                For example, if the entry 5 is [0, 640, 640, 1280, 5, 6],
                it means that we take a patch as image[0:640, 640:1280, 5:6]
    """
    # We first transform from the patch dimension to the image dimensions, as we will need the image dimensions for the
    # patch indices which will be read from the images_documentation themselves.
    patchsize_im = (np.round(patchsize_patch * voxelsize_patch / voxelsize_im)).astype(np.uint16)
    overlap_im = (np.round(overlap_patch * voxelsize_patch / voxelsize_im)).astype(np.uint16)
    padsize_patch = (np.round(overlap_patch / 2)).astype(np.uint16)
    padsize_im = (np.round(padsize_patch * voxelsize_patch / voxelsize_im)).astype(np.uint16)

    def calc1Dind(ldim, ndim):
        """Calculate the indices for the selected dimension
            Parameters
            ----------
            ndim: int
                Dimension index
            ldim: int
                Number of voxels of the image in that dimension
            Returns
            -------
            list
                a list of the indices for the selected dimension
        """
        blim = False
        maxub = ldim + padsize_im[ndim]
        minub = -1 * padsize_im[ndim]
        lb = minub
        ub = lb + patchsize_im[ndim]
        lInd = [lb, ub]
        while not blim:
            lb = ub - overlap_im[ndim]
            ub = lb + patchsize_im[ndim]
            if ub == ldim:
                blim = True
            elif ub > maxub:
                ub = maxub
                lb = maxub - patchsize_im[ndim]
                blim = True
            lInd.append([lb, ub])
        return lInd

    ind_aux = []
    for im_ndim, im_ldim in enumerate(image_shape):
        ind_aux.append(calc1Dind(im_ldim, im_ndim))
    list_inds = []
    for lb1, ub1 in ind_aux[0]:
        for lb2, ub2 in ind_aux[1]:
            for lb3, ub3 in ind_aux[2]:
                list_inds.append([lb1, ub1, lb2, ub2, lb3, ub3])
    return list_inds


def get_modelload(model_dir):
    if os.path.isfile(os.path.join(model_dir, 'model_best.h5')):
        model_file = os.path.join(model_dir, 'model_best.h5')
    else:
        model_file = os.path.join(model_dir, 'model_best.tf')
    return model_file


def split_batch(data_batch):
    if (len(data_batch) == 2) and not tf.is_tensor(data_batch):
        if isinstance(data_batch, dict):
            x, y = data_batch['main']
        else:
            x, y = data_batch
    else:
        x = data_batch
        y = None
    return x, y


def save_layers_img(x, model_vislayers, step, save_dir):
    yl = model_vislayers(x)
    for n, l in enumerate(yl):
        laux = l[0] if l.ndim == 4 else l[0, ..., 0]
        plt.imshow(tf.reduce_mean(laux, axis=-1))
        name = os.path.join(save_dir, str(step) + "_" + str(n) + model_vislayers[n].name)
        plt.savefig(name)


def model_basicstats(model, l=0):
    xarray = model.trainable_variables[l].numpy()
    print("Max: {}\nMin: {}\nMean: {}\nStd: {}".format(
        np.max(xarray), np.min(xarray), np.mean(xarray), np.std(xarray)
    ))
