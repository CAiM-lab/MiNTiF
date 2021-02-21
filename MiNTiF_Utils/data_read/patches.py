# (c) 2018-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
import numpy as np
import os
from  data_read.imarisfiles import ImarisFiles
from skimage.transform import resize
from  utils.im_processing import interp3
from  data_read.spots import Spots
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Patch:
    def __init__(self, filename, out_psize=None, overlap_vox=None, padsize_vox=None, patchsize_vox=None,
                 lChannels=None, lChannelsGT=None, padtype='constant', z_minstep=10):
        self.filename = filename

        def checkInput(pin, fillz=1):
            pout = pin
            try:
                vaux = pout[0]
            except TypeError:
                pout = np.repeat(pout, 3)
            if len(pout) == 1:
                pout = np.repeat(pout, 3)
            elif len(pout) == 2:
                pout = np.array([fillz, pin[0], pin[1]])
            else:
                pout = np.array(pin)
            return pout

        # in_parameters
        self.filetype = os.path.splitext(filename)[1]
        self.patchsize_vox = checkInput(patchsize_vox)
        if self.filetype == '.ims':
            self.imfile = ImarisFiles(self.filename)
            self.in_imsize = np.flip(self.imfile.imsize, 0)
            self.in_psize = np.flip(self.imfile.voxelSize, 0)
            if len(self.patchsize_vox) == 2:
                self.in_psize = self.in_psize[1::]
        else:
            raise Exception('File extension not recognized')
        if not out_psize:
            out_psize = self.in_psize

        self.padtype = padtype
        self.lChannels = lChannels
        self.lChannelsGT = lChannelsGT
        self.out_psize = checkInput(out_psize)
        self.overlap_vox = checkInput(overlap_vox, 0)
        self.overlap_in = np.round(self.overlap_vox * self.out_psize / self.in_psize).astype(np.uint16)
        self.z_minstep = z_minstep
        if padsize_vox is None:  # If overlap has been calculated to correctly stitch the sample, padsize should be the same
            self.padsize_vox = (np.round(self.overlap_vox / 2)).astype(np.uint16)
        else:
            self.padsize_vox = checkInput(padsize_vox)
        self.padsize_in = np.round(self.padsize_vox * self.out_psize / self.in_psize).astype(np.uint16)
        self.patchsize_in = np.round(self.patchsize_vox * self.out_psize / self.in_psize).astype(np.uint16)
        self.patchsize_in = [1 if x == 1 else self.patchsize_in[c] for c, x in
                             enumerate(self.patchsize_vox)]  # to avoid problems in 2D

        # Corrections for z
        self.patchsize_in[0] = max(self.patchsize_in[0], 1)
        if self.overlap_in[0] == self.patchsize_in[0]:
            Warning("Overlap equals patchsize in z. Decreasing overlap... check if the result makes sense!")
            self.overlap_in[0] -= 1

        schannels = set([x.lower() for x in self.lChannelsGT]) if not (self.lChannelsGT is None) else set()
        self.npatches = np.ceil(self.in_imsize / (self.patchsize_in - self.padsize_in * 2)).astype(np.uint16)
        self.patchsize_crop = self.patchsize_vox - self.padsize_vox * 2

    def getAnnotated_Ind(self):
        zInd = np.empty(shape=[self.in_imsize[0], len(self.lChannelsGT)])
        for nc, channel in enumerate(self.lChannelsGT):
            vol = self.imfile.getVolume((channel,))
            zInd[:, nc] = vol[..., 0].sum(1).sum(1) > 0
        if (zInd == 1).all():  # all slices are annotated
            fInd = np.arange(self.z_minstep, self.in_imsize[0], self.z_minstep)
        else:
            fInd = np.where(zInd.all(1))[0]
        return fInd

    def getPatchIndexes(self, zInd=None, do_zInd=True):
        # vIndex = [lbx, ubx, lby, uby, lbz, ubz]
        if do_zInd:
            zInd = self.getAnnotated_Ind()

        def calc1Dind(ldim, ndim):
            lInd = []
            if ndim == 0 and zInd is not None:
                for zz in zInd:
                    if self.patchsize_in[ndim] % 2:
                        lb = zz - self.patchsize_in[ndim] // 2
                        ub = zz + self.patchsize_in[ndim] // 2 + 1
                    else:
                        lb = zz - self.patchsize_in[ndim] // 2
                        ub = zz + self.patchsize_in[ndim] // 2
                    lInd.append([lb, ub])
            else:
                blim = False
                maxub = ldim + self.padsize_in[ndim]
                minub = -1 * self.padsize_in[ndim]
                lb = minub
                ub = lb + self.patchsize_in[ndim]
                lInd.append([lb, ub])
                if ub >= maxub:
                    blim = True
                while not blim:
                    lb = ub - self.overlap_in[ndim]
                    ub = lb + self.patchsize_in[ndim]
                    if ub == ldim:
                        blim = True
                    elif ub > maxub:
                        ub = maxub
                        lb = maxub - self.patchsize_in[ndim]
                        blim = True
                    lInd.append([lb, ub])
            return lInd

        ind_aux = []
        for ndim, ldim in enumerate(self.in_imsize):
            ind_aux.append(calc1Dind(ldim, ndim))
        vIndex = []
        for lb1, ub1 in ind_aux[0]:
            for lb2, ub2 in ind_aux[1]:
                for lb3, ub3 in ind_aux[2]:
                    vIndex.append([lb1, ub1, lb2, ub2, lb3, ub3])
        return vIndex

    def getPatch_spots(self, channel=None, ind=None, spot_rad=None):
        channel = self.lChannelsGT if channel is None else channel
        cspot = self.imfile.getSpotsObj(channel, spot_rad=spot_rad)
        cspot.set_newlimits_vox(ind)
        return cspot.X_um0

    def transform_spotframe_um0(self, X0, ind):
        if len(X0) == 0:
            return X0
        else:
            ind_aux = [max(x, 0) for x in [ind[0], ind[2], ind[4]]]
            Xf_aux = X0 * self.out_psize + ind_aux * self.in_psize
            Xf = Xf_aux[:, ::-1]
            return Xf

    def spot_gtmatch(self, X0, channel=None, ind=None, spot_rad=None):
        Xgt_um = self.getPatch_spots(channel=channel, ind=ind, spot_rad=spot_rad)
        Xf_um = self.transform_spotframe_um0(X0, ind)
        metrics = Spots.gt_match_um(Xgt=Xgt_um, Xpred=Xf_um, rmatch=spot_rad)
        return metrics

    def transform_spotframe_um(self, X0, ind=None):
        Xf_um0 = self.transform_spotframe_um0(X0, ind)
        Xf_um = Xf_um0 + self.imfile.imExtends[0]
        return Xf_um

    def dataset_norm(self, X):
        Xout = X.copy().astype(np.float64)
        # if (pconfig['dataset_type'] in ['synthesis']) and (Xout.shape[-1] > len(self.lChannels)):
        #     lChannels = self.lChannels + self.lChannelsGT
        # else:
        lChannels = self.lChannels
        for cnum, cname in enumerate(lChannels):
            Xout[..., cnum] = (Xout[..., cnum] - self.imfile.get_stat(cname, 'mean')) / self.imfile.get_stat(cname,
                                                                                                             'std')
        return Xout

    def patch_resize(self, lPatch_orig, dim_mode=3):
        lPatch = np.empty(shape=list(self.patchsize_vox) + [lPatch_orig.shape[-1]], dtype=lPatch_orig.dtype)
        if dim_mode == 3:
            for nc in range(lPatch_orig.shape[-1]):
                lPatch[..., nc] = interp3(lPatch_orig[..., nc],
                                          [self.patchsize_vox[0], self.patchsize_vox[1], self.patchsize_vox[2]],
                                          interp_method='linear')
        elif dim_mode == 2:
            for nc in range(lPatch_orig.shape[-1]):
                for zz in range(lPatch_orig.shape[0]):
                    lPatch[zz, ..., nc] = resize(lPatch_orig[zz, ..., nc],
                                                 [self.patchsize_vox[1], self.patchsize_vox[2]], order=1,
                                                 mode='reflect', preserve_range=True)
        return lPatch

    def padded_inds(self):
        aInds = self.aInd
        # Equivalent size of the cropped patch in original resolution

        size_orig = np.round(self.patchsize_crop * self.out_psize / self.in_psize).astype(np.uint16)
        size_orig = [1 if x == 1 else size_orig[c] for c, x in enumerate(self.patchsize_vox)]  # to avoid problems in 2D

        # round to closest even
        # size_orig = np.array([np.round(x / 2.) * 2 for x in size_orig], dtype=np.uint16)
        # size_orig = size_orig if len(self.patchsize_vox) > 2 else np.array([1] + list(size_orig), dtype=np.uint16)
        pad_aux = np.array(self.patchsize_in) - size_orig
        pad_orig = pad_aux // 2
        # We compensate the left border in uneven compensation of pad
        pad_orig0 = [x // 2 + 1 if x % 2 else x // 2 for x in pad_aux]
        if self.patchsize_vox[0] == 1:
            pad_orig[0] = 0
            pad_orig0[0] = 0
        aInds_pad = [None] * len(aInds)
        for c, ind in enumerate(aInds):
            v_aux = np.empty(shape=len(pad_orig0 * 2), dtype=np.int32)
            v_aux[0::2] = np.array(ind)[0::2] + pad_orig0
            v_aux[1::2] = np.array(ind)[1::2] - pad_orig
            if min(v_aux) < 0:
                logger.warning("Indices smaller than 0 for padded indices in sample {}, patch {}".format(
                    self.filename, str(c)))
            aInds_pad[c] = v_aux
        return aInds_pad

    def ind2pos(self, ind):
        # patchsize_eff = self.patchsize_in - self.padsize_in * 2
        overlap_aux = np.array([x + 1 if (x > 0) else x for x in self.overlap_in])
        patchsize_eff = self.patchsize_in - overlap_aux

        ndims = int(len(ind) / 2)
        pos = np.empty(shape=ndims, dtype=np.uint16)
        for dim in range(ndims):
            dind = ind[dim * 2:dim * 2 + 2]
            sind = dind[0] + self.padsize_in[dim]
            # pos_aux = max((sind - 1), 0) / patchsize_eff[dim]
            if patchsize_eff[dim] == 1:
                pos_aux = sind
            else:
                pos_aux = sind / (patchsize_eff[dim] + 1)
            assert (dind[1] >= self.in_imsize[dim] or (pos_aux % 1) == 0)
            pos[dim] = pos_aux
        return pos

    def h5resize(self, ngroup, new_size=None):
        if new_size is None:
            pass  # todo: take the one from the original image
        pass

    def create_spots(self, X, spotrad_um=None):
        return Spots(X_um_rw=X, imExtends=self.imfile.imExtends, voxelSize=self.imfile.voxelSize,
                     imsize=self.imfile.imsize, radius_um=spotrad_um)
