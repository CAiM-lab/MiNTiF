# (c) 2018-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
import numpy as np
from skimage.draw import ellipsoid
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import distance_matrix
from  utils.im_processing import gndkernel
from scipy.optimize import linear_sum_assignment
import logging

logging.getLogger(__name__)


class Spots:
    """Input and output functions with ims files as hdf5:

    Attributes:
        X_um0: coordinates in um starting at 0
        X_voxel0: coordinates in voxels starting at 0
        X_um_rw: coordinates in um in the rw frame
        X_voxel_rw: coordinates in voxels in the rw frame
        radius_um: radius of the spots in um
        radius_voxel: radius of the spots in voxels
        imExtends: physical extends of the image
        voxelSize: voxel size of the image
        mask: tissue boundaries
        nSpots: number of spots
        NN dist: nearest neighbour distances

    """""

    def __init__(self, X_um_rw, imExtends=None, voxelSize=None, imsize=None, mask=None, radius_voxel=None,
                 radius_um=None, save_dir=None, apply_mask=True):
        self.save_dir = save_dir
        self.X_um_rw = X_um_rw
        self.imsize = imsize
        self.voxelSize = voxelSize
        self.ndims = len(self.imsize)
        self.mask = mask
        if imExtends is None:
            self.imExtends = np.array([[0]*self.ndims, self.imsize])
        else:
            self.imExtends = imExtends
        if radius_um is not None and radius_voxel is not None:
            raise Exception('The parameter cannot be given in two different units')
        elif radius_um is not None:
            self.radius_um = radius_um
            self.radius_voxel = self.radius_um / np.array(self.voxelSize)
        elif radius_voxel is not None:
            self.radius_voxel = radius_voxel if type(radius_voxel) is not int else np.repeat(radius_voxel, self.ndims)
            self.radius_um = self.radius_voxel * self.voxelSize
        self.X_um0 = self.convert_Xumrw2Xum0(self.X_um_rw)
        self.X_voxel_rw = self.convert_um2voxel(self.X_um_rw)
        # self.X_voxel0 = ((self.X_um_rw - self.imExtends[0]) / self.voxelSize + 0.5).astype(np.int64)
        self.X_voxel0 = self.convert_um2voxel(self.X_um0)
        # Delete points out of extends

        keep_pos = np.all((self.X_um_rw >= self.imExtends[0],
                           self.X_um_rw <= self.imExtends[1],
                           self.X_voxel0 >= 0,
                           self.X_voxel0 < self.imsize,
                           self.X_um0 < (self.imExtends[1] - self.imExtends[0])),
                          axis=0).all(axis=1)
        self.apply_limits(keep_pos)
        if self.mask is not None and apply_mask:
            keep_pos2 = [c for c, x in enumerate(self.X_voxel0) if self.mask[x[2], x[1], x[0]]]
            self.apply_limits(keep_pos2)

        self.nSpots = self.X_um_rw.shape[0]

    def convert_um2voxel(self, X):
        return np.around(X / self.voxelSize).astype(np.int64)

    def convert_Xumrw2Xum0(self, X):
        return X - self.imExtends[0]

    def convert_Xvoxel02Xum0(self, X):
        return X * self.voxelSize

    def convert_Xum02Xumrw(self, X):
        return X + self.imExtends[0]

    def convert_Xvoxel02Xumrw(self, X):
        return self.convert_Xum02Xumrw(self.convert_Xvoxel02Xum0(X))

    def apply_limits(self, keep_pos, hard_const=True):
        self.X_um_rw = self.X_um_rw[keep_pos]
        self.X_um0 = self.X_um0[keep_pos]
        self.X_voxel_rw = self.X_voxel_rw[keep_pos]
        self.X_voxel0 = self.X_voxel0[keep_pos]
        if hard_const:
            assert ((self.X_um_rw.min(axis=0) >= self.imExtends[0]).all())
            assert ((self.X_um_rw.max(axis=0) <= self.imExtends[1]).all())
            assert ((self.X_um0.max(axis=0) < (self.imExtends[1] - self.imExtends[0])).all())
            assert (self.X_um0.min() >= 0)
            assert ((self.X_voxel0.max(axis=0) < self.imsize).all())

    def set_newlimits_vox(self, voxExtends):
        # voxExtends: upz, lowz, upy, lowy, upx, lowx
        uExtends = [voxExtends[5], voxExtends[3], voxExtends[1]]
        lExtends = [voxExtends[4], voxExtends[2], voxExtends[0]]
        # uExtends_aux = np.array([x * self.voxelSize[c] for c, x in enumerate(uExtends)])
        # lExtends_aux = [x * self.voxelSize[c] for c, x in enumerate(lExtends)]
        self.imExtends = [lExtends, uExtends]
        keep_pos = np.all((self.X_voxel0 > self.imExtends[0],
                           self.X_voxel0 < self.imExtends[1]),
                          axis=0).all(axis=1)
        self.apply_limits(keep_pos, hard_const=False)

    def points2image(self, limInd=None):
        imsize = self.imsize if limInd is None else [limInd[5] - limInd[4], limInd[3] - limInd[2],
                                                     limInd[1] - limInd[0]]
        vol = np.zeros(shape=imsize, dtype=np.uint8)
        if limInd:
            limlow = [limInd[4], limInd[2], limInd[0]]
            limhigh = [limInd[5], limInd[3], limInd[1]]
            keep_pos = np.all((self.X_voxel0 > limlow, self.X_voxel0 < limhigh), axis=0).all(axis=1)
            X_crop = self.X_voxel0[keep_pos] - limlow
        else:
            X_crop = self.X_voxel0
        for p in range(X_crop.shape[0]):
            coor = X_crop[p]
            vol[coor[0], coor[1], coor[2]] = 255.0

    def spots2image(self, limInd=None):
        # volPoints = self.points2image(limInd=limInd)
        # bstr = ellipsoid(self.radius_voxel[0], self.radius_voxel[1], self.radius_voxel[2])
        # return binary_dilation(volPoints > 0, selem=bstr)
        imsize = self.imsize if limInd is None else [limInd[5] - limInd[4], limInd[3] - limInd[2],
                                                     limInd[1] - limInd[0]]
        vol = np.zeros(shape=imsize, dtype=np.uint8)
        if limInd:
            limlow = [limInd[4], limInd[2], limInd[0]]
            limhigh = [limInd[5], limInd[3], limInd[1]]
            keep_pos = np.all((self.X_voxel0 > limlow, self.X_voxel0 < limhigh), axis=0).all(axis=1)
            X_crop = self.X_voxel0[keep_pos] - limlow
        else:
            X_crop = self.X_voxel0
        bstr = ellipsoid(self.radius_voxel[0], self.radius_voxel[1], self.radius_voxel[2])
        elshape = np.array(bstr.shape)
        for p in range(X_crop.shape[0]):
            coor = X_crop[p]
            clow_aux = np.around(coor - np.floor(elshape / 2)).astype(np.int32)
            clow = np.max([clow_aux, np.array([0, 0, 0])], axis=0)
            chigh_aux = np.around(coor + np.ceil(elshape / 2)).astype(np.int32)
            chigh = np.min([chigh_aux, vol.shape], axis=0)
            blow = np.max([np.array([0, 0, 0]) - clow_aux, np.array([0, 0, 0])], axis=0)
            bhigh = np.max([chigh_aux - vol.shape, np.array([0, 0, 0])], axis=0)
            vol[clow[0]:chigh[0], clow[1]:chigh[1], clow[2]:chigh[2]] = \
                bstr[blow[0]:-bhigh[0] or None, blow[1]:-bhigh[1] or None, blow[2]:-bhigh[2] or None]
        return vol

    def gaus_spots2image(self, ksigma=None, limInd=None):
        #todo: In 2d we need to transpose the image, check if it's the case for 3D
        if ksigma is None:
            ksigma = self.radius_voxel / 2
        imsize = self.imsize if limInd is None else [limInd[5] - limInd[4], limInd[3] - limInd[2],
                                                     limInd[1] - limInd[0]]
        vol = np.zeros(shape=imsize, dtype=np.uint16)
        if limInd:
            limlow = [limInd[4], limInd[2], limInd[0]]
            limhigh = [limInd[5], limInd[3], limInd[1]]
            keep_pos = np.all((self.X_voxel0 > limlow, self.X_voxel0 < limhigh), axis=0).all(axis=1)
            X_crop = self.X_voxel0[keep_pos] - limlow
        else:
            X_crop = self.X_voxel0
        zero_ar = np.array([0] * self.ndims)
        gstr = np.around(gndkernel(ksigma=ksigma, klen=np.around(10 * ksigma).astype(np.int16), ndims=self.ndims) * 255).astype(np.uint8)
        gshape = np.array(gstr.shape)
        for p in range(X_crop.shape[0]):
            coor = X_crop[p]
            clow_aux = np.around(coor - np.floor(gshape / 2)).astype(np.int32)
            clow = np.max([clow_aux, zero_ar], axis=0)
            chigh_aux = np.around(coor + np.ceil(gshape / 2)).astype(np.int32)
            chigh = np.min([chigh_aux, vol.shape], axis=0)
            blow = np.max([zero_ar - clow_aux, zero_ar], axis=0)
            bhigh = np.max([chigh_aux - vol.shape, zero_ar], axis=0)
            if self.ndims == 3:
                vol[clow[0]:chigh[0], clow[1]:chigh[1], clow[2]:chigh[2]] = \
                    vol[clow[0]:chigh[0], clow[1]:chigh[1], clow[2]:chigh[2]] + \
                    gstr[blow[0]:-bhigh[0] or None, blow[1]:-bhigh[1] or None, blow[2]:-bhigh[2] or None]
            elif self.ndims == 2:
                vol[clow[0]:chigh[0], clow[1]:chigh[1]] = \
                    vol[clow[0]:chigh[0], clow[1]:chigh[1]] + \
                    gstr[blow[0]:-bhigh[0] or None, blow[1]:-bhigh[1] or None]
            else:
                raise Exception("Function not implemented for this dimensionality")
        return np.around(vol / (vol.max() / 255)).astype(np.uint8)

    def blurPoints(self, vol=None, limInd=None):
        if vol is None:
            vol = self.pts2image(limInd=limInd)
            imsize = self.imsize if limInd is None else [limInd[5] - limInd[4], limInd[3] - limInd[2],
                                                         limInd[1] - limInd[0]]

        else:
            imsize = vol.shape
        outVol = np.zeros(shape=imsize, dtype=np.float64)
        gaussian_filter(np.around(vol * 255).astype(np.uint8), sigma=self.radius_voxel, output=outVol, mode='constant')
        return np.around(outVol).astype(np.uint8)

    @staticmethod
    def gt_match_um(Xgt, Xpred, rmatch):
        """
        Function to calculate the quality of spot detection
        :param Xgt: the ground truth coordinates
        :param Xpred: the predicted coordinates
        :param rmatch: the threshold distance below which we count a positive
        :return:
        """
        if len(Xgt) == 0:
            qmetrics = {'TP': np.nan,
                        'FN': 0,
                        'FP': len(Xpred),
                        'precision': np.nan,
                        'recall': np.nan,
                        'fscore': np.nan,
                        'derror_pos': None,
                        'derror_all': None}
        elif len(Xpred) == 0:
            qmetrics = {'TP': 0,
                        'FN': len(Xgt),
                        'FP': 0,
                        'precision': 0,
                        'recall': 0,
                        'fscore': 0,
                        'derror_pos': np.nan,
                        'derror_all': np.nan}
        else:
            # Method 1
            # tree = cKDTree(Xpred)
            # distgt, indgt = tree.query(Xgt, 5)
            # nngt = np.empty((len(Xgt),), dtype=np.intp)
            # nngt.fill(-1)
            # nnpred = np.empty((len(Xpred),), dtype=np.intp)
            # nnpred.fill(-1)
            # for j, neigh_j in enumerate(indgt):
            #     for j2, k in enumerate(neigh_j):
            #         if distgt[j][j2] <= rmatch and nnpred[k] == -1:
            #             nngt[j] = k
            #             nnpred[k] = j
            #             break
            # Hungarian algorithm
            C = distance_matrix(Xgt, Xpred)
            igt, ipred = linear_sum_assignment(C)
            nngt = np.empty((len(Xgt),), dtype=np.intp)
            nngt.fill(-1)
            nnpred = np.empty((len(Xpred),), dtype=np.intp)
            nnpred.fill(-1)
            for pgt, ppred in zip(igt, ipred):
                indval = C[pgt, ppred]
                if indval <= rmatch:
                    nngt[pgt] = indval
                    nnpred[ppred] = indval

            tp = sum(nnpred > -1)
            fn = sum(nngt == -1)
            fp = sum(nnpred == -1)
            derror_all = C.min(axis=0)
            qmetrics = {'TP': tp,
                        'FN': fn,
                        'FP': fp,
                        'precision': tp / (tp + fp),
                        'recall': tp / (tp + fn),
                        'fscore': 2 * tp / (2 * tp + fp + fn),
                        'derror_pos': C[igt, ipred],
                        'derror_all': derror_all}
        return qmetrics
