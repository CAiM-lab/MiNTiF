# (c) 2018-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel

import numpy as np
import h5py
import os
from  utils import common_utils, cnn_utils
from skimage.transform import resize
import logging
from  data_read.imarisfiles import ImarisFiles
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from time import sleep
from  data_read.patches import Patch
# import pygsheets
from  utils import im_processing

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DTFM:
    # Channels always in lower case
    channel_ids = {
        1: ['dapi'],
        2: ['endomucin', 'vessels', 'lyve1', 'emcn'],
        3: ['endoglin', 'cd105'],
        4: ['collagen', 'collagen iv', 'col iv', 'laminin'],
        5: ['cxcl12', 'cxcl12gfp', 'cxcl12-gfp'],
        6: ['ckit', 'ckit AF488'],
        7: ['hlf'],
        8: ['evi1', 'a-catulin', 'acatulin', 'ctnnal1'],
        9: ['tml'],
        10: ['rfp'],
        11: ['cd31'],
        12: ['sma'],  # smooth muscle actin
    }
    labels_ids = {
        1: ['gt tissue'],
        2: ['gt sinusoids', 'gt vessels'],
        3: ['gt tv'],
        4: ['gt arteries'],
        5: ['gt largevessels']
    }
    spotslabels_ids = {
        1: ['gt dapi'],
        2: ['gt ckit'],
        3: ['gt sec'],
        4: ['gt cxcl12', 'gt carc'],
        5: ['HSPC', 'hlf_bright', 'hlf_pos_ctnnal1_neg', 'hlf_pos_evi1_neg'],
        6: ['HSC', 'hlf_pos_ctnnal1_pos', 'hlf_pos_evi1_pos']
    }

    inv_channel_ids = common_utils.invert_listdict(channel_ids)
    inv_label_ids = common_utils.invert_listdict(labels_ids)
    inv_spotslabels_ids = common_utils.invert_listdict(spotslabels_ids)

    def __init__(self, filename, msettings, data_name=None):
        self.data_name = data_name
        self.msettings = msettings
        self.patch_size = self.msettings['patch_size']
        self.patch_size_out = self.msettings['patch_size_out']
        file_aux, fext = os.path.splitext(filename)
        self.original_ext = fext
        # The base folder corresponds to the created file, we ignore the input
        self.folder_base = os.path.split(filename)[0]
        # Construct from the defined filename
        if fext in ('.h5', '.hdf5'):
            logger.info("hdf5 file identified: using {}".format(filename))
            self.filename = filename
            if data_name is None:
                # with self.try_open('a') as f:
                logger.warning("data_name is None")
                with self.try_open('a') as f:
                    keys = list(f.keys())
                    if len(keys) == 1 and not (keys[0][:6] == 'Sample'):
                        self.data_name = keys[0]
            self.dnames = self.get_channel_names()

        # Construct from the ims file
        elif fext == '.ims':
            self.filename = file_aux + ".h5"
        else:
            raise Exception("File extension unknown. fext={}".format(fext))

    def try_open(self, mode='r'):
        count = 0
        while count < 1000:
            try:
                fh = h5py.File(self.filename, mode)
                break
            except OSError as errmsg:
                count += 1
                sleep(count ** 2)
                logger.warning('Could not access file {} with open mode {}.\nError code: {}.\nTrying again...'.format(
                    self.filename, mode, errmsg))
        if count > 0:
            logger.debug('file opened: {} attempts'.format(count))
        if (self.data_name is not None) and (self.data_name not in fh):
            # fh.close()
            # with h5py.File(self.filename, 'a') as fh:
            fh.create_group(self.data_name)
            # fh = h5py.File(self.filename, mode)
        return fh

    def get_channel_names(self):
        dnames = {}
        with self.try_open('r') as f:
            if self.data_name is not None:
                f = f[self.data_name]
            for nsample in f:
                dnames[nsample] = {}
                if 'Patch_0' in f[nsample]:
                    fsample = f[nsample + '/Patch_0']
                    for nchannel in fsample:
                        fchannel = fsample[nchannel]
                        cname = fchannel.attrs['name'].replace("'", "")
                        dnames[nsample][cname] = nchannel
        return dnames

    def predictions_to_ims(self, exp_name='pred', cdesc="No description"):
        with self.try_open('r') as gdata:
            if self.data_name is not None:
                gdata = gdata[self.data_name]
            dsamples = {}
            for sample_name in gdata:
                gsample = gdata[sample_name]
                if 'filename' in gsample.attrs:
                    dsamples[sample_name] = gsample.attrs['filename']
                elif 'original_filename' in gsample.attrs:
                    dsamples[sample_name] = gsample.attrs['original_filename']
                else:
                    logger.error(
                        "The field with the original filename could not be found in {}".format(sample_name))
        for ns, (sample_name, ims_name) in enumerate(dsamples.items()):
            print("{} of {} -- Converting {}".format(ns, len(dsamples.keys()), ims_name))
            self.hdf5_to_ims(ims_name, sample_name, exp_name=exp_name, cdesc=cdesc)

    def coordinates2dm(self):
        name_coordinates = [x.replace("dm", "coordinates") for x in self.msettings["labels"]]
        crad = self.msettings['cells_radius']
        if isinstance(crad, float):
            crad = [crad] * len(name_coordinates)
        elif isinstance(crad, list):
            if not (len(crad) == len(name_coordinates)):
                raise Exception("The length of the cell_radius array is different than the labels provided")
        else:
            raise Exception("cell_radius was provided in an unknown format")

        # Generate the kernels for each label
        crad = (np.array(crad) / self.msettings['pixel_size']).astype(np.int16)
        lkernels = [None] * len(crad)
        for nx, xrad in enumerate(crad):
            rad_kernel = int(xrad * 4)
            # We want to make sure the kernel is odd
            lkernels[nx] = im_processing.gndkernel((rad_kernel * 2) + 1, xrad,  len(self.patch_size_out))

        def convert_coords(Xcoor, nx):
            """
            Convert function

            Parameters
            ----------
            Xcoor : np.array
                Coordinates
            nx : int
                Class id

            Returns
            -------
            np.array
                Density map
            """
            kernel = lkernels[nx]
            rkernel = [int(x / 2) for x in kernel.shape]
            imdm = np.zeros(shape=self.patch_size_out, dtype=np.float64)


            if len(self.patch_size_out) == 2:
                Xcoor=np.delete(Xcoor,0,axis=1)

            for x in Xcoor.astype(np.uint16):
                lb = [None] * len(rkernel)
                ub = [None] * len(rkernel)
                lcrop = [None] * len(rkernel)
                ucrop = [None] * len(rkernel)
                for nx, xd in enumerate(x):
                    lb[nx] = max(xd - rkernel[nx], 0)
                    ub[nx] = min(xd + rkernel[nx] + 1, self.patch_size_out[nx])
                    lcrop[nx] = max(-(xd - rkernel[nx]), 0)
                    ucrop[nx] = max(xd + rkernel[nx] + 1 - self.patch_size_out[nx], 0)

                if len(lcrop)==3:
                    kernel_aux = kernel[lcrop[0]:-ucrop[0] or None, lcrop[1]:-ucrop[1] or None, lcrop[2]:-ucrop[2] or None]
                    assert np.equal(
                        imdm[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]].shape,
                        kernel_aux.shape
                    ).all(), "The cropped kernel has a different size than the region where it is to be fit"
                    # max pooling instead of sum for better cell separation
                    imdm[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]] = np.maximum(imdm[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]],
                                                                        kernel_aux)
                elif len(lcrop)==2:
                    kernel_aux = kernel[lcrop[0]:-ucrop[0] or None, lcrop[1]:-ucrop[1] or None]
                    assert np.equal(
                        imdm[lb[0]:ub[0], lb[1]:ub[1]].shape,
                        kernel_aux.shape
                    ).all(), "The cropped kernel has a different size than the region where it is to be fit"
                    # max pooling instead of sum for better cell separation
                    imdm[lb[0]:ub[0], lb[1]:ub[1]] = np.maximum(
                        imdm[lb[0]:ub[0], lb[1]:ub[1]],
                        kernel_aux)
                else:
                    raise UserWarning ("can only handle 2D or 3D dimensions")
                # for nd in len(rkernel):
            return imdm * 100

        with self.try_open('r+') as f:
            for sname in f:
                fs = f[sname]
                for pname in fs:
                    fp = fs[pname]
                    for nclass, ncoor in enumerate(name_coordinates):
                        X = fp[ncoor][:]
                        # if len(X) < 2:
                        #     continue
                        Xim = convert_coords(X, nclass)
                        data_name = ncoor.replace("coordinates", "dm")
                        # fdm = fp[ncoor.replace("coordinates", "dm")]
                        if data_name in fp:
                            del fp[data_name]
                        fdm = fp.create_dataset(data_name,
                                                data=Xim,
                                                compression="gzip",
                                                compression_opts=9,
                                                )
                        fdm.attrs['name'] = fp[ncoor].attrs['name']

    def hdf5_to_ims(self, ims_file, sample_name=None, exp_name='pred', cdesc="No description"):
        imF = ImarisFiles(ims_file)
        # Create new channels in ims
        l_ims_rchannels = [None] * len(self.msettings['labels'])
        l_ims_wchannels = [None] * len(self.msettings['labels'])
        for nc, cname in enumerate(self.msettings['labels']):
            l_ims_rchannels[nc] = imF.create_channel()
            l_ims_wchannels[nc] = exp_name + '_' + cname
            imF.write_info(l_ims_rchannels[nc], cname=l_ims_wchannels[nc], cdesc=cdesc)
        with self.try_open('r') as gdata:
            if self.data_name is not None:
                gdata = gdata[self.data_name]
            sample_name = sample_name or [x for x in gdata][0]
            logger.info("Reconstructing {} in {}".format(sample_name, self.filename))
            gsample = gdata[sample_name]
            pbar = tqdm(total=len(gsample))
            for patch_name in gsample:
                pbar.update(1)
                gpatch = gsample[patch_name]
                pinds = gpatch.attrs['inds_label']
                patchsize_orig = [pinds[1] - pinds[0], pinds[3] - pinds[2], pinds[5] - pinds[4]]
                if len(self.msettings['patch_size']) == 2:  # number of dimensions
                    patchsize_orig = patchsize_orig[1::]
                for nchannel, channel_name in enumerate(l_ims_wchannels):
                    if channel_name in gpatch:
                        try:
                            vol_channel = np.array(gpatch[channel_name])
                        except KeyError:
                            logger.warning("Loading didn't work, trying again")
                            vol_channel = gdata[sample_name][patch_name][channel_name][:]
                    else:
                        logger.warning("The channel {} is not in patch {}".format(channel_name, patch_name))
                    vol_channel_re = resize(vol_channel,
                                            patchsize_orig,
                                            order=0,
                                            preserve_range=True,
                                            mode='symmetric')
                    if vol_channel_re.max() <= 1:
                        vol_channel_re *= 255
                    imF.write_dataset_patches(np.expand_dims(vol_channel_re, 0),
                                              channel=l_ims_rchannels[nchannel],
                                              limInd=[pinds],
                                              cname=l_ims_wchannels[nchannel])
            pbar.close()
            for nc in l_ims_rchannels:
                imF.write_resLev(nc)

    @staticmethod
    def write_attrs(fh, dinfo):
        for k, v in dinfo.items():
            if v is not None:
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
                    v = [x.encode('ascii', 'ignore') for x in v]
                try:
                    fh.attrs[k] = v
                except Exception as ex:
                    fh.attrs[k] = str(ex)
                    logger.warning(str(ex))

    def ims_to_hdf5(self, do_labels, file_ims=None, max_samples=None, max_patches=None):
        for nsample, fname in enumerate(file_ims):
            if (max_samples is not None) and (nsample >= max_samples):
                break
            # if nsample<21:
            #     continue
            logger.info("Writing sample {}/{}: {}".format(nsample + 1, len(file_ims), fname))
            chnames = list(set([x.lower() for x in ImarisFiles(fname).channelNames]))
            scenenames = ImarisFiles(fname).sceneNames
            in_channels = [x for x in chnames if x in DTFM.inv_channel_ids]
            out_channels = [x for x in chnames if x in DTFM.inv_label_ids]
            not_used = [x for x in chnames if not ((x in DTFM.inv_channel_ids) or (x in DTFM.inv_label_ids))]
            logger.info("The following channels are not indexed: {}".format(not_used))
            if scenenames:
                spot_names = list(set([x.lower() for x in scenenames]))
                out_spots = [x for x in spot_names if x in DTFM.inv_spotslabels_ids]
                out_channels += out_spots
            # Maybe this should not be here
            _, lP = self.process_patchind(
                [fname],
                labels=do_labels,
                shuffle_inds=False,
                shuffle_dataset=False,
                nsamples=all,
                lChannels=in_channels,
                lChannelsGT=out_channels
            )
            self.ims2hdf5_im(lP[0], nsample, max_patches)
        # self.calc_weights_dataset()

    def process_patchind(self, filenames, labels=True, shuffle_inds=False, shuffle_dataset=False, nsamples=all,
                         lChannels=None,
                         lChannelsGT=None):
        if lChannels is None:
            lChannels = self.msettings['channels']
        if labels:
            if lChannelsGT is None:
                lChannelsGT = self.msettings['labels']
        else:
            lChannelsGT = None
        lP = [None] * len(filenames)
        totalinds = 0
        if self.msettings["padding"] == "same":
            overlap = 0
        else:
            overlap = cnn_utils.get_overlap(
                self.patch_size, self.msettings['scale_factor'], self.msettings['nlevels'],
                self.msettings['kernel_size'])
        for nf, filename in enumerate(filenames):
            lP[nf] = Patch(filename, out_psize=self.msettings['pixel_size'], overlap_vox=overlap,
                           patchsize_vox=self.patch_size, lChannels=lChannels, lChannelsGT=lChannelsGT)
            # if self.msettings['train']
            if labels:
                do_zInd = self.msettings['dataset_type'] in ['multiclass_seg', 'semantic_seg']
            else:
                do_zInd = False
            lP[nf].aInd = lP[nf].getPatchIndexes(do_zInd=do_zInd)
            if self.msettings["padding"] == "same":
                lP[nf].aInd_crop = lP[nf].aInd.copy()
            else:
                lP[nf].aInd_crop = lP[nf].padded_inds()
            totalinds += len(lP[nf].aInd)
        allInd = np.empty(shape=(totalinds, 2), dtype=np.uint32)
        p2 = 0
        for nf in range(len(filenames)):
            p1 = p2
            p2 += len(lP[nf].aInd)
            allInd[p1:p2, 0] = nf
            inds_aux = np.arange(len(lP[nf].aInd))
            if shuffle_inds:
                np.random.shuffle(inds_aux)
            allInd[p1:p2, 1] = inds_aux
        vInds = allInd
        if shuffle_dataset:
            np.random.shuffle(vInds)
        if nsamples is all:
            nsamples = len(vInds)
        # niters = int(np.ceil(nsamples / batch_size))
        logging.info("Dataset consists of {:d} files with {:d} patches. We use {:d} patches".format(
            len(filenames), len(vInds), nsamples))
        return vInds, lP

    def ims2hdf5_im(self, P, nsample=None, max_patches=None):
        # Hierarchy: 'LabeledData'/Sample_X/Channel_X/
        vInds = P.aInd
        vInds_crop = P.aInd_crop
        if nsample is None:
            nsample = None
        with self.try_open('a') as hdata:
            if self.data_name is not None:
                hdata = hdata[self.data_name]
            group_sample_name = 'Sample_' + str(nsample)
            if hdata.get(group_sample_name):
                del hdata[group_sample_name]
            hsample = hdata.create_group(group_sample_name)
            hsample.attrs['name'] = os.path.splitext(os.path.basename(P.filename))[0]
            hsample.attrs['original_filename'] = P.filename
            # DTFM.write_attrs(hsample, P.__dict__.copy())
            # lchannels = P.imfile.channelNames
            lchannels = P.lChannels + (P.lChannelsGT or [])
            lchannels = [x.lower() for x in lchannels]
            # Iterate indexes
            dweights_sample = {'label_channel_' + str(self.inv_label_ids[x]): 0 for x in lchannels if
                               x in self.inv_label_ids}
            dweights_sample['total'] = 0
            for nind, (vind_channel, vind_label) in enumerate(tqdm(zip(vInds, vInds_crop), total=len(vInds))):
                if (max_patches is not None) and (nind >= max_patches):
                    break
                # Fix indices
                indpad = np.array([[max([0, -vind_channel[0]]), max([0, vind_channel[1] - P.in_imsize[0]])],
                                   [max([0, -vind_channel[2]]), max([0, vind_channel[3] - P.in_imsize[1]])],
                                   [max([0, -vind_channel[4]]), max([0, vind_channel[5] - P.in_imsize[2]])],
                                   [0, 0]])
                if not (indpad == 0).all():
                    do_pad = True
                    vind_channel = [
                        max([vind_channel[0], 0]), min([vind_channel[1], P.in_imsize[0]]),
                        max([vind_channel[2], 0]), min([vind_channel[3], P.in_imsize[1]]),
                        max([vind_channel[4], 0]), min([vind_channel[5], P.in_imsize[2]])]
                    if self.msettings["padding"] == 'same':
                        # In principle there should be no padding because padding is same, but this means the patchsize
                        # is greater than the sample and it is thus required for both input and outpout
                        vind_label = vind_channel
                else:
                    do_pad = False
                hpatch = hsample.create_group('Patch_' + str(nind))
                hpatch.attrs['inds_channel'] = vind_channel
                hpatch.attrs['inds_label'] = vind_label
                hpatch.attrs['do_pad'] = do_pad
                # Iterate channels
                do_writetotal = True
                for nc, channel_name in enumerate(lchannels):
                    def write_channel_im(in_patch, channelgroup_name, interp_order=1, do_labelweight=False):
                        if in_patch.shape[0] == 1:
                            in_patch = in_patch[0]
                        do_antialiasing = False if interp_order == 0 else True
                        # out_type = np.uint16 if interp_order == 0 else np.float32
                        out_type = np.uint8 if interp_order == 0 else np.float16
                        out_patch = resize(in_patch, re_size, order=interp_order, anti_aliasing=do_antialiasing,
                                           preserve_range=True, mode='reflect').astype(out_type)
                        hchannel = hpatch.create_dataset(channelgroup_name,
                                                         data=out_patch,
                                                         compression="gzip",
                                                         compression_opts=9,
                                                         )
                        # For Fiji
                        psize = P.out_psize
                        psize = psize if psize.size > 1 else np.array([psize] * out_patch.ndim)
                        hchannel.attrs['element_size_um'] = psize
                        hchannel.attrs['name'] = channel_name
                        if do_labelweight:
                            sizeclass = (out_patch > 0).sum()
                            if do_writetotal:
                                sizeim = out_patch.size
                                dweights_sample['total'] += sizeim
                            dweights_sample[channelgroup_name] += sizeclass
                            # labelweight = sizeclass / sizeim
                            # hchannel.attrs['weight'] = labelweight

                    # If input
                    if channel_name in DTFM.inv_channel_ids:
                        interp_order = 1
                        re_size = self.patch_size
                        channelgroup_name = 'in_channel_' + str(DTFM.inv_channel_ids[channel_name])
                        patch = P.imfile.getVolume(channels=[channel_name], limInd=vind_channel)[..., 0]
                        if nind == 0:
                            cmean = P.imfile.get_stat(channel_name, 'mean')
                            cstd = P.imfile.get_stat(channel_name, 'std')
                            hsample.attrs["mean_" + channelgroup_name] = cmean
                            hsample.attrs["std_" + channelgroup_name] = cstd
                        else:
                            cmean = hsample.attrs["mean_" + channelgroup_name]
                            cstd = hsample.attrs["std_" + channelgroup_name]
                        patch = (patch - cmean) / cstd
                        # Pad
                        if do_pad:
                            patch = np.pad(patch, indpad[0:-1, :], 'constant')
                        write_channel_im(patch, channelgroup_name, interp_order)

                    # If output
                    elif channel_name in DTFM.inv_label_ids:
                        interp_order = 0
                        re_size = self.patch_size_out
                        channelgroup_name = 'label_channel_' + str(DTFM.inv_label_ids[channel_name])
                        patch = P.imfile.getVolume(channels=[channel_name], limInd=vind_label)[..., 0]
                        patch = patch > 0
                        write_channel_im(patch, channelgroup_name, interp_order, do_labelweight=True)
                        do_writetotal = False
                    # If spot
                    elif channel_name in DTFM.inv_spotslabels_ids:
                        channelgroup_name = 'label_coordinates_' + str(DTFM.inv_spotslabels_ids[channel_name])

                        pname = "Points" + str(P.imfile.sceneName2Number([channel_name])[0])
                        with h5py.File(P.filename, 'r') as fim:
                            Xumreal = fim["/Scene/Content/" + pname + "/CoordsXYZR"][:, 0:3]
                            ImageAttrs = fim['/DataSetInfo/Image'].attrs
                            vExtMaxX = float("".join(ImageAttrs['ExtMax0'].astype(str)))
                            vExtMaxY = float("".join(ImageAttrs['ExtMax1'].astype(str)))
                            vExtMaxZ = float("".join(ImageAttrs['ExtMax2'].astype(str)))
                            vExtMinX = float("".join(ImageAttrs['ExtMin0'].astype(str)))
                            vExtMinY = float("".join(ImageAttrs['ExtMin1'].astype(str)))
                            vExtMinZ = float("".join(ImageAttrs['ExtMin2'].astype(str)))

                        # We calculate the pixel size from scratch because of some previous implementation problems
                        vExtMin = np.array([vExtMinX, vExtMinY, vExtMinZ])
                        vExtMax = np.array([vExtMaxX, vExtMaxY, vExtMaxZ])
                        voxelSize = (vExtMax - vExtMin) / P.imfile.imsize
                        Xvox = (Xumreal - P.imfile.imExtends[0]) / voxelSize  # + 0.5

                        cspot = P.imfile.getSpotsObj(channel_name)
                        # Xvox = cspot.X_voxel0
                        keep_pos = np.all((Xvox >= vind_channel[0::2][::-1],
                                           Xvox <= vind_channel[1::2][::-1]),
                                          axis=0).all(axis=1)
                        Xlim = Xvox[keep_pos]
                        Xlim += indpad[:-1, 0][::-1] - vind_channel[0::2][::-1]
                        X = Xlim * P.patchsize_vox[::-1] / P.patchsize_in[::-1]
                        X = np.fliplr(X)
                        patch_maxcoor = self.patch_size if (len(self.patch_size) == 3) else [1] + self.patch_size
                        if (X.shape[0] > 0) and (
                                (X.max(axis=0).astype(np.uint16) > patch_maxcoor).any() or
                                (X.min(axis=0).astype(np.uint16) < [0, 0, 0]).any()):
                            logger.error("Dimensions of the coordinates are not good")
                        hchannel = hpatch.create_dataset(channelgroup_name,
                                                         data=X,
                                                         compression="gzip",
                                                         compression_opts=4)
                        hchannel.attrs['name'] = channel_name
                        hchannel.attrs['radius_um'] = cspot.radius_um
                        hchannel.attrs['radius_voxels'] = cspot.radius_voxel[::-1]
                        hchannel.attrs['inds_channel'] = vind_channel
                        hchannel.attrs['inds_label'] = vind_label
                        hchannel.attrs['do_pad'] = False
                    # If channel name not in list
                    else:
                        raise Exception("This channel should not have been included, as it is not indexed. Check why it"
                                        " was included")
            # vbg = dweights_sample['bg']
            for k, v in dweights_sample.items():
                hsample.attrs[k + '_cnumber'] = v
                # if k is not 'bg':
                #     hsample.attrs[k + '_weight'] = v / vbg


    def write_preds(self, x, metadata, exp_name=None):
        voxel_size = self.msettings['pixel_size']
        if not isinstance(voxel_size, list):
            voxel_size = [voxel_size] * len(self.patch_size)
        with self.try_open('r+') as f:
            if self.data_name is not None:
                f = f[self.data_name]
            for cc in range(x.shape[-1]):
                in_cname = metadata['l_channels_out'][cc]
                cid = [int(s) for s in in_cname.split('_') if s.isdigit()][-1]
                try:
                    raw_cname = DTFM.labels_ids[cid][0]
                except:
                    raw_cname = ""
                out_cname_aux = exp_name or 'pred'
                out_cname = out_cname_aux + '_' + in_cname
                pdata = f[metadata['sample'] + '/' + metadata['patch']]
                # if out_cname in pdata:
                #     # Normally this occurs because we repeat the initial patches to complete the last batch of the sample
                #     logger.warning("The channel {} already exists in {}, {}. Deleting...".format(
                #         out_cname, metadata['sample'], metadata['patch']))
                #     # continue
                #     for pname in f[metadata['sample']]:
                #         pgroup = f[metadata['sample'] + '/' + pname]
                #         if out_cname in pgroup:
                #             pgroup.__delitem__(out_cname)
                # else:
                #     logger.debug("Writing {} in {}, {}".format(out_cname, metadata['sample'], metadata['patch']))
                if out_cname in pdata:
                    pdata[out_cname][:] = x[..., cc].astype('uint8')
                    cdata = pdata[out_cname]
                else:
                    cdata = pdata.create_dataset(out_cname,
                                                 data=x[..., cc].astype('uint8'),
                                                 dtype='uint8',
                                                 compression="gzip",
                                                 compression_opts=4)
                cdata.attrs['name'] = raw_cname
                cdata.attrs['element_size_um'] = voxel_size

    def clean_channels(self, lchannels=None, do_ask=False):
        if lchannels is None:
            lchannels = ['pred_' + x for x in self.msettings['labels']]
        with self.try_open('r+') as dgroup:
            if self.data_name is not None:
                dgroup = dgroup[self.data_name]
            for sname in dgroup:
                sgroup = dgroup[sname]
                groups_checked = False
                for pname in sgroup:
                    pgroup = sgroup[pname]
                    for channel_name in lchannels:
                        if channel_name in pgroup:
                            if do_ask:
                                q = input("Channel {} already exists in sample {}. Delete it? (y/n): "
                                          .format(channel_name, sname))
                            else:
                                q = 'y'
                            if q == 'y':
                                pbar = tqdm(total=len(sgroup))
                                for pname_aux in sgroup:
                                    pbar.update(1)
                                    pgroup_aux = sgroup[pname_aux]
                                    if channel_name in pgroup_aux:
                                        pgroup_aux.__delitem__(channel_name)
                                logger.info("Channel {} has been deleted in sample {}"
                                            .format(channel_name, sname))
                                groups_checked = True
                            else:
                                logger.info("Channel {} has NOT been deleted in sample {}"
                                            .format(channel_name, sname))
                    if groups_checked:
                        break

    def get_samples(self):
        with self.try_open('r') as f:
            if self.data_name is not None:
                f = f[self.data_name]
            return len(f)


class DTFMdataset(DTFM):
    def __init__(self, filename, msettings, data_name='LabeledData', l_channels_in=None, l_channels_out=None,
                 batch_size=None, miss_channels=None):
        super().__init__(filename, msettings, data_name)
        self.batch_size = batch_size if (batch_size is not None) else self.msettings['batch_size']
        self.nsamples = self.get_samples()
        if l_channels_in is None:
            self.l_channels_in = ["in_channel_" + str(x) for x in self.channel_ids.keys()]
        else:
            self.l_channels_in = l_channels_in
        if l_channels_out is None:
            self.l_channels_out = ["label_channel_" + str(x) for x in self.channel_ids.keys()]
        else:
            self.l_channels_out = l_channels_out
        self.miss_channels = [] if miss_channels is None else miss_channels
        self.extra_class = 0 if (
                ('dataset_type' in self.msettings) and self.msettings['dataset_type'] in ('detection')
        ) else 1
        if self.extra_class > 1:
            logger.error("Methods not implemented for more than 1 extra classes")
        self.max_spots = 1000

    def set_classweights(self, linds):
        channel_count = np.zeros((len(linds), len(self.l_channels_out + ['total'])), np.int64)
        with self.try_open('r') as f:
            if self.data_name is not None:
                f = f[self.data_name]
            for cnumber, cname in enumerate(self.l_channels_out + ['total']):
                for ns, sind in enumerate(linds):
                    sname = "Sample_" + str(sind)
                    attname = cname + '_cnumber'
                    val = 0. if attname not in f[sname].attrs else f[sname].attrs[attname]
                    channel_count[ns][cnumber] += val
        # The last position is for image size, so we substract all other classes
        channel_count[:, -1] -= np.sum(channel_count[:, :-1], axis=1)
        cweights = common_utils.get_weights(np.sum(channel_count, axis=0))
        return cweights

    def get_list_validsamples(self):
        if self.msettings['dataset_condition'] is None or self.msettings['dataset_condition'] == 'all':
            l_samples = list(np.arange(self.nsamples))
        else:
            l_samples = []
            with self.try_open('r') as f:
                if self.data_name is not None:
                    f = f[self.data_name]
                for nsample, name_sample in enumerate(f):
                    if len(f[name_sample])==0:
                        continue
                    list_channels = f[name_sample + '/Patch_0'].keys()
                    all_cond = self.msettings['dataset_condition']['channels'] + self.msettings['dataset_condition'][
                        'labels']
                    if set(all_cond).issubset(list_channels):
                        l_samples.append(int(name_sample.replace("Sample_", "")))
        return l_samples

    def crossval_datasets(self, kfold=5, rval=0.16, do_random=True, l_samples=None):
        # kfold = np.around(1/rtest).astype(int)
        if l_samples is None:
            l_samples = self.get_list_validsamples()
        if kfold > len(l_samples):
            logging.warning(
                "The number of crossvalidation steps cannot be bigger than the number of samples: \n"
                "Number of cv steps: {}\n"
                "Number of samples: {}\n"
                "Changing the cv steps to match the number of samples".format(kfold, len(l_samples)))
            kfold = len(l_samples)
        if do_random:
            np.random.seed(seed=12)
            np.random.shuffle(l_samples)
        lsets = []
        linds = self.get_indices(kfold, l_samples)
        for ninds, (inds_train, inds_val, inds_test) in enumerate(linds):
            dataset, newinds = self.get_split_datasets(inds_train, inds_val, inds_test)
            linds[ninds][0] = newinds[0]
            lsets.append(dataset)
        return lsets, linds

    def get_indices(self, kfold, l_samples, rval=0.16):
        if "indices" in self.msettings and self.msettings["indices"] is not None:
            linds_aux = self.msettings["indices"]
            logger.info("Using indices predefined in the model: \n {} \n\n".format(linds_aux))

            lcond = self.get_list_validsamples()
            linds = [None] * len(linds_aux)
            for ncv in range(len(linds_aux)):
                # We sort by index it so that the order initially stated is kept (useful when deleting ch)
                linds[ncv] = [sorted(
                    set(linds_aux[ncv][ns]).intersection(set(lcond)), key=lambda x: linds_aux[ncv][ns].index(x)
                ) for ns in range(len(linds_aux[ncv]))]
            logger.info("Indices after conditions: \n {} \n\n".format(linds))

        else:
            # Define the starting test index for each of the validation steps
            test_inds = np.around(np.linspace(0, len(l_samples), kfold + 1)[:-1]).astype(np.int64)
            linds = []
            for nmodel in range(kfold):
                t1 = test_inds[nmodel]
                t2 = len(l_samples) if nmodel == (kfold - 1) else test_inds[nmodel + 1]
                inds_test = l_samples[t1:t2]
                l_samples_aux = common_utils.shift_list(l_samples[:t1] + l_samples[t2:], (t2 - t1) * nmodel)
                indcut = np.around(np.array([1 - rval, rval]) * len(l_samples_aux)).astype(np.uint16)
                # Correction in case there are not enough samples for validation set
                if indcut[1] == 0:
                    indcut[0] -= 1
                    indcut[1] += 1
                inds_train = l_samples_aux[0:indcut[0]]
                inds_val = l_samples_aux[indcut[0]:indcut[0] + indcut[1]]
                logger.debug("Training samples: {}".format("".join(str(inds_train))))
                logger.debug("Validation samples: {}".format("".join(str(inds_val))))
                logger.debug("Testing samples: {}".format("".join(str(inds_test))))
                assert (len(set(inds_train).intersection(set(inds_val))) == 0)
                assert (len(set(inds_train + inds_val).intersection(set(inds_test))) == 0)
                linds.append([inds_train, inds_val, inds_test])
            cum_test = [x for xaux in linds for x in xaux[2]]
            assert (set(cum_test) == set(l_samples)), "Not all the test samples are covered"

        # Tests
        len_sets = []
        for sample_cv in linds:
            lx = 0
            for sset in sample_cv:
                lx += len(sset)
            len_sets.append(lx)
        # assert (len(set(len_sets)) == 1), "Some cross validation sets contain more samples than others"
        if not (len_sets[0] == len(l_samples)):
            logger.debug("The indices do not cover all the accessible samples")
        return linds

    def split_datasets(self, data_split=(0.6, 0.2, 0.2), do_random=True, l_samples=None, verbose=True):
        rtrain, rval, rtest = data_split
        assert ((rtrain + rval + rtest) == 1)
        if l_samples is None:
            l_samples = self.get_list_validsamples()
        if do_random:
            np.random.seed(seed=12)
            np.random.shuffle(l_samples)
        indcut = np.around(np.array([rtrain, rval, rtest]) * len(l_samples)).astype(np.uint16)
        inds_train = l_samples[0:indcut[0]]
        inds_val = l_samples[indcut[0]:indcut[0] + indcut[1]]
        inds_test = l_samples[indcut[0] + indcut[1]:indcut[0] + indcut[1] + indcut[2]]
        if verbose:
            logger.info("Training samples: {}".format("".join(str(inds_train))))
            logger.info("Validation samples: {}".format("".join(str(inds_val))))
            logger.info("Testing samples: {}".format("".join(str(inds_test))))
        datasets, _ = self.get_split_datasets(inds_train, inds_val, inds_test), (inds_train, inds_val, inds_test)
        return datasets

    def get_split_datasets(self, inds_train, inds_val, inds_test):
        logger.debug("Train indices: {}".format(inds_train))
        logger.debug("Validation indices: {}".format(inds_val))
        logger.debug("Test indices: {}".format(inds_test))

        # Correct list of training samples with specific channel combinations
        chcomb = None if 'chcomb' not in self.msettings else self.msettings['chcomb']
        if chcomb and not self.msettings['dataset_condition'] == 'all':
            lkeep = []
            for nind in range(len(inds_train)):
                nsample = inds_train[nind]
                name_sample = 'Sample_' + str(nsample)
                all_cond = ["in_channel_" + str(x) for x in chcomb[nind]]
                if set(self.msettings['channels']).issubset(all_cond):
                    lkeep += [nind]
            inds_train = [x for n, x in enumerate(inds_train) if n in lkeep]
        ds_train = self.tf_dataset(
            data_read=inds_train,
            do_shuffle=True,
            chcomb=chcomb,
            data_set='train'
        ) if len(inds_train) > 0 else None
        ds_val = self.tf_dataset(
            data_read=inds_val,
            do_shuffle=False,
            data_set='val'
        ) if len(inds_val) > 0 else None
        ds_test = self.tf_dataset(
            data_read=inds_test,
            do_shuffle=False,
            data_set='test'
        )
        return [ds_train, ds_val, ds_test], [inds_train, inds_val, inds_test]

    def tf_dataset(self, data_read=None, do_shuffle=False, augm_dict=None, chcomb=None, data_set=None):
        # Check number of regions
        if augm_dict is None:
            augm_dict = self.msettings['augmentation']
        if isinstance(data_read, dict):
            n = 0
            for k, v in data_read.items():
                n += len(v)
            if n < self.batch_size:
                data_read[k] = v * self.batch_size

        if self.extra_class == 0 and (data_set in ('val', 'test')):
            out_shape = [self.max_spots, 3, len(self.l_channels_out)]
        else:
            out_shape = self.patch_size_out + [len(self.l_channels_out) + self.extra_class]
        dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(data_read, do_shuffle=do_shuffle, chcomb=chcomb, data_set=data_set),
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                self.patch_size + [len(self.l_channels_in)],
                out_shape
            ))

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def data_generator(self,
                       data_read=None,
                       chcomb=None,
                       sample_markers=None,
                       data_set=None,
                       do_shuffle=False,
                       isout=None):

        if isout is None:
            isout = {}
            if data_set and (data_set in ('val', 'test') and (self.extra_class == 0)):
                isout['coordinates'] = True
            else:
                isout['image'] = True
        for k in ('metadata', 'image', 'coordinates'):
            if k not in isout:
                isout[k] = False

        # Account for transitional vessels
        str_mix = ['gt tv', 'gt largevessels']
        may_mix_names = ['label_channel_' + str(self.inv_label_ids[x]) for x in str_mix]
        mix_names = [x for x in may_mix_names if x in self.msettings['labels']]
        vess_name = 'label_channel_' + str(self.inv_label_ids['gt sinusoids'])
        dospot_doeval = (self.extra_class == 0) and (data_set in ('val', 'test'))
        # Create background if task is segmentation
        if len(mix_names) == 1:
            mix_index = self.msettings['labels'].index(mix_names[0])
            if vess_name in self.msettings['labels']:
                vess_index = self.msettings['labels'].index(vess_name)
                do_mix = True
            else:
                do_mix = False
        elif len(mix_names) == 0:
            do_mix = False
        else:
            raise Exception("The number of mixed structures is greater than 1, this option is not implemented")

        do_balance_cells = ('balance_cells' in self.msettings) and (self.msettings['balance_cells'] is not None)

        # Data preparation
        last_sample = None
        if data_read is None:
            with self.try_open('r') as f:
                if self.data_name is not None:
                    f = f[self.data_name]
                data_read = np.arange(len(f))
        ldata = []
        if chcomb and self.msettings['dataset_condition'] == 'all':
            assert len(chcomb) == len(data_read)

        if isinstance(data_read, dict):
            for k, v in data_read.items():
                for vi in v:
                    ldata.append([k, vi])
                    # ldata.append(['Sample_' + str(k), 'Patch_' + str(vi)])
        else:
            for csample in data_read:
                nsample = 'Sample_' + str(csample)
                with self.try_open('r') as f:
                    if self.data_name is not None:
                        f = f[self.data_name]
                    gsample = f[nsample]
                    lpatches = []
                    for ns in gsample.keys():
                        if ns in gsample:
                            lpatches += [int(ns.replace("Patch_", ""))]
                        else:
                            logger.warning("Problem with sample {}, patch {}. Skipping...".format(nsample, ns))
                    # lpatches = [int(x[0].replace("Patch_", "")) for x in gsample.items()]

                    # ldata += [[nsample, 'Patch_' + str(x)] for x in range(npatches)]
                    if do_balance_cells:
                        # Account for the number of cells
                        if len(self.l_channels_out) > 1:
                            logger.error(
                                "Method implemented only for one class, found {}".format(len(self.l_channels_out)))
                        else:
                            name_class = self.l_channels_out[0].replace("dm", "coordinates")
                            ldata += [[csample, x, gsample['Patch_' + str(x)][name_class][:].shape[0]] for x in
                                      lpatches]
                    else:
                        ldata += [[csample, x] for x in lpatches]
                    # Check if channels exist - This should not be needed, but just for debugging
                    dpatch_aux = gsample['Patch_0']
                    for nchannel in (self.l_channels_in + self.l_channels_out):
                        if nchannel not in dpatch_aux:
                            logger.debug("The channel " + nchannel + " does not exist in sample: " + nsample)
                            break
        ldata_aux = []
        adata = np.array(ldata)
        if ('sample_patches' in self.msettings) and self.msettings['sample_patches'] and (data_set in ('train', 'val')):
            logger.info("Taking a maximum of {} patches per sample".format(self.msettings['sample_patches']))
            sinds = np.unique(adata[:, 0])
            for sind in sinds:
                daux = adata[adata[:, 0] == sind]
                ldata_aux += daux[daux[:, 1].argsort(axis=0)[:self.msettings['sample_patches']]].tolist()
            ldata = ldata_aux

        # Batches of samples
        blocksize = self.msettings['batch_size']
        curr_sample = -1
        lblock = []
        count = -1
        block = []
        for i in range(len(ldata)):
            count += 1
            change_sample = not (ldata[i][0] == curr_sample)
            if change_sample and count > 0:
                while not (count == blocksize):
                    block += [ldata[b_curr_sample + count]]
                    count += 1
            if count == blocksize or count == 0:
                if change_sample:
                    b_curr_sample = i
                if block:
                    lblock.append(block)
                count = 0
                curr_sample = ldata[i][0]
                block = []
            block += [ldata[i]]
        if ((count+1) == blocksize) and block:
            lblock.append(block)


        # Batch shuffle
        if do_shuffle:
            np.random.shuffle(lblock)
        # As list
        ldata_batches = [b for bs in lblock for b in bs]

        if do_balance_cells:
            ldata_aux = np.array(ldata_batches)
            ldata_sorted = np.flipud(ldata_aux[ldata_aux[:, 2].argsort()])
            ldata_sorted = np.delete(ldata_sorted, np.argwhere(ldata_sorted[:, 2] < 2), 0)
            ncells = np.sum(ldata_aux[:, 2])
            density = ncells / len(ldata_aux)
            count = -1
            while density < 1.5:
                count += 1
                if count == len(ldata_sorted):
                    count = 0
                ldata_aux = np.vstack([ldata_aux, ldata_sorted[count, :]])
                density = np.sum(ldata_aux[:, 2]) / len(ldata_aux)
            ldata_batches = ldata_aux

        # Read data
        for idata in ldata_batches:
            if idata is None:
                yield None, None
            else:
                nsample, npatch = 'Sample_' + str(idata[0]), 'Patch_' + str(idata[1])
                with self.try_open('r') as f:
                    if self.data_name is not None:
                        f = f[self.data_name]

                    def get_coordinates(l_channels):
                        # We assume a maximum of 1000 spots
                        self.max_spots = 1000
                        X = -1 * np.ones(shape=(self.max_spots, 3, len(l_channels)))
                        for c_counter, name_channel in enumerate(l_channels):
                            pgroup = f[nsample + '/' + npatch]
                            xaux = pgroup[name_channel.replace("dm", "coordinates")][:]
                            if len(xaux) > self.max_spots:
                                logger.error(
                                    "We assumed the maximum number of spots per patch was {}, but got {}".format(
                                        self.max_spots, len(xaux)))
                            X[:len(xaux), :, c_counter] = xaux
                        return X

                    def get_image(l_channels, create_bg=False):
                        do_create = True
                        im_channels = len(l_channels) + self.extra_class if create_bg else len(l_channels)
                        channels_exist = False
                        for c_counter, name_channel in enumerate(l_channels):
                            pgroup = f[nsample + '/' + npatch]

                            samplename = f[nsample].attrs['name']
                            if sample_markers and samplename in sample_markers:
                                missmarker_sample = int(name_channel.replace("in_channel_", "")) not in sample_markers[
                                    samplename]
                            else:
                                missmarker_sample = False
                            if name_channel in pgroup and name_channel not in self.miss_channels and not missmarker_sample:
                                channels_exist = True
                                attempts = 0
                                max_attempts = 50
                                while attempts < max_attempts:
                                    try:
                                        im_aux = pgroup[name_channel][:]
                                        break
                                    except:
                                        attempts += 1
                                        logger.warning("Data in {} could not be opened, trying again...".format(
                                            [nsample + '/' + npatch + '/' + name_channel]))
                                        sleep(5)
                                        if attempts == max_attempts:
                                            logger.warning(
                                                "Dataset could not be opened in {} attempts. Closing program".format(
                                                    attempts))
                                # Consider structures which should be mixed
                                if (name_channel == vess_name) and (not do_mix):
                                    mix_names = [x for x in may_mix_names if x in pgroup]
                                    if len(mix_names) > 0:
                                        for mix_name in mix_names:
                                            mix_im = pgroup[mix_name][:]
                                            im_aux += (im_aux < 1) * mix_im
                                if do_create:
                                    im = np.zeros(shape=(list(im_aux.shape) + [im_channels]), dtype=np.float32)
                                    do_create = False
                                im[..., c_counter] = im_aux

                        if not channels_exist:
                            logger.warning("No channels available for {}, {} \n List of channels: {}".format(
                                nsample, npatch, l_channels
                            ))
                        if create_bg and self.extra_class > 0:
                            im[..., -1] = (im[..., :-1] == 0).all(axis=-1)
                        return im

                    data_in = get_image(self.l_channels_in, create_bg=False)
                    # Apply chdel
                    if chcomb:
                        keep_inds = chcomb[data_read.index(idata[0])]
                        for cc in range(data_in.shape[-1]):
                            if (cc + 1) not in keep_inds:
                                data_in[..., cc] = np.zeros_like(data_in[..., cc])

                    if isout['coordinates']:
                        data_coordinates = get_coordinates(self.l_channels_out)
                    if isout['image']:
                        data_outimage = get_image(self.l_channels_out, create_bg=True)
                        if do_mix:  # substract tv from sinusoids
                            data_outimage[..., vess_index] *= (data_outimage[..., mix_index] < 1)
                    else:
                        data_outimage = None
                    data_in, data_outimage = cnn_utils.image_augmentation(self.msettings['augmentation'], data_in,
                                                                          data_outimage)
                    fsample = f[nsample + '/' + npatch]
                    inds_channel = fsample.attrs['inds_channel']
                    inds_label = inds_channel if 'inds_label' not in fsample.attrs else fsample.attrs['inds_label']
                data_metadata = {'l_channels_out': self.l_channels_out,
                                 'patch': npatch,
                                 'sample': nsample,
                                 'inds_channel': inds_channel,
                                 'inds_label': inds_label}

                if isout['image'] and not (list(data_outimage.shape[:-1]) == self.patch_size_out):

                    def dim_bounds(ddiff):
                        lb = ddiff // 2
                        if ddiff % 2:
                            ub = lb + 1
                        else:
                            ub = lb
                        return lb, ub

                    sdiff = np.array((data_outimage.shape[:-1])) - self.patch_size_out
                    cind = [dim_bounds(xdiff) for xdiff in sdiff]
                    sx = tuple(
                        [slice(cc[0], data_outimage.shape[nx] - cc[1]) for (nx, cc) in enumerate(cind)] + [slice(None)])
                    data_outimage = data_outimage[sx]

                # We assume the coordinates are stored in the frame of patch_in
                if isout['coordinates']:
                    # calculate patch padding
                    sdiff = np.array((data_in.shape[:-1])) - self.patch_size_out
                    if data_coordinates.shape[-1] > 1:
                        logger.error(
                            "This method is only designed for one class, but got {}".format(data_coordinates.shape[-1]))

                    condind1 = (data_coordinates < 0).any(axis=1)[..., 0]
                    condind2 = (data_coordinates >=
                                np.expand_dims(np.repeat([self.patch_size_out], data_coordinates.shape[0], axis=0), -1)
                                ).any(axis=1)[..., 0]

                    data_coordinates[condind1 | condind2] = -1 * np.ones_like(data_coordinates[condind1 | condind2])


                def parse_outdata(data_key):
                    if data_key == 'coordinates':
                        return data_coordinates
                    elif data_key == 'metadata':
                        return data_metadata
                    elif data_key == 'image':
                        return data_outimage
                    else:
                        raise Exception("data key {} unknown".format(data_key))

                data_out = {}
                for k in isout:
                    if isout[k]:
                        data_out[k] = parse_outdata(k)
                if len(data_out.keys()) == 1:
                    data_out = [x for x in data_out.values()][0]

                yield (data_in, data_out)
