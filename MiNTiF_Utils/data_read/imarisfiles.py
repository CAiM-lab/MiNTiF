# (c) 2018-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
# -*- coding: utf-8 -*-


import h5py
import numpy as np
from  data_read.spots import Spots
from skimage import transform
from time import sleep
import logging
from  utils.im_processing import convert_uint8

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ImarisFiles:
    """Input and output functions with ims files as hdf5:

    Attributes:
        h5file: the file to work with
        channelNames: name of the channels in the file
        imsize: image size
        voxelSize: voxel size
        imExtends: image extends

    """""

    def __init__(self, filename):
        self.filename = filename
        self.dataset_stats = {}
        self.display_errorspots = []
        self.reset()

    def reset(self):
        # try:
        #     fh.flush()
        #     fh.close()
        # except:
        #     pass
        # fh = h5py.File(self.filename, 'r+')
        self.channelNames = self.getChannelNames()
        self.sceneNames = self.getSceneNames()
        self.imsize = self.getDataSize()
        self.imExtends = self.getExtends()
        self.voxelSize = self.getVoxelSize()
        self.dtype = self.getdtype()
        self.res_levels = self.getResLevels()

    def getResLevels(self):
        with self.try_open('r+') as fh:
            res_levels = len(fh['DataSet'])
        return res_levels

    def get_stat(self, cname, stat):
        if cname not in self.dataset_stats:
            with self.try_open('r') as fh:
                nres = 1 if self.res_levels > 1 else 0
                f_resolution = '/ResolutionLevel ' + str(nres)
                f_channel = '/Channel ' + str(self.channelName2Number([cname])[0])
                fr = fh['/DataSet' + f_resolution + '/TimePoint 0' + f_channel]
                ImageAttrs = fr.attrs
                lx = int(float("".join(ImageAttrs['ImageSizeX'].astype(str))))
                ly = int(float("".join(ImageAttrs['ImageSizeY'].astype(str))))
                lz = int(float("".join(ImageAttrs['ImageSizeZ'].astype(str))))
                dataset = fh['/DataSet' + f_resolution + '/TimePoint 0' + f_channel + '/Data']
                vol = dataset[:lz, :ly, :lx]
            # vol = self.getVolume([cname])[..., 0]
            self.dataset_stats[cname] = {
                "mean": vol.mean(),
                "std": vol.std()
            }
        return self.dataset_stats[cname][stat]

    def try_open(self, mode):
        count = 0
        while True:
            try:
                fh = h5py.File(self.filename, mode)
                break
            except OSError:
                count += 1
                if count > 11:
                    raise Exception(
                        'File {} could not be opened after {} attempts. Stopping.'.format(self.filename, count))
                sleep(2 ** count)
                logger.warning('error accessing {}. {} attempts. Trying again...'.format(self.filename, count))
        if count > 0:
            logger.debug('file opened: {} attempts'.format(count))
        return fh

    def getdtype(self):
        with self.try_open('r') as fh:
            htype = fh['/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data'].dtype
        return htype

    def getSpotsRadii(self, sID):
        if type(sID) is int:
            sName = "Points" + str(sID)
        elif type(sID) is str:
            sName = "Points" + str(self.sceneName2Number((sID,))[0])
        else:
            raise Exception("Unknown type for spot ID")
        with self.try_open('r') as fh:
            vrad = fh["/Scene/Content/" + sName + "/CoordsXYZR"][:, 3]
        return vrad.mean()

    def getSpotsCoordinates(self, sID):
        if type(sID) is int:
            sName = "Points" + str(sID)
        elif type(sID) is str:
            sName = "Points" + str(self.sceneName2Number((sID,))[0])
        else:
            raise Exception("Unknown type for spot ID")
        with self.try_open('r') as fh:
            X = fh["/Scene/Content/" + sName + "/CoordsXYZR"][:, 0:3]
        return X

    def writeChannelName(self, channel, name):
        name_aux = [x for x in name]
        fName = np.array(name_aux, dtype='|S1')
        f_channel = '/Channel ' + str(channel)
        with self.try_open('r+') as fh:
            fh['/DataSetInfo' + f_channel].attrs['Name'] = fName

    def sceneName2Number(self, lScenes):
        nScene = np.zeros((len(lScenes))).astype(np.int8) - 1
        scene_names = self.sceneNames
        for i in range(0, len(lScenes)):
            nScene_aux = [s for s, e in enumerate(scene_names) if lScenes[i].lower() == e.lower()]
            if nScene_aux:
                nScene[i] = int(nScene_aux[-1])
        return nScene

    def channelName2Number(self, lChannels):
        nChannel = np.zeros((len(lChannels))).astype(np.int8) - 1
        channel_names = self.channelNames
        for i in range(0, len(lChannels)):
            nChannel_aux = [s for s, e in enumerate(channel_names) if lChannels[i].lower() == e.lower()]
            if nChannel_aux:
                nChannel[i] = int(nChannel_aux[-1])
        return nChannel

    def setChannelNames(self, tmpfolder, signal_message=('cxcl12', 'dapi', 'endomucin', 'endoglin', 'collagen')):
        channel_names = self.channelNames
        vol = self.getVolume(channels=all)
        q1 = True
        for i in range(0, len(signal_message)):
            nChannel_aux = [s for s, e in enumerate(channel_names) if signal_message[i].lower() == e.lower()]
            if not nChannel_aux:
                if q1:
                    q1 = False
                    self.show_channels(vol, tmpfolder)
                nChannel = input('Which channel is ' + signal_message[i] + '?: ')
                if nChannel >= 0:
                    self.writeChannelName(nChannel, signal_message[i])

    def getExtends(self):
        with self.try_open('r') as fh:
            ImageAttrs = fh['/DataSetInfo/Image'].attrs
            vExtMaxX = float("".join(ImageAttrs['ExtMax0'].astype(str)))
            vExtMaxY = float("".join(ImageAttrs['ExtMax1'].astype(str)))
            vExtMaxZ = float("".join(ImageAttrs['ExtMax2'].astype(str)))
            vExtMinX = float("".join(ImageAttrs['ExtMin0'].astype(str)))
            vExtMinY = float("".join(ImageAttrs['ExtMin1'].astype(str)))
            vExtMinZ = float("".join(ImageAttrs['ExtMin2'].astype(str)))
            vExtMax = np.array([vExtMaxX, vExtMaxY, vExtMaxZ])
            vExtMin = np.array([vExtMinX, vExtMinY, vExtMinZ])
        return vExtMin, vExtMax

    def getDataSize(self):
        with self.try_open('r') as fh:
            ImageAttrs = fh['/DataSetInfo/Image'].attrs
            Xsize = int("".join(ImageAttrs['X'].astype(str)))
            Ysize = int("".join(ImageAttrs['Y'].astype(str)))
            Zsize = int("".join(ImageAttrs['Z'].astype(str)))
        return np.array([Xsize, Ysize, Zsize])

    def getVoxelSize(self):
        vExtMin, vExtMax = self.imExtends
        return [round(x, 2) for x in ((vExtMax - vExtMin) / self.imsize.astype(np.float64))]

    def getSceneName(self, scene):
        with self.try_open('r') as fh:
            lScenes = fh['/Scene8/Content/']
            for n, ss in enumerate(lScenes):
                if n == scene:
                    sName = fh['/Scene8/Content/' + ss].attrs["Name"]
                    break
        return ''.join(sName.astype(str))

    def getSceneNames(self):
        with self.try_open('r') as fh:
            if 'Scene8' in fh:
                lScenes = fh['/Scene8/Content/'].keys()
                sNames = list()
                for c, ss in enumerate(lScenes):
                    if ss[:6] == 'Points':
                        errorscene = 'CoordsXYZR' not in fh['Scene/Content/' + ss]
                        sName = self.getSceneName(c)
                        if errorscene and sName not in self.display_errorspots:
                            self.display_errorspots += [sName]
                            logger.warning(
                                "CoordsXYZR does not appear in {} with name {}. Filename is {}. Discarding points".format(
                                    ss, sName, self.filename))
                        else:
                            sNames.append(sName)
            else:
                sNames = None
        return sNames

    def getChannelName(self, channel):
        f_channel = '/Channel ' + str(channel)
        with self.try_open('r') as fh:
            try:
                cName = fh['/DataSetInfo' + f_channel].attrs['Name']
                rcname = ''.join(cName.astype(str))
            except KeyError:
                logger.warning("Channel {} not found in {}".format(channel, self.filename))
                rcname = None
        return rcname

    def getChannelNames(self):
        with self.try_open('r') as fh:
            lChannels = fh['/DataSet/ResolutionLevel 0/TimePoint 0/'].keys()
            lengthclist = max([int(x.replace('Channel ', "")) for x in lChannels]) + 1
            cNames = [None] * lengthclist
            for nchannel in lChannels:
                try:
                    cName = fh['DataSetInfo'][nchannel].attrs['Name']
                    cname = ''.join(cName.astype(str))
                except KeyError:
                    logger.warning("Channel {} not found in {}".format(nchannel, self.filename))
                    cname = None
                cNames[int(nchannel.replace("Channel ", ""))] = cname

        # cNames = list()
        # for cc in range(0, nChannels):
        #     cname = self.getChannelName(cc)
        #     if cname:
        #         cNames.append(cname)
        return cNames

    def getVolume(self, channels=0, timepoint=0, resolution=0, limInd=None, spot_rad=None, dtype=None):
        dtype = dtype or self.dtype
        imsize = self.imsize
        if not resolution == 0:
            raise Exception(
                "Method not implemented for different resolutions yet. The default array size must be adapted")
        if limInd is None:
            limInd = [0, imsize[2], 0, imsize[1], 0, imsize[0]]
        limsize = [limInd[1] - limInd[0], limInd[3] - limInd[2], limInd[5] - limInd[4]]
        f_timepoint = '/TimePoint ' + str(timepoint)
        f_resolution = '/ResolutionLevel ' + str(resolution)
        if channels is all:
            lChannels = np.arange(len(self.channelNames))
        elif not ((type(channels) in (tuple, list)) or type(channels) is np.ndarray):
            lChannels = (channels,)
        elif type(channels[0]) is str:
            lChannels = self.channelName2Number(channels)
        else:
            lChannels = channels
        nChannels = len(lChannels)
        vol = np.zeros(shape=limsize + [nChannels], dtype=dtype)

        def fromChannel(channel):
            f_channel = '/Channel ' + str(channel)
            limIndclip = np.clip(limInd, 0, None)
            padinds = limIndclip - limInd
            with self.try_open('r') as fh:
                dataset = fh['/DataSet' + f_resolution + f_timepoint + f_channel + '/Data']
                im_aux = dataset[limIndclip[0]:limIndclip[1], limIndclip[2]:limIndclip[3], limIndclip[4]:limIndclip[5]]
            if (padinds > 0).any():
                im_aux = np.pad(im_aux, np.reshape(padinds, (3, 2)), 'constant')
            if dtype is not float:
                im_aux = convert_uint8(im_aux)
            return im_aux

        def fromSpot(count):
            return self.getSpotsVol(channels[count], limInd=limInd, spot_rad=spot_rad)

        count = -1
        for channel in lChannels:
            count += 1
            if channel > -1:
                vol[..., count] = fromChannel(channel)
            else:
                vol[..., count] = fromSpot(count)
        self.reset()
        return vol

    def getSpotsObj(self, channel, spot_rad=None):
        if spot_rad is not None:
            spot_rad_um = spot_rad
        else:
            spot_rad_um = self.getSpotsRadii(channel)
        cspots = Spots(self.getSpotsCoordinates(channel), voxelSize=self.voxelSize, imExtends=self.imExtends,
                       imsize=self.imsize, radius_um=spot_rad_um)
        return cspots

    def getSpotsVol(self, channel, spot_rad=None, limInd=None):
        cspots = self.getSpotsObj(channel, spot_rad=spot_rad)
        spVol = cspots.gaus_spots2image(limInd=limInd)
        spVol = np.transpose(spVol, (2, 1, 0))
        return spVol

    def create_hiddenData(self, channel, shape, voxel_size=None, dname='resized', chunk_size=None,
                          cdesc='No description'):
        dname_aux = "DataSet " + dname
        with self.try_open('r+') as fh:
            if dname_aux not in fh:
                aux_group1 = fh.create_group("DataSet " + dname)
                aux_group2 = aux_group1.create_group("ResolutionLevel 0")
                dataset = aux_group2.create_group("TimePoint 0")
            else:
                dataset = fh.get(dname_aux + "/ResolutionLevel 0/TimePoint 0")
            cname = "Channel " + str(channel)
            if dataset is not None and cname in dataset:
                dataset.__delitem__(cname)
            c_group = dataset.create_group(cname)
            chunks = True if chunk_size is None else tuple(np.min([tuple(chunk_size), shape], axis=0))
            c_group.create_dataset("Data", shape=shape, dtype=self.dtype, chunks=chunks)
            if voxel_size is not None:
                c_group.attrs['VoxelSizeZ'] = np.array([x for x in str(voxel_size[0])], dtype='|S1')
                c_group.attrs['VoxelSizeY'] = np.array([x for x in str(voxel_size[1])], dtype='|S1')
                c_group.attrs['VoxelSizeX'] = np.array([x for x in str(voxel_size[2])], dtype='|S1')
            c_group.attrs['Description'] = np.array([x for x in cdesc], dtype='|S1')
        self.reset()

    def write_info(self, channel, cname='CNNseg', cdesc='No description'):
        # Write info
        with self.try_open('r+') as fh:
            info_group = fh.get("DataSetInfo")
            cgroup = "Channel " + str(channel)
            if cgroup in info_group:
                info_group.__delitem__(cgroup)
            infoc = info_group.create_group(cgroup)
            infoc.attrs['Color'] = np.array([x for x in "1.000 1.000 1.000"], dtype='|S1')
            infoc.attrs['ColorMode'] = np.array([x for x in "BaseColor"], dtype='|S1')
            infoc.attrs['ColorOpacity'] = np.array([x for x in "1.000"], dtype='|S1')
            infoc.attrs['ColorRange'] = np.array([x for x in "0.000 255.000"], dtype='|S1')
            infoc.attrs['Description'] = np.array([x for x in cdesc], dtype='|S1')
            infoc.attrs['GammaCorrection'] = np.array([x for x in "1.000"], dtype='|S1')
            infoc.attrs['Name'] = np.array([x for x in cname], dtype='|S1')

    def write_channelAttrs(self, c_group, data_shape):
        c_group.attrs["HistogramMax"] = np.array([x for x in "255.000"], dtype='|S1')
        c_group.attrs["HistogramMin"] = np.array([x for x in "0.000"], dtype='|S1')
        c_group.attrs["ImageSizeX"] = np.array([x for x in str(data_shape[2])], dtype='|S1')
        c_group.attrs["ImageSizeY"] = np.array([x for x in str(data_shape[1])], dtype='|S1')
        c_group.attrs["ImageSizeZ"] = np.array([x for x in str(data_shape[0])], dtype='|S1')

    def create_channel(self, chunk_size=None, dtype=None):
        dtype = dtype or self.dtype
        imsize = self.imsize
        newc = len(self.channelNames)
        # Write info
        self.write_info(channel=newc, cname='Empty', cdesc='Empty channel')
        with self.try_open('r+') as fh:
            dataset = fh.get("DataSet")
            resLev = len(dataset)
            for r in range(resLev):
                r_group = dataset.get("ResolutionLevel " + str(r) + "/TimePoint 0")
                c_group = r_group.create_group("Channel " + str(newc))
                dset_shape = r_group.get("Channel 0/Data").shape
                # datafile = np.zeros(shape=dset_shape, dtype=self.dtype) #Not needed: sometimes Imaris uses crazy big numbers
                self.write_channelAttrs(c_group, [imsize[2], imsize[1], imsize[0]])
                chunks = True if chunk_size is None else tuple(np.min([tuple(chunk_size), dset_shape], axis=0))
                c_group.create_dataset("Data", shape=dset_shape, dtype=dtype, chunks=chunks)
        self.reset()
        return newc

    def write_dataset_patches(self, lPatch, channel=None, dname=None, limInd=None, cname='CNNseg',
                              cdesc='No description'):
        # limInd = [lbx, ubx, lby, uby, lbz, ubz]
        if limInd is None:
            self.write_dataset(lPatch, channel, dname=dname, cname=cname, cdesc=cdesc)
            return None
        if not channel:
            channel = self.create_channel()
            self.write_info(channel, cname, cdesc)
        lPatch = convert_uint8(lPatch)
        dname_aux = "DataSet" if dname is None else "DataSet " + dname
        with self.try_open('r+') as fh:
            dataset = fh.get(dname_aux + "/ResolutionLevel 0/TimePoint 0")
            c_group = dataset.get("Channel " + str(channel))
            patch_shape = lPatch.shape
            # self.write_channelAttrs(c_group, data_shape) #TODO: if channels exists, do we need this?
            for nim, vol in enumerate(lPatch):
                ind = limInd[nim]
                if (np.array(ind) < 0).any():
                    logger.warning("Patch contains negative indices, skipping... This can cause some tiling problems")
                    continue
                indpad = np.array([[max([0, -ind[0]]), max([0, ind[1] - self.imsize[2]])],
                                   [max([0, -ind[2]]), max([0, ind[3] - self.imsize[1]])],
                                   [max([0, -ind[4]]), max([0, ind[5] - self.imsize[0]])]])
                if not (indpad == 0).all():
                    ind = [max([ind[0], 0]), min([ind[1], self.imsize[2]]),
                           max([ind[2], 0]), min([ind[3], self.imsize[1]]),
                           max([ind[4], 0]), min([ind[5], self.imsize[0]])]
                    findpad = indpad.flatten()
                    if len(vol.shape) == 2:
                        vol = vol[findpad[2]:patch_shape[1] - findpad[3],
                              findpad[4]:patch_shape[2] - findpad[5]]
                    elif len(vol.shape) == 3:
                        vol = vol[findpad[0]:patch_shape[1] - findpad[1], findpad[2]:patch_shape[2] - findpad[3],
                              findpad[4]:patch_shape[3] - findpad[5]]
                    else:
                        raise Exception('Dimensionality unknown')
                if vol.ndim == 2:
                    vol = np.expand_dims(vol, axis=0)
                c_group['Data'][ind[0]:ind[1], ind[2]:ind[3], ind[4]:ind[5]] = vol
        self.reset()
        # After completing the dataset, read it, and write the different resolution levels

    def delete_channel(self, channel):
        logger.warning("Unfinished method. Deleting channels in hdf5 breaks a lot of other methods")
        nchannel = self.channelName2Number([channel])[0]
        while nchannel > -1:
            with self.try_open('r+') as fh:
                # Delete info
                info_group = fh.get("DataSetInfo")
                cgroup = "Channel " + str(channel)
                if cgroup in info_group:
                    info_group.__delitem__(cgroup)
                # Delete channels
                dataset = fh.get("DataSet")
                resLev = len(dataset)
                for r in range(resLev):
                    r_group = dataset.get("ResolutionLevel " + str(r) + "/TimePoint 0")
                    del r_group["Channel " + str(nchannel)]
            self.reset()
            nchannel = self.channelName2Number([channel])[0]

    def write_resLev_patches(self, channel, csize=None):
        if csize is None:
            csize = [500, 500, 500]
        d1 = np.append(np.arange(0, self.imsize[2], csize[0]), self.imsize[2])
        d2 = np.append(np.arange(0, self.imsize[1], csize[1]), self.imsize[1])
        d3 = np.append(np.arange(0, self.imsize[0], csize[2]), self.imsize[0])
        inds = []
        for i1 in range(1, len(d1)):
            for i2 in range(1, len(d2)):
                for i3 in range(1, len(d3)):
                    inds.append([d1[i1 - 1], d1[i1], d2[i2 - 1], d2[i2], d3[i3 - 1], d3[i3]])
        with self.try_open('r+') as fh:
            dataset = fh.get("DataSet")
            resLev = len(dataset)
            c_group = [None] * resLev
            rsize = [None] * resLev
            zoom_aux = [None] * resLev
            r_group = dataset.get("ResolutionLevel 0/TimePoint 0")
            c_group[0] = r_group.get("Channel 0")
            imsizex = int("".join(r_group.get('Channel 0').attrs['ImageSizeX'].astype(np.str)))
            imsizey = int("".join(r_group.get('Channel 0').attrs['ImageSizeY'].astype(np.str)))
            imsizez = int("".join(r_group.get('Channel 0').attrs['ImageSizeZ'].astype(np.str)))
            rsize[0] = [imsizez, imsizey, imsizex]
            for cind, ind in enumerate(inds):
                image = self.getVolume(channel, limInd=ind)[..., 0]
                for r in range(1, resLev):
                    if cind == 0:
                        r_group = dataset.get("ResolutionLevel " + str(r) + "/TimePoint 0")
                        c_group[r] = r_group.get("Channel " + str(channel))
                        imsizex = int("".join(r_group.get('Channel 0').attrs['ImageSizeX'].astype(np.str)))
                        imsizey = int("".join(r_group.get('Channel 0').attrs['ImageSizeY'].astype(np.str)))
                        imsizez = int("".join(r_group.get('Channel 0').attrs['ImageSizeZ'].astype(np.str)))
                        rsize[r] = [imsizez, imsizey, imsizex]
                        self.write_channelAttrs(c_group[r], rsize[r])
                        # zoom_aux[r] = (np.array(rsize[r-1]) / np.array(rsize[r]))
                        zoom_aux[r] = (np.array(rsize[0]) / np.array(rsize[r]))
                    nsize = np.array(np.round(image.shape / zoom_aux[r]), dtype=np.uint16)
                    zoom_aux2 = image.shape / nsize
                    data = transform.resize(image, nsize, order=0, preserve_range=True,
                                            mode='reflect', anti_aliasing=True).astype(self.dtype)
                    indr_aux = np.around(ind[0::2] / zoom_aux2).astype(np.uint16)
                    indr = np.empty(6, dtype=np.uint16)
                    for i, k in enumerate(indr_aux):
                        nx0 = indr_aux[i]
                        nx1 = nx0 + nsize[i]
                        if nx1 >= rsize[r][i]:
                            nx1 = rsize[r][i]
                            nx0 = nx1 - nsize[i]
                        indr[i * 2] = nx0
                        indr[i * 2 + 1] = nx1
                    try:
                        c_group[r]['Data'][indr[0]:indr[1], indr[2]:indr[3], indr[4]:indr[5]] = data
                    except:
                        print("error")
                    # del data
            image = fh['/DataSet/ResolutionLevel ' + str(r) + '/TimePoint 0/Channel ' + str(channel) + '/Data'][()]
            h = np.histogram(image.flatten(), bins=256, range=(0, 256))[0]
            if "Histogram" in c_group[0]:
                c_group[0]["Histogram"][...] = h
            else:
                c_group[0].create_dataset("Histogram", h.shape, data=h, dtype=np.uint64)
        self.reset()

    def write_resLev(self, channel):  # deprecated
        try:
            image = self.getVolume(channel)[..., 0]
            with self.try_open('r+') as fh:
                dataset = fh.get("DataSet")
                r_group = dataset.get("ResolutionLevel 0/TimePoint 0")
                c_group = r_group.get("Channel " + str(channel))
                resLev = len(dataset)
                h = np.histogram(image.flatten(), bins=256, range=(0, 256))[0]
                if "Histogram" in c_group:
                    c_group["Histogram"][...] = h
                else:
                    c_group.create_dataset("Histogram", h.shape, data=h, dtype=np.uint64)
                for r in range(1, resLev):
                    r_group = dataset.get("ResolutionLevel " + str(r) + "/TimePoint 0")
                    c_group = r_group.get("Channel " + str(channel))
                    imsizex = int("".join(r_group.get('Channel 0').attrs['ImageSizeX'].astype(np.str)))
                    imsizey = int("".join(r_group.get('Channel 0').attrs['ImageSizeY'].astype(np.str)))
                    imsizez = int("".join(r_group.get('Channel 0').attrs['ImageSizeZ'].astype(np.str)))
                    # data = transform.resize(image, [imsizez, imsizey, imsizex], order=0, preserve_range=True,
                    #                         mode='reflect', anti_aliasing=True).astype(self.dtype)
                    data = transform.resize(image, [imsizez, imsizey, imsizex], order=0, anti_aliasing=False,
                                            preserve_range=True, mode='reflect').astype(self.dtype)
                    self.write_channelAttrs(c_group, [data.shape[0], data.shape[1], data.shape[2]])
                    c_group['Data'][0:data.shape[0], 0:data.shape[1], 0:data.shape[2]] = data
                    del data
            self.reset()
        except:
            self.write_resLev_patches(channel)

    def write_dataset(self, image, channel=None, dname=None, cname='CNNseg', cdesc='No description', dtype=None):
        dtype = dtype or self.dtype
        if not channel:
            channel = self.create_channel(dtype=dtype)
        # image = convert_uint8(image)
        dname_aux = "DataSet" if dname is None else "DataSet " + dname
        with self.try_open('r+') as fh:
            dataset = fh.get(dname_aux)
            resLev = range(len(dataset))
            self.write_info(channel, cname, cdesc)
            h = np.histogram(image.flatten(), bins=256, range=(0, 256))[0]
            for r in resLev:
                r_group = dataset.get("ResolutionLevel " + str(r) + "/TimePoint 0")
                c_group = r_group.get("Channel " + str(channel))
                if not type(r) is int:
                    data = image
                elif r == 0:
                    data = image
                    if "Histogram" in c_group:
                        c_group["Histogram"][...] = h
                    else:
                        c_group.create_dataset("Histogram", h.shape, data=h, dtype=np.uint64)
                else:
                    imsizex = int("".join(r_group.get('Channel 0').attrs['ImageSizeX'].astype(np.str)))
                    imsizey = int("".join(r_group.get('Channel 0').attrs['ImageSizeY'].astype(np.str)))
                    imsizez = int("".join(r_group.get('Channel 0').attrs['ImageSizeZ'].astype(np.str)))
                    data = transform.resize(image, [imsizez, imsizey, imsizex], order=0, preserve_range=True,
                                            mode='reflect', anti_aliasing=True).astype(dtype)

                # datafile[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]] = data
                self.write_channelAttrs(c_group, [data.shape[0], data.shape[1], data.shape[2]])
                c_group['Data'][0:data.shape[0], 0:data.shape[1], 0:data.shape[2]] = data
                del data
        self.reset()

    def write_spots(self, Xf_um, channel, spot_rad=None):
        with self.try_open('r+') as fh:
            fsc = fh['Scene/Content']
            fsc8 = fh['Scene8/Content']
            sname = 'Points' + str(len(self.sceneNames))
            scn = self.sceneName2Number([channel])[0]
            npoints = Xf_um.shape[0]
            # Match other points in the 4th dimension (I think it is time)
            try:
                for kpoints in fsc:
                    tval = fsc[kpoints + '/CoordsXYZR'][0, -1]
            except Exception:
                tval = 0.0
            tx = tval * np.ones((npoints, 1))
            X = np.concatenate([Xf_um, tx], axis=-1)
            pgroup = fsc.create_group(sname)
            # pgroup.attrs["Name"] = np.array([x for x in channel], dtype='|S1')
            # pgroup.attrs["Unit"] = np.array([x for x in 'um'], dtype='|S1')
            pgroup.attrs["Name"] = channel
            pgroup.attrs["Unit"] = 'um'
            pgroup.create_dataset("CoordsXYZR", data=X, dtype='float32')
            pgroup.create_dataset("Time", data=np.zeros(shape=(len(Xf_um), 1)))
            pgroup.create_dataset('TimeInfos', data=np.array([b'2017-11-14 10:50:23.688']), dtype='|S23')

            pgroup8 = fsc8.create_group(sname)
            x = np.array([(i, Xf_um[i, 0], Xf_um[i, 1], Xf_um[i, 2], spot_rad) for i in range(npoints)],
                         dtype=[('ID', '<i8'), ('PositionX', '<f4'), ('PositionY', '<f4'), ('PositionZ', '<f4'),
                                ('Radius', '<f4')])
            pgroup8.create_dataset("Spot", data=x)
            pgroup8.attrs["Name"] = np.array([x for x in channel], dtype='|S1')
            pgroup8.attrs["Unit"] = np.array([x for x in 'um'], dtype='|S1')
            pgroup8.attrs["Id"] = 200000 + len(self.sceneNames) + 1

            # pgroup8.create_dataset("CoordsXYZR", data=X, dtype='float32')
            catspots = np.array([(0, b'Spot', b'Spot'), (1, b'Overall', b'Overall')],
                                dtype=[('ID', '<i8'), ('CategoryName', 'S256'), ('Name', 'S256')])
            pgroup8.create_dataset("Category", data=catspots)
            pgroup8.create_dataset('LabelGroupNames',
                                   np.array([]), dtype=[('LabelGroupName', 'S256'), ('EndLabelValue', '<i8')])
            pgroup8.create_dataset('LabelSetLabelIDs', data=np.array([]))
            pgroup8.create_dataset('LabelSetObjectIDs', data=np.array([]))
            pgroup8.create_dataset('LabelSets', data=np.array([]))
            pgroup8.create_dataset('LabelValues', data=np.array([]))

    @staticmethod
    def show_channels(vol, tmpfolder):
        import matplotlib.pylab as plt
        import os
        f, axarr = plt.subplots(3, 3)
        z = 10
        for c in range(min(vol.shape[-1], 9)):
            find = np.unravel_index(c, [3, 3])
            axarr[find].imshow(vol[z, ..., c], cmap='gray')
            axarr[find].set_title('Channel: ' + str(c))
        f.savefig(os.path.join(tmpfolder, 'fig' + str(int(np.random.rand() * 1000)) + '.pdf'), format='pdf', dpi=1200)
