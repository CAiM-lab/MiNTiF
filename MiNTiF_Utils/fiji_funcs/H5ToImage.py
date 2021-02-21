# (c) 2020-2021,  Luca Widmer  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel

import csv
import h5py
import sys
import os
import tifffile
import numpy as np
import site
site.addsitedir(os.getcwd())
from  fiji_funcs import CoordinateUtils


"""
This script reconstructs information form a MINTIF file using the function reconstruct_MINTIF_file() . 
Images (including predicted labels and density maps) or coordinates can be reconstructed independently. 
The Output will be a multichannel tiff file for each sample contained in the MINTIF File (with 
predicted labels and density maps as separate channels), 
as well as separate coordinate files for all predicted Coordinate samples & classes .
reconstruct_MINTIF_file() can also simply return coordinates as Dataframes without saving any information to disk.
"""


def remove_padding(patch, in_patch_indices, padding, image_shape):
    """
    removes padding and returns unpadded patch with new indices (should be equivalent to out patch index...)
    """
    z1, z2, x1, x2, y1, y2 = in_patch_indices
    padd_z, padd_x, padd_y = patch_z1, patch_x1, patch_y1 = padding
    patch_z2, patch_x2, patch_y2 = np.subtract(patch.shape, padding)

    # drop padding for all indices
    z1n = z1 + padd_z
    z2n = z2 - padd_z
    x1n = x1 + padd_x
    x2n = x2 - padd_x
    y1n = y1 + padd_y
    y2n = y2 - padd_y
    # crop padding from patch
    new_patch = patch[patch_z1:patch_z2, patch_x1:patch_x2, patch_y1:patch_y2]
    return new_patch, [z1n, z2n, x1n, x2n, y1n, y2n]

def get_slice(img, indices):
    """
    return image slice by index array
    """
    return img[indices[0]:indices[1], indices[2]:indices[3], indices[4]: indices[5]]


def paste_patch_to_image(img, patch, patch_indices, is_label, padding, image_shape, detection_margin=0.):
    """
    inserts patch in image averages values of label patches if previously written patches overlap wirth current one

    """
    # force patch to be 3D
    if len(patch.shape) == 2:
        patch = np.reshape(patch, (1,) + patch.shape)
    # adapt padding
    if is_label:
        padding_ = [int(round(detection_margin))]*len(padding)
    else:
        padding_ = padding + int(round(detection_margin))
    # calculate new patch without padding
    patch, patch_indices = remove_padding(patch, patch_indices, padding_, image_shape)
    #insert patch
    # if is label, average results in overlapping sections(i.e. sections that already contain output, nans are ignored in average function
    if is_label:
        img[patch_indices[0]:patch_indices[1],
        patch_indices[2]:patch_indices[3],
        patch_indices[4]: patch_indices[5]] = np.nanmean(np.array([img[patch_indices[0]:patch_indices[1],
                                                                   patch_indices[2]:patch_indices[3],
                                                                   patch_indices[4]: patch_indices[5]],
                                                                   patch.astype('float64')]), axis=0)
    else:
        img[patch_indices[0]:patch_indices[1],
        patch_indices[2]:patch_indices[3],
        patch_indices[4]: patch_indices[5]] = patch


    return


def scale_to_255(img, ch_mean=None, ch_std=None):
    """
    Rescale Image to 0 to 255 range
    @param img:
    @param ch_mean:
    @param ch_std:
    @return:
    """
    if ch_mean is None or ch_std is None:
        min_ = np.min(img)
        return (img - min_) * (255 / (np.max(img) - min_))
    else:
        res=img*ch_std + ch_mean
        return np.round(res-np.min(res))

def dist2(p1, p2):
    """
    calculate 3d euclidean distance between points.
    """
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2

def fuse_points(points, d):
    """
    UNUSED. Fuses points in list that are closer than distance d.
    @param points: list of points.
    @param d: maximum distance that should still fuse points.
    @return: list of fused points.
    """
    #todo I think this expects list of tuples => change to nd array
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i+1, n):
                if dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count+=1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            ret.append((point[0], point[1]))
    return ret

def reconstruct_MINTIF_file(file_path, image_reconstruction=True, coordinate_reconstruction=True, reconstruct_dms=False,
                            safe_coord_files = True, ids=None, delete_test_channel=False):

    # load file
    f = h5py.File(file_path, 'r+')
    # initialize
    temp_memmap = ''
    # initialize return lists. these returns are currently only used in model testing.
    image_file_output, pred_coords, testing_coords, GT_coords = [], [], [], []
    # used to delete test datasets after Testing
    test_datasets = []
    outputs= [image_file_output, pred_coords, testing_coords, GT_coords]

    for sample_num, (sample_name, sample) in enumerate(f.items()):
        # append empty sublist for each sample
        for op in outputs:
            op=op.append([])
        # skip samples, used during testing to not reconstruct non-test samples
        if ids is not None and sample_num not in ids:
            continue
        # try clause over whole reconstruction to avoid one corrupt sample breaking reconstruction  of other samples
        try:
            channel_names = list(sample["Patch_0"].keys())

            # try clause is legacy code
            try:
                voxel_size=sample.attrs["voxel_size"]
            except:
                voxel_size=[1,1,1]
                print("WARNING: Did not find voxel size, set to [1,1,1]")

            dm_labels = [x for x in channel_names if "pred_label_dm_" in x or "testing_label_dm_" in x]
            dm_labels = [x for x in channel_names if "label_dm_" in x]

            channel_is_label = ["label_channel" in x for x in channel_names]
            # parse out image size
            original_image_shape = sample.attrs["image_shape"]
            out_shape = original_image_shape.copy()
            out_shape[1] += sum(["pred_label_channel" in x for x in channel_names])

            # also reconstruct GT DMs in reconstruct_DMs mode
            if reconstruct_dms:
                out_shape[1] += len(dm_labels)
            #setup coordinate dictionary
            out_coord_dict = {i: np.empty(shape=(0, 3), dtype='int32') for i in dm_labels}

            #set up temporary memory map to save output image
            if image_reconstruction:
                # if patch size in z was chosen larger than image
                first_channel = list(sample["Patch_0"].keys())[0]

                first_patch_shape = list(sample["Patch_0/{}".format(first_channel)].shape)
                if len(first_patch_shape) == 2:
                    first_patch_shape = [1] + first_patch_shape
                out_shape[0] = max(out_shape[0], first_patch_shape[0])
                out_shape[2] = max(out_shape[2], first_patch_shape[1])
                out_shape[3] = max(out_shape[3], first_patch_shape[2])

                # define a memory map array to temporarily save the patches to to save on RAM
                temp_memmap = os.path.join(os.path.dirname(hdf5_file_path) , "temp.memmap")
                if os.path.exists(temp_memmap):
                    os.remove(temp_memmap)
                res_image = np.memmap(temp_memmap, dtype='float32', mode='w+', shape=tuple(out_shape))
                res_image[...] = None
                # pred_dm_ind = []

            for j, (patch_name, patch) in enumerate(sample.items()):
                # if j % 100 == 0:
                #     print("patch {} of {}".format(j, len(sample.keys())))
                # parse patch attributes
                inds_channel = patch.attrs["inds_channel"]
                inds_label = patch.attrs["inds_label"]
                overlap_patch = patch.attrs["overlap_patch"]
                do_pad=patch.attrs["do_pad"]

                if do_pad:
                    padsize_patch=overlap_patch*0
                else:
                    padsize_patch = (np.round(overlap_patch / 2)).astype(np.uint16)
                #remove additional padding necesary in detection
                if sample.attrs["dataset_type"] == 'detection':
                    detection_margin = sample.attrs["radius_pixel"]
                else:
                    detection_margin = 0


                non_coord_counter =0
                for k, (channel_name, channel_patch) in enumerate(patch.items()):

                    # skip coordinate channels (add here anay other future channels that should be skipped)
                    if "coordinates_" in channel_name:
                        continue
                    if "test" in channel_name:
                        test_datasets += [channel_patch.name]

                    # predict coordinates and save them in dictionary to write to csv later
                    if coordinate_reconstruction and channel_name in out_coord_dict.keys():
                        # pred_dm_ind.append(non_coord_counter)
                        cell_radius = round(sample.attrs["radius_um"])

                        # predict coordinates in patch

                        coords = CoordinateUtils.predict_coordinates(channel_patch[:].astype(float), cell_radius)
                        #convert to 3D
                        if len(coords)>0 and coords.shape[1] == 2:
                            coords=np.c_[np.ones(coords.shape[0]),coords]
                            # append coordinates to output array if they are not already there (overlapping patches)
                        for c in coords:
                            zcc = c + inds_label.take((0,2,4))
                            #if zcc not in out_coord_dict[channel_name].tolist():

                            # sum coordinates with inds_label to receive coordinates in whole image reference frame
                            out_coord_dict[channel_name] = np.vstack([out_coord_dict[channel_name],
                                                                     np.round(zcc)])

                    # only reconstruct image if demanded. additionally reconstruct density maps if in reconstruct_DMs mode
                    if image_reconstruction and ("in_channel" in channel_name or "label_channel" in channel_name or reconstruct_dms):
                        # check i channel is input channel
                        is_label = "in_" not in channel_name
                        # convert data to ndarray
                        # write patch to image
                        paste_patch_to_image(res_image[:, non_coord_counter, :, :], channel_patch[:],
                                             # select indices for label or marker
                                             inds_label if is_label else inds_channel,
                                             is_label, padsize_patch, res_image.shape, detection_margin)

                        # if sample_num%500==0 and sample_num>0:
                        #     print(inds_label if is_label else inds_channel)
                        #     print(non_coord_counter)

                        non_coord_counter += 1


            # finish image reconstruction
            if image_reconstruction:
                # cut remaining padding
                # check if any dimension except number of channels does not match expected output
                if any(np.array(res_image.shape).take([0,2,3]) != original_image_shape.take([0,2,3])):
                    # if res_image.shape[2:] != out_shape[2:]:
                    res_image = res_image[:original_image_shape[0], :, :original_image_shape[2], :original_image_shape[3]]
                # channel_is_label = ["label" in i for i in channel_names]

                # check for nans in output, warn and replace with 0s
                nan_ind = np.argwhere(np.isnan(res_image))
                if np.sum(nan_ind) > 0:
                    print("WARNING, found NaN values in reconstructed image and filled with 0")
                    print("Slices:\n", np.unique(nan_ind[1:, :2]), flush=True)
                    np.nan_to_num(res_image, copy=False, nan=0)

                # all_labels_in_slice_contain_anot = np.sum(res_image[:, channel_is_label, ...], axis=(2, 3)).all(axis=1)
                for c in range(original_image_shape[1]):
                    if not channel_is_label[c]:
                        # res_image[:, c, ...] = scale_to_255(res_image[:, c, ...])
                        try:
                            ch_mean = sample.attrs["mean_{}".format(channel_names[c])]
                            ch_std = sample.attrs["std_{}".format(channel_names[c])]
                        except:
                            ch_mean = None
                            ch_std = None
                        res_image[:, c, ...] = scale_to_255(res_image[:, c, ...],ch_mean=ch_mean, ch_std=ch_std)


                # SAVE RESULT IMAGE
                if np.min(res_image) < 0 and np.max(res_image) > 255:
                    print("something went wrong while scaling image back  to [0,255] range. squash to [0,255] range")
                    res_image[res_image < 0] = 0
                    res_image[res_image > 255] = 255
                image_file = os.path.join(os.path.dirname(hdf5_file_path), "{}_{}_{}_reconstructed.tiff".format(
                                        sample_name,os.path.basename(hdf5_file_path).split(".")[0],
                                        sample.attrs["name"].split(".")[0]))
                image_file_output += [image_file]
                print("Saving image to {}".format(image_file), flush=True)
                tifffile.imsave(image_file, res_image.astype('uint8'), imagej=True)
                res_image._mmap.close()
                os.remove(temp_memmap)
            # --------------------------
            # Finish Predict Coordinates
            # --------------------------
            if coordinate_reconstruction:
                # WRITE PREDICTED COORDINATES FROM PRED_LABEL_DM CHANNELS
                for name, coords in out_coord_dict.items():
                    if len(coords)>0:
                        # remove duplicate coordinates
                        # apparently quick way to remove duplicate coordinates in one line
                        coords=np.vstack(list({tuple(row) for row in coords}))
                        # sort by first column for easiear reading
                        coords=coords[coords[:, 0].argsort()]
                        # format image extends for csv file
                        img_extend=[np.array([0,0,0]),np.round(out_shape.take([0,2,3])*voxel_size,1)]
                        # append coordinates + metadata to output list
                        if ('pred_label_dm' in name):
                            pred_coords[sample_num]+=[img_extend,voxel_size, coords]
                        elif ("testing_label_dm_" in name):
                            testing_coords[sample_num]+=[img_extend,voxel_size, coords]
                        elif ('label_dm' in name):
                            GT_coords[sample_num]+=[img_extend,voxel_size, coords]
                        # write coordinates to csv only for predicted coordinates
                        if safe_coord_files and 'pred_label_dm' in name:
                            csv_filename="{}_{}_pred_coordinates_{}.csv".format(sample.attrs["name"].split(".")[0],
                                                                                sample_name, name)
                            coord_file = os.path.join(os.path.dirname(hdf5_file_path), csv_filename)
                            with open(coord_file, 'w', newline='') as csvfile:
                                # write header info
                                w = csv.writer(csvfile, delimiter=',')
                                csvfile.write("# Image Extends\n")
                                w.writerow(tuple(img_extend[0]))
                                w.writerow(tuple(img_extend[1]))
                                csvfile.write("# Voxel Size\n")
                                w.writerow(tuple(voxel_size))
                                csvfile.write("# Coordinates\n")
                                # write coordinates of points in order:Z,X, Y
                                for r in coords:
                                    w.writerow(np.round(r*voxel_size,1))
                    else:
                        print("could not predict coordinates (predicted coordinates array empty)")

        # during exception clean up memmap
        except Exception as e:
            print("encountered error when reconstructing {}:\n{}\n continuing with next Sample".format(sample_name, e))
            # try to remove memory map
            if os.path.isfile(temp_memmap):
                try:
                    os.remove(temp_memmap)
                except:
                    try:
                        res_image._mmap.close()
                        os.remove(temp_memmap)
                    except Exception as e:
                        print("encountered error when deleting {}:\n{}\n continuing with next Sample".format(
                            temp_memmap, e))
        else:
            print("finished reconstruction of {}".format(sample_name))

    if delete_test_channel:
        for ds in test_datasets:
            del f[ds]
    if image_reconstruction:
        print("done")
    return image_file_output, np.array(pred_coords), np.array(testing_coords), np.array(GT_coords)


if __name__ == "__main__":
    # path to hdf5 file
    hdf5_file_path = sys.argv[1]
    image_reconstruction = eval(sys.argv[2])
    coordinate_reconstruction = eval(sys.argv[3])
    reconstruct_DMs = eval(sys.argv[4])
    reconstruct_MINTIF_file(hdf5_file_path, image_reconstruction, coordinate_reconstruction, reconstruct_DMs)
