# (c) 2020-2021, Luca Widmer  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
import csv
import os
from cnn.metrics_keras import CellMetrics
from fiji_funcs import H5ToImage
import numpy as np
import pandas as pd

"""
various utility function used by background scripts to work with coordinates.
"""


def read_coord_file(path, desired_voxel_size, convert_coords=True):
    """
    Function Reads coordinate file and returns voxel coordinates.
    assumes distances in file are in um  & the following format
    # {Title}
    image_extend[0]
    image_extend[1]
    # {Title}
    voxel_size
    # {Title}
    Coordinates
    """
    # parse coordinate file
    coords = []
    with open(path, 'r') as r:
        reader = csv.reader(r, quoting=csv.QUOTE_NONNUMERIC)
        _1 = r.readline()
        image_extend = [next(reader), next(reader)]
        _2 = r.readline()
        voxel_size = next(reader)
        _3 = r.readline()
        assert "#" in _1 and "#" in _2 and "#" in _3, "Problem parsing the coordinate csv file"
        for row in reader:
            coords.append(row)
        coords=np.array(coords)
        if convert_coords:
            coords = convert2voxels(coords, image_extend, desired_voxel_size)
    return voxel_size, coords


def convert2voxels(x_um_rw, imExtends, voxelSize):
    """
    Converting from real world um coordinates to 0 origin voxel.

    :param x_um_rw: coordinates in real world frame, dimensions in um
    :param imExtends (list of lists): the first list are the initial extends of the image, and the second list the
     final ones. Dimensions are um and they are used to localize the image in the real world frame
    :param voxelSize: voxel size
    :return: coordinates in 0 centered frame, dimensions in voxels
    """

    # First we bring the coordinates origin to 0
    x_um_0 = x_um_rw - imExtends[0]
    # And then we transform the dimensions to voxels
    X_voxel_0 = x_um_0 / voxelSize
    return X_voxel_0


def get_scaled_voxel_coords(coord_file_path, desired_voxel_size, convert_coords=True):
    """
    gets Coordinates from file and transforms them to scaled voxel dimensions (ZXY).
    :param desired_voxel_size: desired voxel size in um in ZXY dimensions
    :return: coordinates in 0 centered frame, dimensions in rescaled voxels in ZXY dimension
    """
    #returns coordinates in micrometers
    original_voxel_size, coords = read_coord_file(coord_file_path,desired_voxel_size, convert_coords, )
    # transform to desired voxel size
    if len(desired_voxel_size)==2:
        desired_voxel_size = np.append(1, desired_voxel_size)
        print("converted desired voxel size to 3D (ZXY): {}".format(desired_voxel_size))
    if len(original_voxel_size) == 2:
        original_voxel_size = np.append(1, original_voxel_size)
        print("converted original_voxel_size size to 3D (ZXY): {}".format(original_voxel_size))
    return coords, original_voxel_size


def predict_coordinates(density_map, cell_radius, detect_threshold=15):
    """
    predicts coordinates from density map.

    @param density_map: 3D image containing density estimation for coordinates.
    @param cell_radius: minimum distance from other peaks AND image border (should be compensated by adding additional padding earlier in pipeline).
    @param detect_threshold:
    @return: coordinates in 3D (ZXY)
    """
    cell_metrics = CellMetrics()
    # set parameters for detection
    cell_metrics.mindist = cell_radius
    cell_metrics.crop_border = cell_radius
    cell_metrics.detect_threshold = detect_threshold
    # predict and return coordinates
    return cell_metrics.peak_detect(density_map)


def get_hungarian_matching_from_files(Xgt_file, Xpred_file):
    """
    depreciated. calculates metricws from two csv coodinate files. used for 'manual' testing.
    @param Xgt_file: path to first coordinate file
    @param Xpred_file: path to second coordinate file
    @return: metrics as Dictionary
    """
    Xgt,_1= get_scaled_voxel_coords(Xgt_file, [1,1,1])
    Xpred,_2 = get_scaled_voxel_coords(Xpred_file, [1,1,1])
    cell_metrics = CellMetrics()
    metrics = cell_metrics.gt_match(Xpred, Xgt)
    # print(metrics)
    return metrics


def hungarian_test_on_samples (data_file , output_folder, sample_ids=None):
    """
    Produces metrics files and saves to output location
    @param data_file: path to hdf5 data file.
    @param output_folder: path to output folder
    @param sample_ids: ids of samples that should be tested. All if None
    @return: paths to metrics file and metrics pd.Dataframe
    """
    # predict coordinates and reconstruct GT coordinates from samples, delete test_channels from file.
    _, _, testing_coords, GT_coords=H5ToImage.reconstruct_MINTIF_file(data_file, False, True, safe_coord_files=False,
                                                                      ids=sample_ids, delete_test_channel=True)
    # calculate metrics using hungarian algorithm and save to csv file
    out=pd.DataFrame()
    for (id, pC, gtC) in zip(sample_ids, testing_coords[sample_ids], GT_coords[sample_ids]):
        cell_metrics = CellMetrics()
        # calculate metrics
        metrics=cell_metrics.gt_match(pC[2], gtC[2])
        # format and remove error lists
        metrics.pop('derror_all')
        metrics.pop('derror_pos')
        res=pd.DataFrame(metrics, index=[0])
        res["sample_id"]=id
        out =out.append(res, ignore_index=True)
    # write metrics file to csv
    path=os.path.join(output_folder,"metrics.csv")
    out.to_csv(path, index=False)
    # return paths and metrics, currently unused
    return out, path

