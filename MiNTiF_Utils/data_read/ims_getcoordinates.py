# (c) 2018-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
import argparse
import os
import sys
sys.path.append(".")
from data_read.imarisfiles import ImarisFiles
import numpy as np
import csv

"""
This script produces a coordinate CSV from a ims file which contains the real world extends of the image, 
the Voxel Size and the Coordinates of all points in XYZ dimension in um.
"""


def ims_to_coordinate_csv(path, names=None):
    imf = ImarisFiles(path)
    snames = names or imf.getSceneNames()
    file_extends = imf.imExtends
    file_voxelsize = imf.voxelSize
    for scene in snames:
        try:
            X_um_rw = imf.getSpotsCoordinates(scene)
        except:
            print("Scene {} could not be extracted from {}. Check if it is corrupted or not a spot".
                  format(scene, args.filename))
            continue

        #  Write to csv the following variables: X_um_rw, file_extends, file_voxelsize.
        #  This is what you assume the user provides you
        out_name=os.path.join(os.path.dirname(path), "coordinates_{}_{}.csv".format(scene, os.path.basename(path)))
        with open(out_name, 'w', newline='') as csvfile:
            w = csv.writer(csvfile, delimiter=',')
            csvfile.write("# Image Extends\n")
            w.writerow(file_extends[0][[2,1,0]])
            w.writerow(file_extends[1][[2,1,0]])
            csvfile.write("# Voxel Size\n")
            # write  pixel dims as different format in order:(Z) pixelDepth, (X) pixelHeight, (Y) pixelWidth,
            w.writerow(np.array(file_voxelsize).take([2,1.,0]))
            csvfile.write("# Coordinates\n")

            # write coordinates of points in order:X, Y, Z
            for r in X_um_rw:
                w.writerow(r.take([2,1,0]))

    # print(convert2voxels(X_um_rw, file_extends,file_voxelsize))
    print('done')




if __name__=="__main__":
    path=r"D:\Documents\Data_Master_Thesis\cell_detection\Dataset_creation_december\20171114_CARcquant_WT_0_20x_meta_ROI1.ims"
    assert os.path.exists(path)
    ims_to_coordinate_csv(path)
    sys.exit(0)




parser = argparse.ArgumentParser("Arguments for file conversion")

parser.add_argument('--filename', '-f', required=False, type=str, help="Imaris file (.ims) path")
parser.add_argument('--names', '-n', required=False, default=None, nargs='+',
                    help="Spots names. Leave empty to use all")

args = parser.parse_args()
imf = ImarisFiles(args.filename)
snames = args.names or imf.getSceneNames()

# filename=r"C:\Users\Luca Widmer\Desktop\Neuer Ordner\test_new\detection\20171114_CARcquant_IFNa_IFNg_d14_20x_dia_CARc_ROI1.ims"
# imf = ImarisFiles(filename)
# snames = imf.getSceneNames()





# def convert2voxels(x_um_rw, imExtends, voxelSize):
#     """
#     Converting from real world um coordinates to 0 origin voxel.
#
#     :param x_um_rw: coordinates in real world frame, dimensions in um
#     :param imExtends (list of lists): the first list are the initial extends of the image, and the second list the
#      final ones. Dimensions are um and they are used to localize the image in the real world frame
#     :param voxelSize: voxel size
#     :return: coordinates in 0 centered frame, dimensions in voxels
#     """
#
#     # First we bring the coordinates origin to 0
#     x_um_0 = x_um_rw - imExtends[0]
#     # And then we transform the dimensions to voxels
#     X_voxel_0 = np.around(x_um_0 / voxelSize).astype(np.int64)
#     return X_voxel_0

ims_to_coordinate_csv(imf,snames)
