# (c) 2021, Luca Widmer  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
import ij.IJ as IJ
from ij.plugin.filter import Analyzer

#@ ImagePlus imp
#@ File (label='Coordinate Output File', persist=true, required=false,description = " ") fileName

"""
This script wil save points currently selected with the multipoint ROI tool in Fiji as a coordinate.csv file
compatible with the MINTIF Plugin.
"""

# read number of point types
IJ.run("Set Measurements...", "mean standard min centroid stack redirect=None decimal=3");
IJ.run("Clear Results", "");
res = IJ.run(imp, "Measure", "");
results= Analyzer.getResultsTable()
# fiji treats x as columns and y as rows, which is different form our tool, so we switch them
# X,Y measurements are in micron
y_list = results.getColumnAsDoubles(results.getColumnIndex("X"))
x_list = results.getColumnAsDoubles(results.getColumnIndex("Y"))
z_list = results.getColumnAsDoubles(results.getColumnIndex("Slice"))

if z_list is None:
    z_list=[1.0]*len(x_list)
calibration = imp.getCalibration()
# get image dimensions and pixel size
dimensions = list(imp.getDimensions())
y_dim = dimensions[0]
x_dim = dimensions[1]
c_dim = dimensions[2]
z_dim = dimensions[3]
x_scale=calibration.pixelHeight
y_scale=calibration.pixelWidth
z_scale=calibration.pixelDepth

# write csv file
with open(str(fileName), "w") as coord_file:
    # write header
    coord_file.write( "# Image Extends\n")
    coord_file.write( "0,0,0\n")
    coord_file.write("{},{},{}\n".format(str(z_dim),str(x_dim),str(y_dim)))
    coord_file.write( "# Voxel Size\n")
    coord_file.write("{},{},{}\n".format(z_scale,x_scale,y_scale))
    # write coordinates
    coord_file.write("# Coordinates\n" )
    for (x,y,z) in zip(x_list, y_list, z_list):
    # measurments are in micron
        coord_file.write("{},{},{}\n".format(z,x,y))
print("saved coordinates in {}".format(fileName))
