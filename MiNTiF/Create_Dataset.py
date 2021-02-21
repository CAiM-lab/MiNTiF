# (c) 2021, Luca Widmer  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel

import shutil
import site
import logging
import sys
import os
import json

from ij import IJ, ImagePlus, WindowManager
from ij.plugin import Duplicator
from ij.gui import GenericDialog
from fiji.util.gui import GenericDialogPlus
from ij.io import FileSaver
# this is necessary to load owned modules in Jython runtime

wd = os.path.join(os.getcwd(), *["MiNTiF_Utils", "fiji_funcs"])
site.addsitedir(wd)
import FijiUtils


"""
This Script implements a GUI for FIJI to create a Datafile compatible with the MINTIF plugin.
It expects an multichannel image loaded in FIJI either as normal Stack or Virtual Stack.
During conversion, slices of the image are temporary saved onto the Harddrive as Tiffs and ater deleted.
"""

# DO NOT CHANGE, These are The GUI Parameters
#@ File (label='Datafile', persist = true, required=true, description = "create new datafile with this path or append to existing file") data_file
#@ File (label='Model file', persist=true, required=true,description = "model .json file corresponding to datafile/experiment") model_file
#@ String (label='Dataset Type ', choices={"Training", "Testing"}, style="radioButtonHorizontal")  training_data
#@ String (label='Slice Type', choices={"2D", "3D"}, style="radioButtonHorizontal")  anot_type
#@ String (label='Task', choices={"Segmentation", "Detection"}, style="radioButtonHorizontal")  prediction_type
#@ File[]  (label="Coordinate files",  value="",style="file", persist=false) listOfCoordinatePaths
#@ Boolean (label="Compress?", description = "this reduces the file size but may increase overall runtime") compress
# String (label="Voxel Size", value=[1, 1, 1], persist=true,description = "desired Voxel Size in um; (Z),X,Y. Used to rescale images in dataset to identical resolution. only give X,Y dimensions if you work with 2D patches") voxel_size

def make_string_float_array(string):
    string = string.strip()[1:-1]
    return list(map(float, string.split(',')))

def import_dialog(no_channels):
    """
    Generates GUI for Users to input channel information. One line per channel
    @param no_channels: the number of channels in the image
    @return: Dictionary of channel information
    """
    gd = GenericDialog('Channel Information')
    # header text
    gd.addMessage("Channel ID needs to be the identical for each marker over all samples.")
    gd.addMessage("'Information' (optional) is saved in hdf5 file and should identify the channel type/marker used")
    gd.addMessage('     Channel ID                            Channel Type/Skip?                         Information')
    # create options. one line per channel
    for i in range(no_channels):
        gd.addStringField('Channel_' + str(i + 1), "", 0)
        gd.addToSameRow()
        gd.addChoice('Type: ', ["Skip Channel", "Marker", "Label"], "Skip Channel")
        gd.addToSameRow()
        gd.addStringField("Information: ", "")
    gd.showDialog()
    choices = []
    user_metadata = []
    channel_ind = []
    # read user input
    if gd.wasOKed():
        for i in range(no_channels):
            choices.append(gd.getNextChoice())
            inp = gd.getNextString()
            channel_ind.append(int(inp) if inp != "" else -1)
            user_metadata.append(gd.getNextString())
        return channel_ind, choices, user_metadata
    else:
        sys.exit(0)


def coordinates_dialogue(listOfCoordinatePaths):
    """
    Generates GUI for Users to input coordinate information. One line per coordinate file
    @param listOfCoordinatePaths: list of the paths to the coordinate channel
    @return: Dictionary of coordinate information
    """

    gd = GenericDialogPlus('3D Point Coordinates')
    gd.addMessage("The Voxel Size and the coordinates of all points in ZXY dimension in micro meters.")
    gd.addMessage("The IDs for these coordinate channels should be consistent over the whole project")
    # header text
    # create options. one line per channel

    for i, p in enumerate(listOfCoordinatePaths):
        gd.addNumericField('File: {} Channel ID: '.format(os.path.basename(str(p))), i, 0)
        gd.addToSameRow()
        gd.addStringField("Information: ","")
    gd.showDialog()
    # read user input
    if gd.wasOKed():
        CoordChannels = {}
        for i, p in enumerate(listOfCoordinatePaths):
            Channel_i = {}
            Channel_i["FilePath"] = str(p)
            Channel_i["Info"] = gd.getNextString()
            Channel_i["label"] = int(gd.getNextNumber())
            # Channel_i["cell_diam"] =gd.getNextNumber()
            CoordChannels["Channel_{}".format(i)] = Channel_i
        return CoordChannels
    else:
        sys.exit(0)


def get_zoom(curren_voxel_dim, desired_voxel_dim):
    """
    Calculates zoom factor per imaeg axis to resize images to desired voxel size, expects voxel size in (Z),X,Y coordinates
    @param curren_voxel_dim: original voxel dimensions
    @param desired_voxel_dim: target voxel dimensions
    @return: 3D Zoom factor in ZXY
    """
    # handle cases of 2D and 3D patches
    # Round for numerical consistency
    if len(desired_voxel_dim) == 2:
        x_zoom = round(float(curren_voxel_dim[1]) / float(desired_voxel_size[0]), 3)
        y_zoom = round(float(curren_voxel_dim[2]) / float(desired_voxel_size[1]), 3)
        z_zoom = 1

    elif len(desired_voxel_dim) == 3:
        z_zoom = round(float(curren_voxel_dim[0]) / float(desired_voxel_size[0]), 3)
        x_zoom = round(float(curren_voxel_dim[1]) / float(desired_voxel_size[1]), 3)
        y_zoom = round(float(curren_voxel_dim[2]) / float(desired_voxel_size[2]), 3)
    else:
        raise UserWarning("error in resize_image(): could not parse curren_voxel_dim or desired_voxel_dim")
        # Round for numerical consistency
    return [z_zoom, x_zoom, y_zoom]



IJ.run("Console")
logging.basicConfig()
logger = logging.getLogger('fiji')
logger.setLevel(logging.DEBUG)

# initialization. setup output directories
filename = str(data_file)
image_folder=os.path.join(os.path.dirname(filename), "temp")
if not os.path.isdir(image_folder):
    os.makedirs(image_folder)

# parse coordinate file paths, remove basefolder that sometimes is included by default
temp= list(listOfCoordinatePaths)
listOfCoordinatePaths=[]
for i, x in enumerate(temp):
    x=str(x)
    if not "Fiji.app" in str(x) and len(x)>0:
        listOfCoordinatePaths.append(x)

logger.debug("listOfCoordinatePaths: {}".format( listOfCoordinatePaths))

# get coordinate information
if prediction_type == "Detection" and len(listOfCoordinatePaths) > 0:
    CoordDict = coordinates_dialogue(listOfCoordinatePaths)
    # print(CoordDict)
else:
    coord_file_path, cell_diam, CoordDict = None, None, None
    CoordDict = {}

# get the opened image from FIJI
imp = IJ.getImage()
imp.setDisplayMode(IJ.COLOR)
title = imp.getTitle()
ip = imp.getProcessor()  # get the image processor
calibration = imp.getCalibration()
# get channel information
no_c = imp.getNChannels()
channel_ind, choices, info = import_dialog(no_c)

# pasre input
logger.debug("Channel Types: {}".format(choices))
# get indices of channels that were not selected to be skipped
keep_channels = [i + 1 for i, x in enumerate(choices) if x != "Skip Channel"]
logger.debug("Keep Channels: {}".format(keep_channels))
no_c = len(keep_channels)
names = []
user_metadata = []
# write input to info.txt file
info_txt_path=os.path.join(os.path.dirname(filename), "info_{}.txt".format(title.split(".")[0]))
# parse user input and write to info file
with open(info_txt_path, "w") as info_file:
    # construct list of names and bools for metadata file
    for i, (ind, label) in enumerate(zip(channel_ind, choices)):
        if label == "Skip Channel":
            continue
        elif label == "Marker":
            name="in_channel_{}".format(ind)
            names.append(name)
            user_metadata.append(info[i])
            info_file.write("{}:\t{}\n".format(name, info[i]))
        elif label == "Label":
            name="label_channel_{}".format(ind)
            names.append(name)
            user_metadata.append(info[i])
            info_file.write("{}:\t{}\n".format(name, info[i]))
        elif label == "Coordinate Mask":
            name="mask_channel_{}".format(ind)
            names.append(name)
            user_metadata.append(info[i])
            info_file.write("{}:\t{}\n".format(name, info[i]))
        else:
            print("Problem when parsing names")
    # save coordinate info
    for (key, value) in CoordDict.items():
        info_file.write("coord_channel_{}:\t{}\n".format(value["label"], value["Info"]))


calibration = imp.getCalibration()  # get the calibration, we need it for metadata
# resize image  to have desired pixel size
dimensions = list(imp.getDimensions())
y = dimensions[0]
x = dimensions[1]
c = dimensions[2]
z = dimensions[3]
#print(dimensions)

with open(str(model_file)) as f:
  model_info = json.load(f)
desired_voxel_size = model_info["voxel_size"]
#print("desired voxel size", desired_voxel_size)
# calculate zoom factor
zoom = get_zoom([calibration.pixelDepth, calibration.pixelHeight, calibration.pixelWidth], desired_voxel_size)
#print("zoom: ", zoom)

# abort if no channel was kept
if keep_channels == []:
    raise UserWarning("All channels were skipped!")
# not sure if this precaution is necessary
if anot_type =="2D" and zoom[0]!=1:
    raise UserWarning("image should not be resized in z dimension when anotation type is 2D. pass 2 dimensional Voxel size parameter, e.g. [1,1]")
assert len(keep_channels) == len(names)

print('calculate channelwise mean, std')



channel_statistics={}
for name,i in zip(names, keep_channels):
    for j in range(z):
        imp.setPosition(i,j,1)
        stats = imp.getStatistics()
        if str(name) not in channel_statistics.keys():
            channel_statistics[str(name)] = {"mean": [], "std": [], "empty": []}
        channel_statistics[name]["mean"].append(stats.mean)
        channel_statistics[name]["std"].append(stats.stdDev)
        channel_statistics[name]["empty"].append(stats.max == 0)

# calculate overall statistics, assuming identical sample size of all stacks
for k,i in channel_statistics.items():
    i["mean"]=sum(i["mean"])/len(i["mean"])
    i["std"]=(sum([s**2 for s in i["std"]])/len(i["std"]))**0.5


# WRITE METADATA
d = list(imp.getDimensions())
rescaled_image_dim =[int(d[3]*zoom[0]), d[2], int(d[1]*zoom[2]), int(d[0]*zoom[1])]
# dimensions[2] = image_c_no # set channel dimension to no of image channels)
metadata = {"original_file_name": title,
            "voxel_dimensions": [calibration.pixelDepth, calibration.pixelHeight, calibration.pixelWidth],
            # d is the image dimensions before rescaling, but we need rescaled image dimensions
            "image_dimensions": rescaled_image_dim,
            "original_image_dimensions": [d[3],d[2],d[1],d[0]],
            "channel_names": names,
            "user_metadata": user_metadata,
            "training_data": training_data == "Training",
            "annotations": anot_type,
            "Coodinate_Channels": CoordDict,
            "zoom" : zoom,
            "channel_statistics" : channel_statistics
            }
# save metadata file
metadata_path = os.path.join(os.path.dirname(filename), "metadata_{}.json".format(title.split(".")[0]))

print('Saving metadata to', metadata_path)
with open(metadata_path, 'w') as outfile:
    json.dump(metadata, outfile, indent=4)

# break up image into Substacks. makes it possible to convert larger than RAM images
# run python subbprocess to calculate indices of Z-Stack.
print("calculating patch indices".format(str(data_file)))
# wd should be Fiji.app path
wd = os.getcwd()
currpath = os.path.join(wd, "MiNTiF_Utils")
code_file = os.path.join(currpath, *['fiji_funcs','CalculatePatchIndices.py'])
# call subprocess. writes to json file
indices_path=os.path.join(image_folder,"indices.json")
cmd = """ "{}" "{}" "{}" "{}" """.format(code_file, str(model_file),str(metadata_path), str(indices_path))
FijiUtils.call_subprocess(cmd, currpath)
# open index file and read z_stack indices
with open(indices_path, 'r') as m:
    indices = json.load(m)
z_stack=indices['z_stack']
z_stack_original=indices['z_stack_original']


channel_statistics = {}
# loop trough stacks
for stack_n, (z_s, z_s_orig) in enumerate(zip(z_stack, z_stack_original)):
    # loop trough channels
    print("Stack {} of {}".format(stack_n, len(z_stack_original)))
    for name,channel_n in zip(names, keep_channels):
        # make an in memory copy of the channel & z substack (necessary for resize operation)
        imp2 =Duplicator().run(imp, channel_n,channel_n,z_s_orig[0], z_s_orig[1],1,1)
        # resize z-stack
        IJ.run(imp2, "Size...", "width={} height={} depth={} interpolation=Bicubic".format(y*zoom[1], x*zoom[2], (z_s_orig[1]-z_s_orig[0])*zoom[0]))
        imp3 = IJ.getImage()
        #can this be deleted?
        imp3.show()
        # save subsatck to temporary folder
        savename = 'image#{}#z{}_{}.tiff'.format (name, z_s[0], z_s[1])
        path_imp_channel = os.path.join(str(image_folder), str(savename))
        IJ.saveAsTiff(imp3, path_imp_channel)
        # close substack
        try:
            imp3.close()
            imp2.close()
        except:
            pass


# append to datafile
print("appending data to file {}".format(str(data_file)))
# get file paths
# wd should be Fiji.app path
wd = os.getcwd()
currpath = os.path.join(wd, "MiNTiF_Utils")
code_file = os.path.join(currpath, *['fiji_funcs','ImageToH5.py'])

# call subprocess
cmd = """ "{}" "{}" "{}" "{}" "{}" "{}" "{}" """.format(code_file, str(image_folder), str(data_file), str(model_file),
                                              str(compress), str(indices_path), str(metadata_path))
print(cmd)
FijiUtils.call_subprocess(cmd, currpath)
shutil.rmtree(image_folder)
print("done!")
