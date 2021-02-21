# (c) 2021, Luca Widmer  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
import ast
import json
import site
import os
import logging
from ij import IJ
# this is necessary to load owned modules in Jython runtime
wd = os.getcwd()
wd = os.path.join(wd, "MiNTiF_Utils")
site.addsitedir(wd)
'''
This Script implements a GUI for FIJI to define a Model compatible with the MINTIF plugin.
The interface allows the user to enter all relevant model parameters, which will then be saved to a model.json file.
'''

# DO NOT CHANGE, These are The GUI Parameters

#@ File  (label="Output Path", required=true, style = "directory", description = "Choose a Directory where model will be saved.") model_dir
#@ String (visibility=MESSAGE, value="Training Parameters", required=false) msg
#@ Integer (label='Batch Size', value=8, persist=false, description = "Number of samples to use per training pass") batch_size
#@ Integer (label="Epochs", value=100,  persist=false, description = "Number of training iterations" ) epoch
#@ Integer (label='Cross Validation Steps', value=5,  persist=false, description="Number of Cross Validation Steps to perform. Repeat network training for every step.") cross_validation
#@ Integer (label='Number of Levels', value=5, persist=false, description = "Number of connvolutional layers. Influences the output patch size") nlevels
#@ String (label="Padding ", choices={"same", "valid"},  persist=false, description = "Padding strategy: detailed description in documentation") padding
#@ Float (label='Dropout', value=0, style = "slider", min=0, max=1, stepSize=0.05,  persist=false, description = "Chance of removing any node from network during training, may reduce overfitting") dropout
#@ Boolean (label='Batch Normalization', value=true,  persist=false, description = "Reduced the impact of outlying large weights in training ") batch_norm
#@ String (label="Patch Size", value="[572, 572]",  persist=false, description = "Size of a single patch in pixels in (Z),X,Y Dimension") patch_size
#@ String (label="Voxel Size", value=[1, 1],  persist=false,description = "Desired Voxel Size in um; (Z),X,Y. Used to rescale images in dataset to identical resolution. only give X,Y dimensions if you work with 2D patches") voxel_size
#@ Boolean (label='Normalization', value=true,  persist=false, description = "Normalize data to mean=0, std=1") normalize_data
#@ String (visibility=MESSAGE, value="Data Augmentation Parameters  (in Development) ", required=false) msg2
#@ Boolean (label='Flipping', value=false, persist=false) flipping
#@ Boolean (label='Rotation', value=false, persist=false) rotation
#@ Boolean (label='Zoom', value=false, persist=false) zoom
#@ Boolean (label='Noise', value=false, persist=false) noise
#@ String (visibility=MESSAGE, value="<html> <b>Model Parameters<hr></b>", required=false) msg3
#@ String (label='Model Type', description="Name of model to be used. needs to be defined in model zoo files. Examples described in Documnetation; UNet2D, UNet3D, MarkerSampling_MarkerExcite") cnn_name
#@ String (label='Channel IDs', persist=true, description="Numerical labels of channels to be used as output. Format: e.g. [0, 1]") ind_of_channels
#@ String (label='Label IDs', persist=true, description="Numerical labels of channels to be used as output. Format: e.g. 2,3]") ind_of_labels
#@ String (label='Train/Test Sample Indicies', persist=true, description = "Use this for reproducible sample use in different cross-validation folds. give 3D array, e.g. [[[0],[1],[2]]]")  indices
#@ String (visibility=MESSAGE, value="Parameters for 3D Detection, ignore if unused", required=false) msg1
#@ String (label="Model Type ", choices={"Segmentation","Detection","Other"}, description="Usecase of this Model") dataset_type
#@ Float (label='Cell Radius',  persist=true, description = "Radius of a cell in micro meter in this dataset") cell_radius


# def dumper(obj):
# 	"""
# 	# Helper functions to cast parameters
# 	@param obj:
# 	@return:
# 	"""
# 	try:
# 		return obj.toJSON()
# 	except:
# 		return obj.__dict__

def make_string_int_array(string):
	string = string.strip()[1:-1]
	return list(map(int, string.split(',')))

def make_string_float_array(string):
	string = string.strip()[1:-1]
	return list(map(float, string.split(',')))


logging.basicConfig()
logger = logging.getLogger('fiji')
logger.setLevel(logging.DEBUG)
IJ.run("Console")
# Parse marker and class channel names/indices
label_channels =[]
channel_channels = []
chanel_inds = make_string_int_array(ind_of_channels)
label_inds = make_string_int_array(ind_of_labels)
for i in chanel_inds:
	channel_channels.append("in_channel_{}".format(i))
for i in label_inds:
	if dataset_type == "Detection":
		label_channels.append("label_dm_{}".format(i))
	else:
		label_channels.append("label_channel_{}".format(i))
# Parameters for model json file
vs=make_string_float_array(voxel_size)
parameters = {
	# 'model_transfer': path_of_model,
	'batch_size': batch_size,
	'nlevels':nlevels,
	'padding': padding,
	'epochs': epoch,
	'dropout': dropout,
	'batchnorm': batch_norm,
	'patch_size': make_string_int_array(patch_size),
	'voxel_size': vs,
	'augmentation': {'flipping': flipping, 'rotation': rotation, 'zoom': zoom, 'noise': noise},
	'normalize_data': normalize_data,
	'cnn_name': cnn_name,
	'markers': channel_channels,
	'classes': label_channels,
	'cross_validation': cross_validation,
	"cells_radius": cell_radius,
	"cells_radius_pixel":float(cell_radius) / (sum(vs) / len(vs)),
	'indices':  ast.literal_eval(indices) if indices != "" else None
}

if dataset_type == "Detection":
	parameters["dataset_type"]='detection'
	# add additional padding the size of cell diameter in pixels to avoid detection artefacts on image border
	logger.info("added additional padding the size of cell diameter in pixels to avoid detection artefacts on image border")
	for i in range(len(parameters["patch_size"])):
		parameters["patch_size"][i] += int(round(2*parameters['cells_radius_pixel']))
elif dataset_type == "Segmentation":
	parameters["dataset_type"]='semantic_seg'
elif dataset_type == "Other":
	parameters["dataset_type"]='other'

# Write parameters into json file
modelpath=os.path.join(str(model_dir), 'model.json')
logger.info("writing parameters in {}".format(modelpath))
with open(modelpath, 'w') as handle:
	json.dump(parameters, handle, indent=2)

print("done!")

