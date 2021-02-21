# (c) 2020-2021, Luca Widmer  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
"""
This will be called from Create_Dataset to calculate patch indices and return them to fiji for cropping
"""
import os
import site
import sys
import numpy as np
site.addsitedir(os.getcwd())

from utils.cnn_utils import get_overlap
from config import settings
import json



def get_patch_indices(img_shape, patchsize, overlap_patch, padding_type):
    """
    calculates patch indices from imageshape and patch dimensions

    @param img_shape: dimensions of whole image in pixel
    @param patchsize: dimensions of patch in pixel
    @param overlap_patch: overlap (usually between in_patches) in pixel
    @return: list of lists, each list entry is one patch index list and denotes the bounds of one patch
    (e.g. list_inds[0] = [0,1,0,100,0,100] means the first patch contains the values of image[0:1, 0:100, 0:100]
    """
    padsize_patch = (np.round(overlap_patch / 2)).astype(np.uint16)

    # list of lists. first list contains  indices in x dimension for input patches. npatches x 2 matrix. always 3 dimensions

    input_indices = []
    for dim_ind, len_dim in enumerate(img_shape):

        # maximum is size of image + padding (0,0) is left top pixel
        # negative index means 0 padding on to the left and top of 1st pixel
        # if padding=same, allow patch to overlap image
        if padding_type =="valid":
            minub = -1 * padsize_patch[dim_ind]
            maxub = len_dim + padsize_patch[dim_ind]
        elif padding_type == "same":
            minub = 0
            maxub = len_dim
        # lower boundry initialized as the minimum lower boundry
        lb = minub
        # upper boundry as lower boundry + patchsize
        ub = lb + patchsize[dim_ind]

        input_indices_1d = [[lb, ub]]
        # repeat until we cross the boundry of the upper patch edge or land exactly on the upper edge of the image
        while ub < maxub:# and ub != len_dim:
            # define next interval
            lb = ub - overlap_patch[dim_ind]
            ub = lb + patchsize[dim_ind]
            # when we reach the end of the image in this dimension we will overshoot ub
            # then reset upper boundry to max value and push lower boundry back to obtain a full sized batch overlapping with the previous
            if ub > maxub:
                ub = maxub
                lb = maxub - patchsize[dim_ind]
            input_indices_1d.append([lb, ub])
        input_indices.append(input_indices_1d)

    list_inds = []
    out_ind=[]
    z_stack=[]
    z=overlap_patch[0]//2
    x=overlap_patch[1]//2
    y=overlap_patch[2]//2

    # create all combinations of index slices for input and output
    for lb3, ub3 in input_indices[0]:
        for lb1, ub1 in input_indices[2]:
            for lb2, ub2 in input_indices[1]:
                list_inds.append([lb3, ub3, lb2, ub2, lb1, ub1])
                # different padding for out slices in 'valid' case
                if padding_type == "valid":
                    out_ind.append([lb3+z, ub3-z, lb2+x, ub2-x, lb1+y, ub1-y])
        z_stack.append([lb3, ub3]) #lb3+z, ub3-z])

    if padding_type != "valid":
        out_ind = list_inds
    return list_inds, out_ind, z_stack


# print(sys.argv)
model_file = sys.argv[1]
metadata_file = sys.argv[2]
indices_path = sys.argv[3]

# load model params and fill in missing values with defaults
msettings = settings.get_settings(model_file)

# load metadata
with open(metadata_file, 'r', encoding='utf8') as m:
    metadata = json.load(m)

# parse msettings
mp_size = msettings[
    'scale_factor']  # a tuple with the downsampling factor applied when changing the level in the network for each dimension
nlevels = msettings['nlevels']  # number of downsampling steps in the network
ksize = msettings['kernel_size']  # kernel size applied in the convolutional layers for each dimension
nconv = [2] * len(
    ksize)  # parameters['nconv'] # number of convolutional layers at each layer of the network for each dimension
patch_size = np.array(msettings['patch_size'])
desired_voxel_size = msettings['voxel_size']
dataset_type=msettings["dataset_type"]
zoom=metadata["zoom"]
if len(desired_voxel_size)==2:
    desired_voxel_size = np.append(1, desired_voxel_size)
    # print("converted desired voxel size to 3D: {}".format(desired_voxel_size))

# CALCULATE INDICES
# first get border size
# calculate overlap for patch dimensions, code adapted from previous project get:overlap()
if msettings["padding"] == "valid":
    overlap_patch = get_overlap(patch_size, mp_size, nlevels, ksize, nconv)
else:
    overlap_patch = patch_size * 0
assert len(patch_size) == len(overlap_patch)
# if patch shape is 2D or z dim is 0 make it 3D with z=1
try:
    if len(patch_size) == 2:
        patch_size = np.append(1, patch_size)
        overlap_patch = np.append(0, overlap_patch)
        # print("set z dimension of patch size / overlap to 1 / 0")
    elif patch_size[0] == 0:
        patch_size[0] = 1
        # print("set z dimension of patch size from 0 to 1")
except Exception as e:
    print("problem in parsing slice shape\n{}".format(e))

img_dim=metadata['image_dimensions']
orig_img_dim=metadata['original_image_dimensions']
# drop channel number
img_dim.pop(1)
"""
If dataset type is detection: create additional padding for detection.
this implementation simply shrinks the effective size of the patch by removing the cell diam on each side before the calculation.
and adding the padding pack on to the indices after. this way, the effective output patches should be tiled nicely.
This extra padding is added to user defined patch size before in Define_Model. 
A cleaner way to implement this would have been to increase padding size in dataset construction. 
"""

if msettings["dataset_type"] == "detection" and msettings["cells_radius"] is not None:
    cr=int(round(msettings["cells_radius_pixel"]))
    # remove cell diameter from effective patch size
    effective_patch_size = patch_size - 2*cr
    assert not any(effective_patch_size<0)
    # calculate effective patch indicies
    in_patch_ind, out_patch_ind, z_stack = get_patch_indices(img_dim, effective_patch_size, overlap_patch, msettings["padding"])
    #add padding back to indicies
    in_patch_ind=np.add(in_patch_ind, [-cr, +cr, -cr, +cr, -cr, +cr])
    out_patch_ind=np.add(out_patch_ind, [-cr, +cr, -cr, +cr, -cr, +cr])
    z_stack=np.add(z_stack, [-cr, +cr])#, -cr, +cr, ])

else:
    in_patch_ind, out_patch_ind, z_stack = get_patch_indices(img_dim, patch_size, overlap_patch, msettings["padding"])

# get zstacks in original image dimension (for initial zstack slicing in FIJI)
z_stack_original=[]
for i,z_scaled in enumerate(z_stack):
    z=1*z_scaled
    z=np.round(np.divide(z,zoom[0])).astype(int)
    # remove top, bottom padding
    # cut padding on image edge
    z=[max(0,z[0]),min(orig_img_dim[0], z[1])]
    z_stack_original.append(z)

# sanity check
assert not any(patch_size < 0)

indices={
    "in_patch_ind":in_patch_ind,
    "out_patch_ind" : out_patch_ind,
    "z_stack":z_stack,
    "z_stack_original": z_stack_original,
    "overlap_patch":overlap_patch
}

# save indes files, custom encoder class to ensure correct formatting
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

# print("save indices to {}".format(indices_path))
with open(indices_path, 'w') as outfile:
    json.dump(indices, outfile, cls=NpEncoder, indent=2)

