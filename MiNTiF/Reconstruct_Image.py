# (c) 2021, Luca Widmer  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel

import os
import site
import logging
from ij import IJ

wd = os.getcwd()
wd = os.path.join(wd, "MiNTiF_Utils")
site.addsitedir(wd)
from fiji_funcs import FijiUtils

"""
This Script implements a GUI to reconstruct all images and cordinates in a MiNTiF dataset.

"""
#@ File (label="MiNTiF Data File", style="both", persist=true) data_file
#@ Boolean (label="Reconstruct Image", description="should original image and predicted labels be reconstructed") recon_image
#@ Boolean (label="Reconstruct Coordinates", description="should predicted coordinates be reconstructed") recon_coords
#@ Boolean (label="Reconstruct Density Maps", description="should original and predicted Density Maps be reconstructed") recon_DM
# Boolean (label="open in ImageJ (not implemented)") open_in_imagej

IJ.run("Console")
logging.basicConfig()
logger = logging.getLogger('fiji')
logger.setLevel(logging.DEBUG)

if recon_DM  and not recon_image:
    raise UserWarning("you can only reconstruct density maps if image is also reconstructed")
if not recon_image and not recon_coords:
    raise UserWarning("you must reconstruct at least the Image or the Coordinates")

wd = (os.getcwd())
currpath = os.path.join(wd, "MiNTiF_Utils")
code_file = os.path.join(currpath, *['fiji_funcs','H5ToImage.py'])

print ("Starting Reconstruction of:\n{}".format(str(data_file)))
# run reconstruction script as subprocess
cmd = """ "{}" "{}" "{}" "{}" "{}" """.format(code_file, str(data_file), str(recon_image), str(recon_coords), str(recon_DM))
print(cmd)
FijiUtils.call_subprocess(cmd, currpath)