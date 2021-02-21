# (c) 2021, Luca Widmer  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
#@ String (visibility=MESSAGE, value="Export", required=false) msg2
#@ File (label='output folder', style='directory', persist=true, required=false) foldername
#@ String (label='Name')  name
#@ File (label='model dir', style='directory', persist=true, required=false, description="folder that contains exported tensorflow model (default name: 'my_model' ") model_dir
#@ File (label='model.json file', style='file', persist=true, required=false) json_file
#@ File (label='info.txt', style='file', persist=true, required=false) info_txt

from zipfile import ZipFile
import os

"""
This script directs the user to gather all the necessary files to generate  model zip directory that is intended as a
way to share a complete trained model in a  a repreducible way. 

Possible future expansions to this:
Include options to generate information files that are expected in certain sharing platforms such as Bioimag.io
"""

def absoluteFilePaths(directory):
    res = []
    for filenames in os.listdir(directory):
        res.append(os.path.abspath(os.path.join(directory, filenames)))
    return res

dst = os.path.join(str(foldername), str(name)) + ".zip"
assert not (os.path.exists(dst + ".zip")), "directory already exists!"
# get all files
files = absoluteFilePaths(str(model_dir)) + [str(json_file), str(info_txt)]
# create a ZipFile object
with ZipFile(dst, 'w') as zipObj:
    # Iterate over all the files in directory
    for src in files:
        print(src)
        # continue
        if not os.path.isdir(src):
            zipObj.write(src, os.path.basename(src))
