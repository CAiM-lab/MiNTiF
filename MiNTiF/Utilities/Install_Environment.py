# (c) 2021, Luca Widmer  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel

import os
import subprocess

# Boolean (label='Overwrite Previous Environment?', value=false, persist=false, description="Should previous existing environmet be overwritten? Only affects MINTIF.") overwrite


def install_MINTIF_env(wd, overwrite=False):
    """
    automatically installs MINTIF conda environment in predictable in  subfolder.
    calls a subprocess conda command and to install environment from .yml file
    @param wd: directory of environment.yml file
    @param overwrite: depreciated, selects if force flag is set in conda command. flag is now always set.
    @return: return code of subprocess
    """
    try:
        cmd = r"""conda env create -f "{}" --force """.format(os.path.join(wd, "environment.yml"))
        # execute command
        pwrite = subprocess.Popen(cmd, shell=True, cwd=wd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        # print output
        for stdout_line in iter(pwrite.stdout.readline, b''):
            print(stdout_line)
        for stdout_line in iter(pwrite.stderr.readline, b''):
            print(stdout_line)
        pwrite.stdout.close()
    except Exception as e:
        print("Was unable to install environment:\n{}\n you might need to select Overwrite == True if you've previously ran this ".format(e))
    else:
        print("Finished Environment Install.")
    return pwrite.wait()

import logging

from ij import IJ

IJ.run("Console")
logging.basicConfig()
logger = logging.getLogger('fiji')
logger.setLevel(logging.DEBUG)

p = os.path.abspath(os.path.join(os.curdir, "MiNTiF_Utils"))
print("Starting Environment Install:\n")
logger.debug(p)
install_MINTIF_env(p)
