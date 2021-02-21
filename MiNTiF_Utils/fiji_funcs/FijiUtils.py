# (c) 2020-2021, Luca Widmer  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
import os
import subprocess
"""
This function is isolated as it is used by jython scripts that dont have access to many CPython libraries
"""
def call_subprocess(cmd, wd=None):
    """
    This function runs a pythonscript as a subprocess in cnnframework conda environment
	cmd: path to script followed by arguments that will be passed to script
    @param cmd: comand string, needs to include path to python script as first argument
    @param wd: working directory
    @return: return code of suprocess
    """
    # todo get output to stream to fiji in realtime
    # get working dir path for fiji scripts
    wd = wd or os.path.join(os.getcwd(),"MiNTiF_Utils")
    # call subprocess in cnnframework conda env
    pwrite = subprocess.Popen(r"""conda run -n MiNTiF python {} """.format(cmd),
                              shell=True, cwd=wd, bufsize=1, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    # print output
    for stdout_line in iter(pwrite.stdout.readline, b''):
        print(stdout_line)
    for stdout_line in iter(pwrite.stderr.readline, b''):
        print(stdout_line)
    pwrite.stdout.close()
    return_code = pwrite.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
    return return_code

