# (c) 2021, Luca Widmer  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel

import site
import os
import logging
# this is necessary to load owned modules in Jython runtime

wd = os.path.join(os.getcwd(), *["MiNTiF_Utils", "fiji_funcs"])
site.addsitedir(wd)
import FijiUtils

"""
This Script implements a GUI to deploy a MINTIF plugin Model, either for training or prediction.
"""
# DO NOT CHANGE, These are The GUI Parameters
#@ String (visibility=MESSAGE, value="<html> <b>Model & Data Files</b>", required=false) msg4
#@ File (label="Model File", style="file", persist=true, description="path to model.json file") model_file
#@ File (label="Data File", style="file", persist=true, description="path to data file containing training or prediction datasets") data_file
#@ File (label="Model Folder", style="directory", persist=true, description="folder that contains trained model files / where those files should be saved") model_folder
#@ Boolean (label="Apply Transfer Learning", description="Use selected model as starting point for training. Used to continue training on same data or finetune model with new data") transfer_learning

#@ String (label="Task:", choices={"Train", "Predict"}, style="radioButtonHorizontal", persist=true) train_predict
# Boolean (label="Debug?", description="only run on a small subsample of the data. only usefull to test the pipeline") debug

logging.basicConfig()
logger = logging.getLogger('fiji')
logger.setLevel(logging.DEBUG)

wd = os.getcwd()
currpath = os.path.join(wd, "MiNTiF_Utils")
model_folder=str(model_folder)
# check if model is a pretrained saved tensorflow model by looking for model files in model_folder
if all([x in os.listdir(model_folder) for x in ["saved_model.pb", "assets", "variables"]]):
    pretrained_model_files=model_folder
    logger.debug("found pretrained tensorflow model, will attempt to load model in deployment")
else:
    logger.debug("found no pretrained tensorflow model in directory")
    pretrained_model_files=None

# print advise on how to open tensorboard
if train_predict == "Train":
    print("""To review Training Progress, open an anaconda prompt and type: 
          \n conda activate MINTIF & tensorboard --logdir "{}" \n
          Next, open the link that should appear""".format(model_folder))

# call subprocess to train/predict using MainModels.py
code_file = os.path.join(currpath , r'MainModels.py')
# generate command string
cmd = """ "{}" -n "{}" -s "{}" -d "{}" -t {} --transfer_learning {}  """.format(
    code_file,  model_folder,str(model_file) ,str(data_file), train_predict.lower(), str(transfer_learning).lower())
if pretrained_model_files is not None:
    cmd=cmd + """ --pretrained_model_files "{}" """.format(pretrained_model_files)

logger.debug("Calling {} with command:\n".format(str(code_file)))
logger.debug(cmd)
ret_code = FijiUtils.call_subprocess(cmd, currpath)

if ret_code == 0:
    logger.info("The model was completed")
