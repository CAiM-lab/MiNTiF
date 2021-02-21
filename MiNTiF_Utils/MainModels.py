# (c) 2019-2021,   Alvaro Gomariz, Luca Widmer  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
import tensorflow as tf

tf.random.set_seed(12)

import os
import time
import json
import argparse
import logging
import site
site.addsitedir(os.getcwd())

from config import settings
from config import system
from cnn import model_zoo, cnn_func
from utils import common_utils
from data_read.datasets.dtfm import DTFMdataset
from fiji_funcs import CoordinateUtils

if system.eager_mode:
    tf.config.experimental_run_functions_eagerly(True)

time_start = time.time()
# Set logging
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Basic arguments
parser = argparse.ArgumentParser(description='Define parameters for experiment')
parser.add_argument('--data_file', '-d', default=None, required=False, type=str, help='Data file (hdf5)')
parser.add_argument('--model_settings', '-s', default=None, required=False, type=str,
                    help='Model settings file (json). By default, a model.json file will be searched in model_dir')
parser.add_argument('--model_dir', '-n', required=True, type=str, help='Model directory')
parser.add_argument('--netmode', '-t', default=False, required=True, type=str,
                    help='Train and test (train), predict (pred), test (test) ')

# Developer arguments
parser.add_argument('--debug', required=False, type=bool, default=False, help='Use debugging mode')
parser.add_argument('--exp_name', default=None, required=False, type=str, help='Experiment name')
parser.add_argument('--ncv', required=False, type=int, default=-1, help='Define cross-validation step: None for all')
parser.add_argument('--trainset_only', required=False, default=False, type=common_utils.str2bool,
                    help='Use training set only')
parser.add_argument('--chdel', required=False, default=False, type=common_utils.str2bool,
                    help='Apply channel deletion at test time')
parser.add_argument('--sample_markers', default=None, required=False, type=str,
                    help="File with a dictionary that assign a subset of markers to samples")
parser.add_argument('--pretrained_model_files', default=None, required=False, type=str,
                    help="path to pretrained model file (tensorflow pb). used instead of files in model_dir")
parser.add_argument('--transfer_learning', default=False, required=False, type=common_utils.str2bool,
                    help="should transfer learning be applied to this model")

args = parser.parse_args()

# Load settings
if args.sample_markers and args.netmode not in ("predict", "pred"):
    raise Exception("sample_markers has only been implemented for prediction")
if args.model_settings is None:
    dir_msettings = os.path.join(args.model_dir, 'model.json')
    if not os.path.isfile(dir_msettings):
        error_msg = "Model not found in {}. \nIf the model settings file is not specified, " \
                    "it should be located under model_dir as model.json".format(dir_msettings)
        raise Exception(error_msg)
else:
    dir_msettings = args.model_settings
msettings = settings.get_settings(dir_msettings)
logger.info("Experiment directory: {}".format(args.model_dir))

# Get model
if args.pretrained_model_files is not None:
    pretrained_model_files=args.pretrained_model_files
    assert os.path.exists(pretrained_model_files)
    logger.info("Model: load from tensorflow files at {}".format(args.pretrained_model_files ))
elif msettings['cnn_name'] in model_zoo.__dict__.keys():
    model_name = getattr(model_zoo, msettings['cnn_name'])
    logger.info("Model: {}".format(msettings['cnn_name']))
else:
    raise Exception("Model {} not found".format(msettings['cnn_name']))

# Check if markers are defined individually for each sample in a file
if args.sample_markers:
    with open(args.sample_markers, 'r') as f:
        sample_markers = json.load(f)
else:
    sample_markers = None
# logger.info("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Training/evaluation
if args.netmode in ('train', 'test'):
    # Load dataset
    D = DTFMdataset(
        filename=args.data_file,
        msettings=msettings,
        data_name=None,  # 'LabeledData',
        l_channels_in=msettings['channels'],
        l_channels_out=msettings['labels'],
    )
    # Create tf datasets
    if args.trainset_only and ("indices" not in msettings) or (msettings["indices"] is None):
        data_split = (0.8, 0.2, 0.)
        ldatasets, linds = D.split_datasets(data_split)
        ldatasets = [ldatasets]
        linds = [linds]
        ldsave = [args.model_dir]
    else:
        cross_validation = 1 if 'cross_validation' not in msettings else msettings['cross_validation']
        ldatasets, linds = D.crossval_datasets(kfold=cross_validation)
        if len(ldatasets) > 1:
            ldsave = [os.path.join(args.model_dir, str(x)) for x in range(len(ldatasets))]
        else:
            ldsave = [args.model_dir]

    # Iterate cross-validation samples
    for ncv, (ds_train, ds_val, ds_test) in enumerate(ldatasets):
        if args.ncv > -1 and not (ncv == args.ncv):
            continue
        # Train
        if args.netmode in ('train',):
            # If metrics exists, skip. dont skip if debug mode ore transfer learning enabled
            if os.path.isdir(ldsave[ncv]) and os.path.isfile(
                    os.path.join(ldsave[ncv], 'metrics.csv')) and not args.debug and not args.transfer_learning:
                logger.warning("The model already exists, skipping job")
                continue
            # If directory exists, but no metrics, delete and train again
            # elif os.path.isdir(ldsave[ncv]) and not args.trainset_only:
            #     shutil.rmtree(ldsave[ncv])
            if len(linds[ncv][0]) == 0:
                logger.warning("The training set is empty, skipping")
                if not os.path.exists(ldsave[ncv]):
                    os.makedirs(ldsave[ncv])
                with open(os.path.join(ldsave[ncv], 'notrain.txt'), 'w') as f:
                    f.write("No training data")
                continue

        model_dir = ldsave[ncv]
        logger.info("\n\nStarting cross-validation {}".format(ncv))
        logger.debug("------ Indices ------  \nTrain: {}\nVal: {}\nTest: {}\n\n".format(
            tuple(linds[ncv][0]), tuple(linds[ncv][1]), tuple(linds[ncv][2])))
        tf.keras.backend.clear_session()
        mtime = time.time()

        if args.debug:
            # We can reduce the dataset for debugging purposes
            logger.warning("Debugging: dataset is reduced")
            ds_train = ds_train.take(1)
            ds_val = ds_val.take(1)
            ds_test = ds_test.take(4)
            msettings['epochs'] = 2
            tf.config.experimental_run_functions_eagerly(True)

        if args.pretrained_model_files is None:
            # Create new  model from model dir
            net = model_name(msettings=msettings, class_weights=D.set_classweights(linds[ncv][0]))
            # smodel = cnn_func.ModelStructure(net, args.model_dir)


        else:
            # load model from pretrained model files
            logging.info("loading model from: {}".format(args.pretrained_model_files))
            # Create model with the defined settings
            net = cnn_func.model_zoo.CustomModel(msettings,class_weights=D.set_classweights(linds[ncv][0]))
            # smodel = cnn_func.ModelStructure(net, args.model_dir, pretrained_model=args.pretrained_model_files)
            # net = model_name(msettings=msettings, class_weights=D.set_classweights(linds[ncv][0]))
        do_transfer = args.transfer_learning
        smodel = cnn_func.ModelStructure(net, model_dir, linds[ncv], do_transfer=do_transfer, msettings=msettings,
                                         pretrained_model=args.pretrained_model_files)
        logger.info("{}".format(smodel.model.summary()))
        if args.netmode == 'train':
            smodel.train((ds_train, ds_val, ds_test), msettings['epochs'], level='low')

        smodel.model.save(os.path.join(smodel.model_dir, 'my_model'), include_optimizer=True)

        # roundabout way to test detection on whole sample (instead of patch-wise)
        if msettings["dataset_type"] == "detection":
            #first predict dms for whole sample
            exp_name = 'testing'
            out_cname = [exp_name + '_' + x for x in msettings['labels']]
            # get data only on test samples
            test_set = D.data_generator(data_read=linds[ncv][2], isout={"metadata": True}, sample_markers=sample_markers)
            # Predict on outgroup testset
            smodel.predict(test_set, D, msettings['batch_size'], exp_name=exp_name)
            # calculate & write metrics and delete test prediction afterwards
            CoordinateUtils.hungarian_test_on_samples(data_file=args.data_file, sample_ids=linds[ncv][2],
                                                      output_folder=os.path.join(args.model_dir, str(ncv)))
        else:
            smodel.test_net(ds_test, do_chdel=args.chdel)

# reload and save the model as tf safemodel files, useful if whole model (with weights etc.) should be exported
elif args.netmode == 'save':
    assert os.path.exists(args.model_dir)
    model_dir=str(args.model_dir)
    net = model_name(msettings=msettings)
    smodel = cnn_func.ModelStructure(net, model_dir, msettings=msettings)
    # load best model epoch trained so far
    smodel.load_net(mode='best')
    model_path = os.path.join(smodel.model_dir, 'my_model')
    logger.info("save full model to {}".format(model_path))
    logger.info("{}".format(smodel.model.summary()))
    # save whole model, can be reloaded with tf.keras.models.load_model(pretrained_model)
    smodel.model.save(model_path, include_optimizer=True)

    # Take metrics in model directory from different cross-validation steps to study them together
    if args.ncv is None:
        common_utils.aggregate_metrics(args.model_dir)

# Prediction
elif args.netmode in ('predict', 'pred'):

    if args.pretrained_model_files is None:
        # Create new  model from model dir
        net = model_name(msettings=msettings)
        smodel = cnn_func.ModelStructure(net, args.model_dir)


    else:
        # load model from pretrained model files
        logging.info("loading model from: {}".format(args.pretrained_model_files))

        net=cnn_func.model_zoo.CustomModel(msettings)
        smodel = cnn_func.ModelStructure(net, args.model_dir, pretrained_model=args.pretrained_model_files)

    logging.info("Predicting for file: {}".format(args.data_file))
    # Create tf dataset
    D = DTFMdataset(filename=args.data_file,
                    msettings=msettings,
                    data_name=None,  # "PredictData",
                    l_channels_in=msettings['channels'],
                    l_channels_out=msettings['labels'])
    exp_name = args.exp_name or 'pred'
    out_cname = [exp_name + '_' + x for x in msettings['labels']]
    # D.clean_channels(out_cname)  # For debug purposes
    ds_pred = D.data_generator(data_read=None, isout={"metadata": True}, sample_markers=sample_markers)
    time0 = time.time()
    # Prediction
    smodel.predict(ds_pred, D, msettings['batch_size'], exp_name=exp_name)
    print("Prediction took a total of {} seconds".format(time.time() - time0))

else:
    raise Exception(
        'Model mode "{}" unknown. Use training (train), predict (pred), test (test) or save (save)'.format(args.netmode))

logger.info("Job completed in {} seconds".format(time.time() - time_start))
