# (c) 2018-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
import argparse
import logging
import os

from  config import settings
from  data_read.datasets.dtfm import DTFM
from  utils import common_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser('Define parameters for data writing')
parser.add_argument('--model_settings', '-s', required=True, type=str, help='Model settings file (json)')
parser.add_argument('--exp_name', default=None, required=False, type=str, help='Experiment name')
parser.add_argument('--filenames', '-f', nargs='+', required=False, default=None,
                    help="Filenames to process. Enter filenames or filenames_dir only")
parser.add_argument('--filenames_dir', required=False, default=None, type=str,
                    help="Directory with files to process. Enter filenames or filenames_dir only")
parser.add_argument('--outfile', '-o', default=None, required=False, type=str, help="Output file")
parser.add_argument('--data_name', '-d', default=None, required=False, type=str, help="Name of the dataset")
parser.add_argument('--max_samples', default=None, required=False, type=int,
                    help='Maximum  number of samples to write')
parser.add_argument('--max_patches', default=None, required=False, type=int,
                    help='Maximum number of patches to write')
parser.add_argument('--training', '-t', default=False, required=False, type=common_utils.str2bool, help='Train model?')
args = parser.parse_args()

# Filenames
if args.filenames:
    if args.filenames_dir:
        logger.error("filenames and filenames_dir cannot be used together")
    filenames = common_utils.input_files_format(args.filenames, fext=['.ims'], do_recursive=False, channels=None)
elif args.filenames_dir:
    filenames = [os.path.join(path, name) for path, subdirs, files in os.walk(args.filenames_dir)
                 for name in files if
                 os.path.splitext(name)[1] == ".ims"]
else:
    logger.error("either filenames or filenames_dir should be used as input")

fext = os.path.splitext(filenames[0])[1]
msettings = settings.get_settings(args.model_settings)
# Debugging
for file in filenames:
    if not (os.path.splitext(file)[1] == fext):
        # We can remove this condition in the future
        raise Exception("all files should have the same extension")

logger.info("Filenames: {}".format(filenames))
logger.info("Output file: {}".format(args.outfile))
logger.debug("File ext: {}".format(fext))

if fext == ".ims":
    # Create h5 with all the files specified
    data_name = args.data_name  # or 'LabeledData'
    D = DTFM(filename=args.outfile, msettings=msettings, data_name=data_name)
    D.ims_to_hdf5(file_ims=filenames,
                  do_labels=args.training,
                  max_samples=args.max_samples,
                  max_patches=args.max_patches)
elif fext == ".h5":
    # Convert all the samples in the h5 file to ims
    for f5 in filenames:
        data_name = args.data_name  # or 'PredictData'
        exp_name = args.exp_name or 'pred'
        # if args.model_settings:
        #     exp_name = os.path.splitext(args.model_settings)[0]
        # else:
        #     exp_name = 'pred'
        D = DTFM(filename=f5,
                 msettings=msettings,
                 data_name=data_name)
        D.predictions_to_ims(exp_name=exp_name)
