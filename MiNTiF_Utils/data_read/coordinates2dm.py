# (c) 2019-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel

import sys
sys.path.append(".")

import h5py, argparse
from data_read.datasets.dtfm import DTFM
from config import settings

parser = argparse.ArgumentParser("Arguments for file conversion")

parser.add_argument('--filename', '-f', required=True, type=str, help="filename (.h5) to convert")
parser.add_argument('--settings', '-s', required=True, type=str, help="model settings (.json) file")

args = parser.parse_args()

msettings = settings.get_settings(args.settings)

df = DTFM(args.filename, msettings)
df.coordinates2dm()
