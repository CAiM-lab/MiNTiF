# (c) 2019-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel

import os, inspect, socket, logging, matplotlib
import tensorflow as tf
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

pc_name = socket.gethostname()
logging.info('Running on {:s}'.format(pc_name))
save_dir = os.path.join(os.path.realpath(__file__), "../..", "..", "output")

gsheets_keys = os.path.join(parentdir, 'gsheets_keys.json')
eager_mode = False
verbose = 2
# matplotlib_be = 'TkAgg'
matplotlib_be = 'Qt5Agg'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42