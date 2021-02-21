# (c) 2020-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
import os, h5py, json, argparse

parser = argparse.ArgumentParser("Arguments for file conversion")

parser.add_argument('--data_h5', '-d', required=True, type=str, help="Filename (.h5) to convert")
parser.add_argument('--traindir', '-t', required=True, type=str, help="Directory with training files")
parser.add_argument('--evaldir', '-e', required=True, type=str, help="Directory with evaluation files")
parser.add_argument(
    '--modelfile', '-m', required=True, type=str, help="Model settings as json file where indices are written")

args = parser.parse_args()

fname=args.data_h5
train_files = [os.path.splitext(x)[0] for x in os.listdir(args.traindir)]
eval_files = [os.path.splitext(x)[0] for x in os.listdir(args.evaldir)]
jfile = args.modelfile

ltrain = []
leval = []
with h5py.File(fname, 'r') as fh:
    for s in fh:
        sind = int(s.replace("Sample_", ""))
        if fh[s].attrs['name'] in train_files:
            ltrain += [sind]
        elif fh[s].attrs['name'] in eval_files:
            leval += [sind]
        else:
            ltrain += [sind]
            print("{} was added to training although it was not listed in the directory".format(fh[s].attrs['name']))
            # raise Exception("Not found")

with open(jfile, 'r') as f:
    data = json.load(f)
data["indices"] = [ltrain, leval, []]
with open(jfile, 'w') as f:
    json.dump(data, f, indent=1)
