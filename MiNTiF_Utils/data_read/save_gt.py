# (c) 2018-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
import os, sys
sys.path.append('..')
from data_read.imarisfiles import ImarisFiles
import h5py
import numpy as np
import logging
import argparse

logging.getLogger(__name__)
logging.basicConfig()


# Define argument parser
parser = argparse.ArgumentParser(description='Define parameters for experiment')
parser.add_argument('--imdir', required=True, type=str, help='Directory with dataset')
# Inputs
# Folder where the files are located
# imdir = "/home/alvaroeg/data/Patrick/For_Alvaro_07_01_2020_from_Patrick/New_GT_data/Training_data_II_prediction_modified"
# Names of the files to process
args = parser.parse_args()
imdir = args.imdir
param_dict = {
    'raw': '',
    'vessels': '_gtvessels',
    'largevessels': '_gtlargevessels',
    'tissue': '_gttissue'
}


def ims_gt(imdir, param_dict):
    lfiles = [x for x in os.listdir(imdir) if os.path.splitext(x)[1] == '.ims']
    lfiles_all_noext = [os.path.splitext(x)[0] for x in os.listdir(imdir)]
    lfiles_all = [x for x in os.listdir(imdir)]
    for nf, fname in enumerate(lfiles):
        print("\n{}/{} Taking image '{}'".format(nf+1, len(lfiles), fname))
        fname_aux, ext = os.path.splitext(fname)
        # Load the files specified
        if param_dict['raw'] in fname_aux and ext in ('.ims'):
            print("\nNew file: " + fname)
            fname_root = fname_aux[:-len(param_dict['raw']) or None]
            imbase = ImarisFiles(os.path.join(imdir, fname))
            # For each file, take the files containing gt
            for sgt, sname in param_dict.items():
                if sgt == 'raw':
                    continue
                sfile_name = fname_root + sname
                if sfile_name in lfiles_all_noext:
                    print("Loading channel '{:s}'".format(sname))
                else:
                    print("Channel '{:s}' not in file - skipping".format(sname))
                    continue
                sfile = lfiles_all[lfiles_all_noext.index(fname_root + sname)]
                sfile_noext, ext_gt = os.path.splitext(sfile)
                if ext_gt == '.ims':
                    imseg = ImarisFiles(os.path.join(imdir, sfile))
                    for cnum, cname in enumerate(imseg.channelNames):
                        if cname[:3] == 'GT ':
                            vol = imseg.getVolume(cnum)[..., 0]
                            print("Writing new channel: '{:s}'".format(cname))
                            imbase.write_dataset(vol, cname=cname)
                        else:
                            logging.warning(
                                "Channel {:d} with name '{:s}' does not have the required format".format(cnum, cname))
                elif ext_gt == '.mat':
                    with h5py.File(os.path.join(imdir, sfile)) as f:
                        if 'material_list' in f:
                            mnames = f['material_list']
                            nmodels = mnames.shape[1]
                        else:
                            logging.warning("material_list not found for {}. Assuming it is '{}'".format(sfile, sgt))
                            mnames = None
                            nmodels = 1
                        bvol = np.array(f['mibModel'])
                        bvol = np.transpose(bvol, [0, 2, 1])
                        for nn in range(nmodels):
                            if mnames:
                                snamemat = ''.join(map(chr, f[mnames[0][nn]][:]))
                            else:
                                snamemat = sgt
                            # if sname == 'vessels': # For Patrick
                            #     sname = 'sinusoids'
                            vol = (bvol == (nn + 1)).astype(np.uint8)
                            if snamemat[:2] == 'gt':
                                print("Correcting name: {:s} to {:s}".format(snamemat, snamemat[2:]))
                                snamemat = snamemat[2:]
                            print("Writing new channel: '{:s}'".format(snamemat))
                            imbase.write_dataset(vol, cname='GT ' + snamemat)
                else:
                    raise Exception('unknown extension')


ims_gt(imdir, param_dict)
