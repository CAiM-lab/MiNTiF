# (c) 2018-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
import os, itertools
from  data_read.imarisfiles import ImarisFiles
from  config import system
import argparse
import numpy as np
import urllib
import zipfile
import logging
import pandas as pd
import glob

logging.getLogger(__name__)


def shift_list(seq, n):
    n = n % len(seq)
    return seq if n == 0 else seq[n:] + seq[:n]


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def input_dataset(dataset):
    if len(dataset) == 1:
        ds_train, ds_val, ds_test = dataset, None, None
    if len(dataset) == 2:
        ds_train, ds_val = dataset
        ds_test = None
    if len(dataset) == 3:
        ds_train, ds_val, ds_test = dataset
    return ds_train, ds_val, ds_test


def check_channels(ifile, channels):
    ext = os.path.splitext(ifile)[1]
    if ext == '.ims':
        imFile = ImarisFiles(ifile)
        lchannels = set([x.lower() for x in channels])
        fchannels = set([x.lower() for x in imFile.channelNames])
        lspots = set([x.lower() for x in system.spots_GT])
        return len(fchannels.union(lspots).intersection(lchannels)) >= len(channels)
    else:
        Warning('extension not recognized, channels are not checked')
        return True


def input_filenames(filenames, fext=None, do_recursive=False):
    if fext is None:
        fext = ['.ims']
    if not isinstance(filenames, list):
        filenames = [filenames]
    l_trainpath = []
    for ifile in filenames:
        if os.path.isdir(filenames):
            if do_recursive:
                trainpath_aux = [os.path.join(dp, f) for dp, dn, filenames in os.walk(ifile) for f in filenames if
                                 os.path.splitext(f)[1] in fext]
            else:
                trainpath_aux = [os.path.join(ifile, x) for x in os.listdir(ifile) if
                                 os.path.splitext(x)[1] in fext]
        else:
            trainpath_aux = [ifile]
        for x in trainpath_aux:
            l_trainpath.append(x)


def input_files_format(in_file, channels=None, do_recursive=False, fext=None):
    if fext is None:
        fext = ['.ims']
    elif not isinstance(fext, list):
        fext = [fext]

    if in_file is None:
        return in_file
    elif not isinstance(in_file, list):
        in_file = [in_file]
    l_trainpath = []
    for ifile in in_file:
        if os.path.isdir(ifile):
            if do_recursive:
                trainpath_aux = [os.path.join(dp, f) for dp, dn, filenames in os.walk(ifile) for f in filenames if
                                 os.path.splitext(f)[1] in fext]
            else:
                trainpath_aux = [os.path.join(ifile, x) for x in os.listdir(ifile) if
                                 os.path.splitext(x)[1] in fext]
        else:
            trainpath_aux = [ifile]
        for x in trainpath_aux:
            l_trainpath.append(x)
    if not channels is None:
        l_trainpath = [x for x in l_trainpath if check_channels(x, channels)]
    return l_trainpath


def download_url_zip(data_url, download_dir, authentify=None):
    # Login if needed
    if authentify is not None:
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, authentify["root_url"], authentify["username"], authentify["password"])
        handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
        opener = urllib.request.build_opener(handler)
        opener.open(authentify["root_url"])
        urllib.request.install_opener(opener)
    logging.info("Downloading: {:s}".format(data_url))
    # Download file
    fname = data_url.split('/')[-1]
    download_dir = os.path.join(download_dir, fname)
    fdir, _ = urllib.request.urlretrieve(data_url, download_dir)
    # Unzip file
    with zipfile.ZipFile(fdir, 'r') as zip_ref:
        zip_ref.extractall(os.path.split(zip_ref.filename)[0])
    # Delete zip
    os.remove(fdir)


def invert_listdict(orig_dict):
    inv_dict = {}
    for id, vals in orig_dict.items():
        for v in vals:
            inv_dict[v] = id
    return inv_dict


def aggregate_metrics(save_dir, fname='metrics.csv', read_dir=None):
    cmetrics = pd.DataFrame()
    has_metrics = False
    if read_dir:
        # If reading boundmax
        dir_aux = read_dir
        fname_aux = 'metrics.csv'
        chcomb = "".join([str(int(x) - 1) for x in
                          os.path.split(read_dir)[1].replace("ch", "").replace("_l2", "").replace("_l4", "")])
        save_metrics = os.path.join(
            save_dir, 'metrics_ch' + chcomb + '.csv')
    else:
        dir_aux = save_dir
        fname_aux = fname
        save_metrics = os.path.join(save_dir, fname)
    for dir in os.listdir(dir_aux):
        metrics_file = os.path.join(dir_aux, dir, fname_aux)
        if os.path.isdir(os.path.join(dir_aux, dir)) and dir.isdigit():
            if os.path.isfile(metrics_file):
                has_metrics = True
                pmetrics = pd.read_csv(metrics_file,
                                       sep=',',
                                       header=0,
                                       index_col=0
                                       ).transpose()
                # pmetrics = pd.read_csv(os.path.join(save_dir, dir, 'metrics.csv'))
                cmetrics['model_cv' + dir] = pmetrics['model']
            elif 'notrain.txt' in os.listdir(os.path.join(dir_aux, dir)):
                has_metrics = True
    cmetrics.to_csv(save_metrics)
    return has_metrics


def aggregate_metrics_chdel(save_dir):
    cmetrics = pd.DataFrame()
    has_metrics = False
    for dir in os.listdir(save_dir):
        metrics_file = os.path.join(save_dir, dir, 'metrics.csv')
        if os.path.isdir(os.path.join(save_dir, dir)) and dir.isdigit() and os.path.isfile(metrics_file):
            has_metrics = True
            pmetrics = pd.read_csv(metrics_file,
                                   sep=',',
                                   header=0,
                                   index_col=0
                                   ).transpose()
            # pmetrics = pd.read_csv(os.path.join(save_dir, dir, 'metrics.csv'))
            cmetrics['model_cv' + dir] = pmetrics['model']
    cmetrics.to_csv(os.path.join(save_dir, 'metrics.csv'))
    return has_metrics


def aggregate_metrics_sample(save_dir, chdel=False):
    cmetrics = pd.DataFrame()
    has_metrics = False
    fdir = os.path.join(save_dir, '0')
    for metrics_file in glob.glob(os.path.join(fdir, 'metrics_sample*.csv')):
        has_metrics = True
        pmetrics = pd.read_csv(metrics_file,
                               sep=',',
                               header=0,
                               index_col=0
                               ).transpose()
        # pmetrics = pd.read_csv(os.path.join(save_dir, dir, 'metrics.csv'))
        sname = os.path.splitext(os.path.split(metrics_file)[1])[0].replace("metrics_", "")
        cmetrics[sname] = pmetrics['model']
    if has_metrics:
        cmetrics.to_csv(os.path.join(save_dir, 'metrics_samples.csv'))
    return has_metrics


def get_weights(class_counts, log_weight=True):
    class_counts = np.array(class_counts)
    class_weight = sum(class_counts) / (len(class_counts) * class_counts)
    if log_weight:
        return np.log(np.e + class_weight)
    else:
        return class_weight


def sort_markers(lmarkers='12345', length_first=True):
    if length_first:
        l = []
        nt = len(lmarkers)
        for n1 in range(nt):
            l += [lmarkers[n1]]

        for n1 in range(nt):
            for n2 in range(n1 + 1, nt):
                l += ["".join([lmarkers[x] for x in (n1, n2)])]

        for n1 in range(nt):
            for n2 in range(n1 + 1, nt):
                for n3 in range(n2 + 1, nt):
                    l += ["".join([lmarkers[x] for x in (n1, n2, n3)])]

        for n1 in range(nt):
            for n2 in range(n1 + 1, nt):
                for n3 in range(n2 + 1, nt):
                    for n4 in range(n3 + 1, nt):
                        l += ["".join([lmarkers[x] for x in (n1, n2, n3, n4)])]

        for n1 in range(nt):
            for n2 in range(n1 + 1, nt):
                for n3 in range(n2 + 1, nt):
                    for n4 in range(n3 + 1, nt):
                        for n5 in range(n4 + 1, nt):
                            l += ["".join([lmarkers[x] for x in (n1, n2, n3, n4, n5)])]
        l = ["".join(sorted([y for y in x])) for x in l]
    else:
        if length_first:
            l = []
            nt = len(lmarkers)

            for n1 in range(nt):
                for n2 in range(n1 + 1, nt):
                    for n3 in range(n2 + 1, nt):
                        for n4 in range(n3 + 1, nt):
                            for n5 in range(n4 + 1, nt):
                                l += ["".join([lmarkers[x] for x in (n1, n2, n3, n4, n5)])]
                            l += ["".join([lmarkers[x] for x in (n1, n2, n3, n4)])]
                        l += ["".join([lmarkers[x] for x in (n1, n2, n3, n4)])]
                    for n4 in range(n3 + 1, nt):
                        l += ["".join([lmarkers[x] for x in (n1, n2, n4)])]
                for n3 in range(n2 + 1, nt):
                    l += ["".join([lmarkers[x] for x in (n1, n3, n4)])]

            for n1 in range(nt):
                for n2 in range(n1 + 1, nt):
                    for n3 in range(n2 + 1, nt):
                        for n4 in range(n3 + 1, nt):
                            l += ["".join([lmarkers[x] for x in (n1, n2, n3, n4, n5)])]

            for n1 in range(nt):
                for n2 in range(n1 + 1, nt):
                    for n3 in range(n2 + 1, nt):
                        l += ["".join([lmarkers[x] for x in (n1, n2, n3, n4, n5)])]

            for n1 in range(nt):
                for n2 in range(n1 + 1, nt):
                    l += ["".join([lmarkers[x] for x in (n1, n2, n3, n4, n5)])]

            for n1 in range(nt):
                l += ["".join([lmarkers[x] for x in (n1, n2, n3, n4, n5)])]
            l = ["".join(sorted([y for y in x])) for x in l]
    return l


def rename_channels(names_old, lmarkers):
    mrename = {str(k): str(v) for k, v in zip(lmarkers, range(1, len(lmarkers) + 1))}
    for k, v in mrename.items():
        names_old = [x.replace(k, chr(int(v))) for x in names_old]
    for v in mrename.values():
        names_old = [x.replace(chr(int(v)), str(v)) for x in names_old]
    names_aux = ['m' + "".join(sorted(x.replace('ch', ''))) for x in names_old]
    names_new_sorted = []
    for n in range(1, len(lmarkers) + 1):
        names_new_sorted += sorted([x for x in names_aux if str(n) in x], key=lambda x: (-len(x), x), reverse=False)
        names_aux = [x for x in names_aux if str(n) not in x]
    names_sorted = [x.replace("m", "") for x in names_new_sorted]
    for k, v in mrename.items():
        names_sorted = [x.replace(v, chr(int(k))) for x in names_sorted]
    for k in mrename.keys():
        names_sorted = [x.replace(chr(int(k)), k) for x in names_sorted]
    names_sorted = ["".join(sorted(x.replace('ch', ''))) for x in names_sorted]
    return names_sorted


def marker_combinations(nmarkers):
    return list(set([tuple(set(x)) for x in itertools.product(np.arange(nmarkers), repeat=nmarkers)]))
