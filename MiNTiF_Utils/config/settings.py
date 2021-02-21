# (c) 2019-2021, Alvaro Gomariz, Luca Widmer  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel

import json
import warnings

from  utils import cnn_utils



current_modelfile_verison=1.0

def correct_settings(msettings):
    default_vals = {
        "model_file_version":current_modelfile_verison,
        "nlevels": 5,
        "augmentation": {
            "flipping": False,
            "zoom": False,
            "noise": False,
            "rotation": False
        },


        "patch_size": [
            572,
            572
        ],
        "pixel_size": 1,
        "model_transfer": None,
        "dropout": 0,

    }


    if ("scale_factor" not in msettings) or msettings["scale_factor"] is None:
        msettings["scale_factor"] = [2] * len(msettings["patch_size"])
    if ("kernel_size" not in msettings) or msettings["kernel_size"] is None:
        msettings["kernel_size"] = [3] * len(msettings["patch_size"])




    for k, v in default_vals.items():
        if k not in msettings:
            msettings[k] = v


    if ("channels" not in msettings) and ("markers" in msettings):
        msettings["channels"] = msettings["markers"]
    if ("labels" not in msettings) and ("classes" in msettings):
        msettings["labels"] = msettings["classes"]
    if "dataset_condition" not in msettings or msettings["dataset_condition"] is None:
        msettings["dataset_condition"] = {
            "channels": msettings["channels"],
            "labels": msettings["labels"]
        }
    if "dataset_type" not in msettings or msettings["dataset_type"] is None:
        msettings['dataset_type'] = 'semantic_seg'
    if "patch_size_out" not in msettings or msettings["patch_size_out"] is None:
        if msettings["padding"] == 'same':
            msettings["patch_size_out"] = msettings["patch_size"].copy()
        else:
            msettings["patch_size_out"] = list(cnn_utils.calc_imcrop(
                msettings["patch_size"],
                msettings['nlevels'],
                msettings['scale_factor'],
                msettings['kernel_size']
            )[0])
    # some variable combinations cause an underflow
    assert msettings["patch_size_out"][0] <= msettings["patch_size"][0]

    return msettings



def get_settings(filename):
    with open(filename, 'r') as f:
        msettings = json.load(f)
    msettings = correct_settings(msettings)
    # warn user of outdated model file
    if msettings["model_file_version"] != current_modelfile_verison:
        warnings.warn("model file is of version {}. most current version is{}".format(msettings["model_file_version"], current_modelfile_verison))
    return msettings
