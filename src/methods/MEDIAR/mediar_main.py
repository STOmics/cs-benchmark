
import torch
import glob
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np

from train_tools import *
from train_tools.models import MEDIARFormer
from core.MEDIAR import Predictor, EnsemblePredictor
from utils import cell_dataset, auto_make_dir, instance2semantics
import os
import tifffile
import argparse
import tqdm
import logging
models_logger = logging.getLogger(__name__)

def MEDIAR_method(para, args):
    input_path = para.image_path
    output_path = para.output

    model_path = "/storeData/USER/data/01.CellBin/00.user/fanjinghong/code/cs-benchmark/src/methods/models/from_phase2.pth"
    #model_path = "/storeData/USER/data/01.CellBin/00.user/fanjinghong/code/cs-benchmark/src/methods/models/MEDIAR_finetuning.pth"
    weights = torch.load(model_path, map_location="cpu")

    model_args = {
    "classes": 3,
    "decoder_channels": [1024, 512, 256, 128, 64],
    "decoder_pab_channels": 256,
    "encoder_name": 'mit_b5',
    "in_channels": 3
    }
    device = (para.is_gpu == True) and "cuda:0" or "cpu"
    model = MEDIARFormer(**model_args)
    model.load_state_dict(weights, strict=False)
    
    predictor = Predictor(model, device, input_path, output_path, algo_params={"use_tta": True})
    _ = predictor.conduct_prediction()
    
    old_string = '_label.tiff'
    for filename in os.listdir(output_path):
        if old_string in filename:
            new_filename = filename.replace(old_string, '.tif')
            os.rename(os.path.join(output_path, filename), os.path.join(output_path, new_filename))
    for filename in os.listdir(output_path):
        if filename.endswith(".tif"):
            file_path = os.path.join(output_path, filename)
            masks = tifffile.imread(file_path)

            semantics = instance2semantics(masks)
            semantics[semantics > 0] = 255

            tifffile.imwrite(file_path, semantics.astype(np.uint8))

USAGE = 'MEDIAR'
PROG_VERSION = 'v0.0.1'

def main():
    
    arg_parser = argparse.ArgumentParser(usage=USAGE)
    arg_parser.add_argument("--version", action="version", version=PROG_VERSION)
    arg_parser.add_argument("-o", "--output", action="store", dest="output",
                            type=str, default=None, help="Save path of stitch result files.")
    arg_parser.add_argument("-i", "--image_path", action="store", dest="image_path",
                            type=str, default=None, help="FOV images storage location.")
    arg_parser.add_argument("-g", "--is_gpu", action="store", dest="is_gpu",
                            type=bool, default=False, help="Use GPU or not.")

    arg_parser.set_defaults(func=MEDIAR_method)
    (para, args) = arg_parser.parse_known_args()
    para.func(para, args)


if __name__ == '__main__':
    import sys
    
    return_code = main()
    sys.exit(return_code)