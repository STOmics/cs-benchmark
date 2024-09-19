
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
    if os.path.isdir(para.image_path):
        imgs = cell_dataset(para.image_path, ['.tif', '.jpg', '.png'])
    else: imgs = [para.image_path]
    model_path1 = "../../models/from_phase1.pth"
    model_path2 = "../../models/from_phase2.pth"
    weights1 = torch.load(model_path1, map_location="cpu")
    weights2 = torch.load(model_path2, map_location="cpu")

    model_args = {
    "classes": 3,
    "decoder_channels": [1024, 512, 256, 128, 64],
    "decoder_pab_channels": 256,
    "encoder_name": 'mit_b5',
    "in_channels": 3
    }
    device = (args.is_gpu == True) and "cuda:0" or "cpu"
    model1 = MEDIARFormer(**model_args)
    model1.load_state_dict(weights1, strict=False)
    
    model2 = MEDIARFormer(**model_args)
    model2.load_state_dict(weights2, strict=False)
    
    predictor = Predictor(model2, device, input_path, output_path, algo_params={"use_tta": True})
    _ = predictor.conduct_prediction()
    
    old_string = '_label.tiff'
    for filename in os.listdir(output_path):
        if old_string in filename:
            new_filename = filename.replace(old_string, '.tif')
            os.rename(os.path.join(output_path, filename), os.path.join(output_path, new_filename))

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