import argparse
import time
from deepcell.applications import Mesmer
import numpy as np
import os
import tensorflow as tf
import math
import cv2
import tqdm
import logging
from utils import cell_dataset, auto_make_dir, instance2semantics,cvtColor,bitwise_not
models_logger = logging.getLogger(__name__)


def f_fillHole(im_in):
    im_floodfill = cv2.copyMakeBorder(im_in, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0])
    # im_floodfill = im_in.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill[2:-2, 2:-2])
    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv

    return im_out


models = {
    "Mesmer": os.path.join(os.path.abspath('.'), "src/methods/models/MultiplexSegmentation"),  
    "Nuclear": os.path.join(os.path.abspath('.'), "src/methods/models/NuclearSegmentation"),   
    "Cytoplasm": os.path.join(os.path.abspath('.'), "src/methods/models/CytoplasmSegmentation"),   
}


class iDeepCell(object):
    # https://deepcell.readthedocs.io/en/master/API/deepcell.html
    def __init__(self, model_type, device="CPU") -> None:
        assert model_type in ['Mesmer', 'Nuclear', 'Cytoplasm']
        self.type = model_type
        self.model = None
        self.app = None
        self._input_shape = (256, 256)
        self.whole_wh = None
        self.mat = None
        self.mat_batches = list()
        self.mask = None
        self._overlap = (0.1, 0.1)
        self.labeled_image = None


def deepcell_method(para, args):
    # https://github.com/vanvalenlab/deepcell-tf/blob/master/notebooks/applications/Mesmer-Application.ipynb
    import tifffile
    
    output_path = para.output
    img_type = para.img_type
    if os.path.isdir(para.image_path):
        imgs = cell_dataset(para.image_path, ['.tif', '.jpg', '.png'])
    else: imgs = [para.image_path]
    
    models_logger.info('Load Model - DeppCell')

    model_ = tf.keras.models.load_model(models['Mesmer'])
    app = Mesmer(model_)
    
    for it in tqdm.tqdm(imgs, desc="Deepcell"):
        img = cv2.imread(it,cv2.IMREAD_GRAYSCALE)
        im = np.stack((img, img), axis=-1)
        im = np.expand_dims(im, 0)
        mask = app.predict(im, image_mpp=0.5, compartment='nuclear')
        mask = np.squeeze(mask) 
        semantics = instance2semantics(mask)
        semantics[semantics > 0] = 255
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        name = os.path.split(it)[-1]    
        tifffile.imwrite(os.path.join(output_path,name), semantics, compression='zlib')
            
    models_logger.info('Dump result to {}'.format(output_path))
    

USAGE = 'DEEPCELL'
PROG_VERSION = 'v0.0.1'

def main():
    # test_stitch_entry('D:\\DATA\\stitchingv2_test\\motic\\result\\demo\\scope_info.json')
    arg_parser = argparse.ArgumentParser(usage=USAGE)
    arg_parser.add_argument("--version", action="version", version=PROG_VERSION)
    arg_parser.add_argument("-o", "--output", action="store", dest="output",
                            type=str, default=None, help="Save path of stitch result files.")
    arg_parser.add_argument("-i", "--image_path", action="store", dest="image_path",
                            type=str, default=None, help="FOV images storage location.")
    arg_parser.add_argument("-g", "--is_gpu", action="store", dest="is_gpu",
                            type=bool, default=False, help="Use GPU or not.")
    arg_parser.add_argument("-t", "--img_type", help="ss/he")
    arg_parser.set_defaults(func=deepcell_method)
    (para, args) = arg_parser.parse_known_args()
    print(para, args)
    para.func(para, args)


if __name__ == '__main__':
    import sys
    
    return_code = main()
    sys.exit(return_code)
