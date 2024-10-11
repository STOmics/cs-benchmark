# RUN CELLPOSE
import os
import glob
from cellpose import models, io,denoise
import tifffile
import argparse
import tqdm
import logging
models_logger = logging.getLogger(__name__)
from utils import cell_dataset, auto_make_dir, instance2semantics,cvtColor,bitwise_not
import cv2
def get_image_size(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    return width, height
def cellpose_method(para, args):
    input_path = para.image_path
    output_path = para.output
    if os.path.isdir(input_path):
        imgs = cell_dataset(input_path, ['.tif', '.jpg', '.png'])
    else: imgs = [para.image_path]
    img_type = para.img_type
    model = models.Cellpose(gpu=True, model_type='cyto3')
    chan = [0, 0]
    for i in tqdm.tqdm(range(len(imgs)), 'Cellpose3'):
        filename = imgs[i]
        img = io.imread(filename)
        masks, flows, styles, diams = model.eval(img, diameter=None, channels=chan)
        semantics = instance2semantics(masks)
        semantics[semantics > 0] = 255
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        name = os.path.split(filename)[-1]    
        tifffile.imwrite(os.path.join(output_path,name), semantics, compression='zlib')



USAGE = 'Cellpose3'
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
    arg_parser.add_argument("-t", "--img_type", help="ss/he")
    arg_parser.set_defaults(func=cellpose_method)
    (para, args) = arg_parser.parse_known_args()
    para.func(para, args)


if __name__ == '__main__':
    import sys
    
    return_code = main()
    sys.exit(return_code)