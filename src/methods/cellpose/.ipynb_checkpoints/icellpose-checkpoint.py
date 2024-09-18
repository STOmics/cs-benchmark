# RUN CELLPOSE
import os
import glob
from cellpose import models, io
import tifffile
import argparse
import tqdm
import logging
models_logger = logging.getLogger(__name__)
from utils import cell_dataset, auto_make_dir, instance2semantics


def cellpose_method(para, args):
    input_path = para.image_path
    output_path = para.output
    if os.path.isdir(para.image_path):
        imgs = cell_dataset(para.image_path, ['.tif', '.jpg', '.png'])
    else: imgs = [para.image_path]
    
    # DEFINE CELLPOSE MODEL
    # model_type='cyto' or model_type='nuclei'
    model = models.Cellpose(gpu=True, model_type='cyto')

    # define CHANNELS to run segementation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0
    # channels = [0,0]
    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    # channels = [0,0] # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

    # or if you have different types of channels in each image

    # channels = [[2, 0], [2, 0]]
    chan = [0, 0]

    # if diameter is set to None, the size of the cells is estimated on a per image basis
    # you can set the average cell `diameter` in pixels yourself (recommended) 
    # diameter can be a list or a single number for all images

    # you can run all in a list e.g.
    # >>> imgs = [io.imread(filename) in for filename in files]
    # >>> masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=channels)
    # >>> io.masks_flows_to_seg(imgs, masks, flows, diams, files, channels)
    # >>> io.save_to_png(imgs, masks, flows, files)

    # or in a loop
    for i in tqdm.tqdm(range(len(imgs)), 'Cellpose'):
        filename = imgs[i]
        img = io.imread(filename)
        masks, flows, styles, diams = model.eval(img, diameter=None, channels=chan)
        out_file = auto_make_dir(filename, src=para.image_path, output=output_path)
        semantics = instance2semantics(masks)
        semantics[semantics > 0] = 255
        tifffile.imwrite(out_file, semantics, compression='zlib')
        # save results so you can load in gui
        # io.masks_flows_to_seg(img, masks, flows, diams, filename, chan)


USAGE = 'Celpose'
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

    arg_parser.set_defaults(func=cellpose_method)
    (para, args) = arg_parser.parse_known_args()
    print(para, args)
    # weights = os.path.join(os.path.abspath('.'), "weights/cellpose")
    # models_logger.info('Load Model - Cellpose from {}'.format(weights))
    
    # os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = weights
    para.func(para, args)


if __name__ == '__main__':
    import sys
    
    return_code = main()
    sys.exit(return_code)