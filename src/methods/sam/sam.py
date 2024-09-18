import cv2
import numpy as np
import os
import torch

import scipy.io as sio
import matplotlib.pyplot as plt
import tifffile
import tqdm

from utils import cell_dataset, auto_make_dir, instance2semantics
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


# https://github.com/artcmd/SAM-cell-detection/blob/main/detection-demo.ipynb
# !pip install git+https://github.com/facebookresearch/segment-anything.git
# pip install opencv-python pycocotools matplotlib onnxruntime onnx
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


def sam_method(para, args):
    # sam_checkpoint = os.path.join(os.path.abspath('.'), 'weights/sam/sam_vit_h_4b8939.pth')
    sam_checkpoint = '/storeData/USER/data/01.CellBin/00.user/fanjinghong/code/benchmark2/weights/sam/sam_vit_b_01ec64.pth'
    model_type = "vit_b"
    if para.is_gpu == True:
        device = 'cuda'
    else:
        device = 'cpu'

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    output_path = para.output
    if os.path.isdir(para.image_path):
        imgs = cell_dataset(para.image_path, ['.tif', '.jpg', '.png'])
    else: imgs = [para.image_path]
    
    # images: list, output_path: str, image_path: str
    for i in tqdm.tqdm(imgs, 'SAM'):
        # if 'mouse_placenta' not in i: continue
        i_output_path = auto_make_dir(i, src=para.image_path, output=output_path)
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = (image.shape)
        mask = np.zeros((h, w), dtype=np.uint16)

        masks = mask_generator.generate(image)
        for i, m in enumerate(masks): 
            mask[np.where(m['segmentation'] == True)] = i
        mask = instance2semantics(mask)
        mask[mask > 0] = 255
        
        tifffile.imwrite(i_output_path, mask)


USAGE = 'SAM CELL'
PROG_VERSION = 'v0.0.1'
import argparse


def main():
    arg_parser = argparse.ArgumentParser(usage=USAGE)
    arg_parser.add_argument("--version", action="version", version=PROG_VERSION)
    arg_parser.add_argument("-o", "--output", action="store", dest="output",
                            type=str, default=None, help="Save path of stitch result files.")
    arg_parser.add_argument("-i", "--image_path", action="store", dest="image_path",
                            type=str, default=None, help="FOV images storage location.")
    arg_parser.add_argument("-g", "--is_gpu", action="store", dest="is_gpu",
                            type=bool, default=False, help="Use GPU or not.")

    arg_parser.set_defaults(func=sam_method)
    (para, args) = arg_parser.parse_known_args()
    para.func(para, args)


if __name__ == '__main__':
    return_code = main()
    sys.exit(return_code)


