import argparse
import time
from deepcell.applications import Mesmer, NuclearSegmentation, CytoplasmSegmentation
import numpy as np
import os
import tensorflow as tf
import math
import cv2
import tqdm
import logging
from utils import cell_dataset, auto_make_dir, instance2semantics
models_logger = logging.getLogger(__name__)


def f_fillHole(im_in):
    ''' 对二值图像进行孔洞填充 '''
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
    # whole-cell 质", "nuclear 核", "both"
    "Mesmer": os.path.join(os.path.abspath('.'), "weights/deepcell/MultiplexSegmentation"),  # 多重分割
    "Nuclear": os.path.join(os.path.abspath('.'), "weights/deepcell/NuclearSegmentation"),  # 核分割
    "Cytoplasm": os.path.join(os.path.abspath('.'), "weights/deepcell/CytoplasmSegmentation")  # 细胞质分割
}


class iDeepCell(object):
    # https://deepcell.readthedocs.io/en/master/API/deepcell.html
    def __init__(self, model_type, device="GPU") -> None:
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

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            return 1
        self.model = tf.keras.models.load_model(model_path)
        if self.type == 'Mesmer':
            self.app = Mesmer(model=self.model)
        elif self.type == 'Nuclear':
            self.app = NuclearSegmentation(model=self.model)
        elif self.type == 'Cytoplasm':
            self.app = CytoplasmSegmentation(model=self.model)
        else:
            return 1
        return 0

    def _preprocessing(self, img):
        self.mat_batches = list()
        self.mat = img
        if self.mat.ndim == 2:
            if self.type == 'Mesmer':
                self.mat = np.stack((self.mat, self.mat), axis=-1)
                self.mat = self.mat.transpose((2, 0, 1))
            else:
                self.mat = np.expand_dims(self.mat, 0)
        c, h, w = self.mat.shape
        if c > w: 
            self.mat = self.mat.transpose((2, 0, 1))[:2]
            c, h, w = self.mat.shape
        self.whole_wh = (w, h)
        h0, w0 = [h, w]
        # h0, w0 = self._input_shape
        H, W = (math.ceil(h / h0) * h0, math.ceil(w / w0) * w0)
        self.mat = np.pad(self.mat, ((0, 0), (0, H - h), (0, W - w)), 'constant')
        for i in range(0, H, h0):
            mat_row_batches = list()
            for j in range(0, W, w0): mat_row_batches.append(self.mat[:, i: i + h0, j: j + w0])
            im = np.stack(mat_row_batches, axis=-1)
            # Input images are required to have 4 dimensions: batch, x, y, channel
            im = im.transpose((3, 1, 2, 0))
            self.mat_batches.append(im)
        del self.mat
        return 0

    def _sbatch_post(self, ):
        if self.labeled_image.shape[0] > 1:
            self.labeled_image = self.labeled_image.squeeze()
            self.labeled_image = np.hstack(self.labeled_image)
        else:
            self.labeled_image = self.labeled_image[0, :, :, 0]
        
    def inference(self, img, mpp=1.0,
                  radius=3, maxima_threshold=0.1, interior_threshold=0.01, small_objects_threshold=0):
        if self._preprocessing(img): return
        batch_masks = list()
        s = time.time()
        for bt in self.mat_batches:
            if self.type == 'Nuclear':
                postprocess_kwargs = {
                    'radius': radius,
                    'maxima_threshold': maxima_threshold,
                    'interior_threshold': interior_threshold,
                    'exclude_border': False,
                    'small_objects_threshold': small_objects_threshold
                }
                self.labeled_image = self.app.predict(bt, image_mpp=mpp, postprocess_kwargs=postprocess_kwargs)
            else:
                self.labeled_image = self.app.predict(bt, image_mpp=mpp)
            self._sbatch_post()
            batch_masks.append(self.labeled_image)
        e = time.time()
        mask = np.vstack(batch_masks)
        mask = mask[:self.whole_wh[1], :self.whole_wh[0]]
        semantics = instance2semantics(mask)
        semantics[semantics > 0] = 255
        semantics = f_fillHole(semantics)
        # semantics = uity.mask_to_outline(semantics)
        return semantics


def deepcell_method(para, args):
    # https://github.com/vanvalenlab/deepcell-tf/blob/master/notebooks/applications/Mesmer-Application.ipynb
    #import tifffile
    import cv2
    
    output_path = para.output
    if os.path.isdir(para.image_path):
        imgs = cell_dataset(para.image_path, ['.tif', '.jpg', '.png'])
    else: imgs = [para.image_path]
    
    models_logger.info('Load Model - DeppCell')
    
    idc = iDeepCell('Nuclear')
    idc.load_model(models['Nuclear'])
    
    for it in tqdm.tqdm(imgs, 'Deepcell'):
        img = cv2.imread(it,0)
        im = np.stack((img, img), axis=-1)
        im = np.expand_dims(im, 0)
        mask = idc.inference(im, mpp=1.0)
        out_file = auto_make_dir(it, src=para.image_path, output=output_path)
        tifffile.imwrite(out_file, mask, compression='zlib')
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

    arg_parser.set_defaults(func=deepcell_method)
    (para, args) = arg_parser.parse_known_args()
    print(para, args)
    para.func(para, args)


if __name__ == '__main__':
    import sys
    
    return_code = main()
    sys.exit(return_code)
