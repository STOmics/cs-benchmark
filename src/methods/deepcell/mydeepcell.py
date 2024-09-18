import sys

sys.path.append("/storeData/USER/data/01.CellBin/00.user/cenweixuan/cellseg3rd/deepcell")
import glob
import cv2
import copy

import argparse
import time
from deepcell.applications import Mesmer, NuclearSegmentation, CytoplasmSegmentation
import numpy as np
import os
import tensorflow as tf
import math

models = {
    "Mesmer": os.path.join(os.path.split(os.path.realpath(__file__))[0], "MultiplexSegmentation"),
    "Nuclear": os.path.join(os.path.split(os.path.realpath(__file__))[0], "NuclearSegmentation"),
    "Cytoplasm": os.path.join(os.path.split(os.path.realpath(__file__))[0], "CytoplasmSegmentation")
}


class CellMask(object):
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
        print("predict time:{}".format(e - s))
        mask = np.vstack(batch_masks)
        mask = mask[:self.whole_wh[1], :self.whole_wh[0]]
        semantics = instance2semantics(mask)
        return semantics


def instance2semantics(ins):
    h, w = ins.shape[:2]
    tmp0 = ins[1:, 1:] - ins[:h - 1, :w - 1]
    ind0 = np.where(tmp0 != 0)

    tmp1 = ins[1:, :w - 1] - ins[:h - 1, 1:]
    ind1 = np.where(tmp1 != 0)
    ins[ind1] = 0
    ins[ind0] = 0
    ins[np.where(ins > 0)] = 1
    return np.array(ins, dtype=np.uint8)


def f_rgb2gray(img, need_not=False):
    """
    rgb2gray

    :param img: (CHANGE) np.array
    :param need_not: if need bitwise_not
    :return: np.array
    """
    if img.ndim == 3:
        if img.shape[0] == 3 and img.shape[1] > 3 and img.shape[2] > 3:
            img = img.transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if need_not:
            img = cv2.bitwise_not(img)
    return img


def f_ij_16_to_8(img, chunk_size=1000):
    """
    16 bits img to 8 bits

    :param img: (CHANGE) np.array
    :param chunk_size: chunk size (bit)
    :return: np.array
    """

    if img.dtype == 'uint8':
        return img
    dst = np.zeros(img.shape, np.uint8)
    p_max = np.max(img)
    p_min = np.min(img)
    scale = 256.0 / (p_max - p_min + 1)
    for idx in range(img.shape[0] // chunk_size + 1):
        sl = slice(idx * chunk_size, (idx + 1) * chunk_size)
        win_img = copy.deepcopy(img[sl])
        win_img = np.int16(win_img)
        win_img = (win_img & 0xffff)
        win_img = win_img - p_min
        win_img[win_img < 0] = 0
        win_img = win_img * scale + 0.5
        win_img[win_img > 255] = 255
        dst[sl] = np.array(win_img).astype(np.uint8)
    return dst


def main(file_lst, output_path, gpu, mpp=1.0):
    import tifffile
    if int(gpu) > -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        tf.config.experimental.list_logical_devices('GPU')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.experimental.list_logical_devices('CPU')

    cm = CellMask('Mesmer')
    cm.load_model(models['Mesmer'])

    for file in file_lst:
        name = os.path.split(file)[-1]
        img = tifffile.imread(file)

        mask = cm.inference(img, mpp=mpp)
        tifffile.imwrite(os.path.join(output_path, name), mask, compression="zlib")


def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input_path', action='store', help='image path')
    ap.add_argument('-o', '--output_path', action='store', help='result path')
    ap.add_argument('-g', '--gpu', action='store', help='gpu', default='-1')
    ap.add_argument('-m', '--mpp', action='store', help='mpp', default=1.0)
    return ap.parse_args()


if __name__ == '__main__':
    args = args_parse()
    print(args)
    args = vars(args)

    input_path = args["input_path"]
    output = args["output_path"]
    gpu = args["gpu"]
    mpp = args["mpp"]

    file_lst = []
    if os.path.isdir(input_path):
        file_lst = glob.glob(os.path.join(input_path, "*.tif"))
    else:
        file_lst = [input_path]

    main(file_lst, output, gpu, float(mpp))
    sys.exit()
