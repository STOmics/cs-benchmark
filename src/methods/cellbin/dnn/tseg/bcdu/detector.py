# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from cellbin.dnn.tseg.bcdu.processing import f_preformat, f_prepocess, f_postformat
from cellbin.dnn.onnx_net import OnnxNet
from cellbin.image.augmentation import f_resize

import copy
import numpy as np


class TissueSegmentationBcdu(object):
    def __init__(self, input_size=(512, 512, 1),
                 img_type="SSDNA", gpu="-1", mode="onnx", num_threads=0, deep="deep"):
        """

        :param deep:deep 深度学习推理
        :param img_type:ssdna，rna
        :param model_path:模型路径
        """
        self._INPUT_SIZE = input_size

        self._deep = deep
        self._img_type = img_type
        # self._model_path = None
        self._gpu = gpu
        self._mode = mode
        self._model = None
        self.mask_num = None
        self._num_threads = num_threads
        # if self._deep == "deep":
        #     self._init_model()

    def f_init_model(self, model_path):
        """

        """
        self._model = OnnxNet(model_path, self._gpu, self._num_threads)
        self._INPUT_SIZE = self._model.f_get_input_shape()

    def f_predict(self, img):
        """

        :param img:CHANGE
        :return: 模型输入图，掩模小图
        """
        img = np.squeeze(img)
        src_shape = img.shape[:2]

        img = f_preformat(f_prepocess(img, self._img_type, self._INPUT_SIZE))
        pred = self._model.f_predict(copy.deepcopy(img))
        pred = f_postformat(pred, self._img_type)
        pred = f_resize(pred, src_shape)

        # pred = tools.uity.f_fill_hole(pred)
        # pred[pred > 0] = 1

        self.mask_num = np.sum(pred)
        return pred


def main():
    import tifffile
    import os
    import argparse
    parser = argparse.ArgumentParser(description="you should add those parameter")
    parser.add_argument('-i', "--input", help="the input img path")
    parser.add_argument('-o', "--output", help="the output file")
    parser.add_argument('-t', "--img_type", help="ssdna or rna", default="ssdna")
    parser.add_argument("-g", "--gpu", help="the gpu index", default=-1)
    parser.add_argument("-m", "--mode", help="onnx or tf", default="onnx")
    parser.add_argument("-d", "--deep", help="deep", default="deep")
    parser.add_argument("-th", "--num_threads", help="num_threads", default="0")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    img_type = args.img_type
    gpu = args.gpu
    mode = args.mode
    num_threads = args.num_threads
    if input_path is None or output_path is None:
        print("please check your parameters")
        sys.exit()
    print(args)
    abs_path = os.path.dirname(os.path.abspath(__file__))
    if mode == "onnx" and img_type == "rna":
        model_path = os.path.join(abs_path, "model/weight_rna_220909.onnx")
    elif mode == "tf" and img_type == "ssdna":
        model_path = os.path.join(abs_path, "model/weight_bic_220822.hdf5")
        # model_path = os.path.join(abs_path, "model/he_tissue_cut.hdf5")
    elif mode == "tf" and img_type == "rna":
        model_path = os.path.join(abs_path, "model/weight_rna_220909.hdf5")
    else:
        model_path = os.path.join(abs_path, r"D:\code\public\tissuecut\model/weight_bic_220822.onnx")

    img = tifffile.imread(input_path)
    sg = TissueSegmentationBcdu(gpu=gpu, img_type=img_type, mode=mode,
                                num_threads=int(num_threads))
    sg.f_init_model(model_path=model_path)
    pred = sg.f_predict(img)
    tifffile.imwrite(output_path, pred)


if __name__ == '__main__':
    import sys

    main()
    sys.exit()
