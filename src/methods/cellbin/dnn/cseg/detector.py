from cellbin.image.wsi_split2 import SplitWSI
from cellbin.utils import clog
from cellbin.dnn.cseg import CellSegmentation
from cellbin.dnn.cseg.predict import CellPredict
from cellbin.dnn.cseg.processing import f_prepocess, f_preformat, f_postformat, f_preformat_mesmer, \
    f_postformat_mesmer, f_padding, f_fusion
from cellbin.dnn.onnx_net import OnnxNet

import numpy as np


# TensorRT/ONNX
# HE/DAPI/mIF
class Segmentation(CellSegmentation):

    def __init__(self, model_path="", net="bcdu", mode="onnx", gpu="-1", num_threads=0,
                 win_size=(256, 256), intput_size=(256, 256, 1), overlap=16):
        """

        :param model_path:
        :param net:
        :param mode:
        :param gpu:
        :param num_threads:
        """
        # self.PREPROCESS_SIZE = (8192, 8192)

        self._win_size = win_size
        self._input_size = intput_size
        self._overlap = overlap

        self._net = net
        self._gpu = gpu
        self._mode = mode
        #self._model_path = model_path
        self._model = None
        self._sess = None
        self._num_threads = num_threads
        #self._f_init_model()

    def f_init_model(self,model_path):
        """
        init model
        """
        self._model = OnnxNet(model_path, self._gpu, self._num_threads)

        if self._net == "mesmer":
            self._sess = CellPredict(self._model, f_preformat_mesmer, f_postformat_mesmer)
        else:
            self._sess = CellPredict(self._model, f_preformat, f_postformat)

    def f_predict(self, img):
        """

        :param img:CHANGE
        :return: 掩模大图
        2023/09/21 @fxzhao 设置need_fun_ret为False,当前版本未用到此结果
        """
        img = f_prepocess(img)
        sp_run = SplitWSI(img, self._win_size, self._overlap, 100, False, True, False, np.uint8,dst_shape=(img.shape[:2]))
        sp_run.f_set_run_fun(self._sess.f_predict)
        sp_run.f_set_pre_fun(f_padding, self._win_size)
        sp_run.f_set_fusion_fun(f_fusion)
        _, _, pred = sp_run.f_split2run()
        #pred[pred > 0] = 1
        return pred


def main():
    import tifffile
    import os
    import argparse

    parser = argparse.ArgumentParser(description="you should add those parameter")
    parser.add_argument('-i', "--input", help="the input img path")
    parser.add_argument('-o', "--output", help="the output file")
    parser.add_argument("-g", "--gpu", help="the gpu index", default=-1)
    parser.add_argument("-n", "--net", help="bcdu or mesmer", default="bcdu")
    parser.add_argument("-m", "--mode", help="onnx or tf", default="onnx")
    parser.add_argument("-th", "--num_threads", help="num_threads", default="0")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    gpu = args.gpu
    mode = args.mode
    num_threads = args.num_threads
    net = args.net
    if input_path is None or output_path is None:
        print("please check your parameters")
        sys.exit()
    print(args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    abs_path = os.path.dirname(os.path.abspath(__file__))
    if mode == "onnx":
        if net == "mesmer":
            model_path = os.path.join(abs_path, r"D:\code\public\cell_segmentation_v03\model/weight_mesmer_20.onnx")
        else:
            model_path = os.path.join(abs_path, r"D:\code\public\cell_segmentation_v03\model/weight_cell_221008.onnx")
    else:
        model_path = os.path.join(abs_path, r"D:\code\public\cell_segmentation_v03\model/weight_cell_221008.hdf5")
        # model_path = os.path.join(abs_path, "model/weight_cell_he_20221226.hdf5")
    img = tifffile.imread(input_path)
    clog.info(f"start load model from {model_path}")
    sg = Segmentation(model_path=model_path, net=net, mode=mode, gpu=gpu, num_threads=int(num_threads))
    sg.f_init_model(model_path=model_path)
    clog.info(f"model loaded,start predict")
    pred = sg.f_predict(img)
    clog.info(f"predict finish,start write")
    tifffile.imwrite(output_path, pred)
    clog.info(f"finish!")


if __name__ == '__main__':
    import sys

    main()
    sys.exit()
