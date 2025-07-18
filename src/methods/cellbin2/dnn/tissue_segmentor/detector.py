from cellbin2.dnn.onnx_net import OnnxNet
from cellbin2.dnn.tissue_segmentor.preprocess import TissueSegPreprocess
from cellbin2.dnn.tissue_segmentor.postprocess import TissueSegPostprocess
from cellbin2.dnn.tissue_segmentor.processing import f_preformat, f_postformat

from typing import Optional, Union, List
from cellbin2.utils.common import TechType
from cellbin2.utils import clog
from cellbin2.contrib.param import TissueSegOutputInfo


class TissueSegmentationBcdu(object):
    def __init__(self,
                 input_size: tuple = (512, 512, 1),
                 stain_type: TechType = '',
                 threshold_list: List = None,
                 gpu: int = -1,
                 mode: str = "onnx",
                 num_threads: int = 0,
                 preprocess: Optional[TissueSegPreprocess] = None,
                 postprocess: Optional[TissueSegPostprocess] = None
                 ):

        self.INPUT_SIZE = input_size

        self.stain_type = stain_type
        self.gpu = gpu
        self.mode = mode
        self.model = None
        self.mask_num = None
        self.num_threads = num_threads
        self.threshold_list = threshold_list

        self.pre_format = f_preformat
        self.post_format = f_postformat

        self.preprocess = preprocess
        self.postprocess = postprocess

    def f_init_model(self, model_path):
        self.model = OnnxNet(model_path=model_path, gpu=self.gpu, num_threads=self.num_threads)
        self.INPUT_SIZE = self.model.f_get_input_shape()

        clog.info(f'model input size:{self.INPUT_SIZE}')

    def f_predict(self, img) -> TissueSegOutputInfo:
        pred_out = TissueSegOutputInfo()
        if self.stain_type == TechType.IF:
            img = self.preprocess(img=img, stain_type=self.stain_type, input_size=None)
            threshold_list, pred = self.postprocess(img=img, stain_type=self.stain_type, src_shape=None, threshold_list=self.threshold_list)
            pred_out.tissue_mask = pred

            pred_out.threshold_list = threshold_list
            return pred_out
        src_shape = img.shape[:2]
        img = self.pre_format(self.preprocess(img=img, stain_type=self.stain_type, input_size=self.INPUT_SIZE))

        pred = self.model.f_predict(img)
        pred = self.postprocess(self.post_format(pred), stain_type=self.stain_type, src_shape=src_shape)

        pred_out.tissue_mask = pred
        return pred_out
