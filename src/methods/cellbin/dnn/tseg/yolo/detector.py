from cellbin.dnn.tseg import TissueSegmentation
from cellbin.dnn.tseg.yolo.processing import f_process_mask, f_preformat, f_scale_image, f_img_process
from cellbin.dnn.tseg.yolo.nms import f_non_max_suppression
from cellbin.dnn.onnx_net import OnnxNet

import numpy as np
import cv2


class TissueSegmentationYolo(TissueSegmentation):
    def __init__(self,
                 gpu=-1,
                 num_threads=0,
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 agnostic_nms=False,  # class-agnostic NMS
                 half=False,  # use FP16 half-precision inference
                 ):
        self._INPUT_SIZE = (640, 640)
        self._model_path = None
        self._gpu = gpu
        self._num_threads = num_threads
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._max_det = max_det
        self._classes = classes
        self._agnostic_nms = agnostic_nms
        self._half = half
        self._model = None
        self.mask_num = 0
        # self._f_init_model()

    def f_init_model(self, model_path):
        """
        init model
        """
        self._model = OnnxNet(model_path, self._gpu, self._num_threads)

    def f_predict(self, img):
        source_shape = img.shape[:2]
        img = f_img_process(img)
        img = f_preformat(img)
        pred, proto = self._model.f_predict(img)
        pred = f_non_max_suppression(pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms,
                                     self._max_det, nm=32)
        mask = np.zeros(source_shape, np.uint8)
        for i, det in enumerate(pred):  # per image
            if len(det):
                det[:, :2] = det[:, :2] * 0.9
                det[:, 2:4] = det[:, 2:4] * 1.1
                det[:, :4] = np.clip(det[:, :4], 0.0, self._INPUT_SIZE[0])
                masks = f_process_mask(proto[i], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)  # HWC
                mask = np.zeros(self._INPUT_SIZE, dtype=np.float32)
                mask_overlap = np.zeros(self._INPUT_SIZE, dtype=np.float32)
                for j in range(masks.shape[0]):
                    min_x, min_y, max_x, max_y = np.uint16(det[j, :4])
                    mask_overlap[min_y:max_y, min_x:max_x] = np.add(mask_overlap[min_y:max_y, min_x:max_x], 1.0)
                    m = masks[j]
                    m = np.asarray(m, dtype=np.float32)
                    mask[min_y:max_y, min_x:max_x] = np.add(mask[min_y:max_y, min_x:max_x], m[min_y:max_y, min_x:max_x])
                mask_overlap = np.clip(mask_overlap, 1.0, max(1.0, mask_overlap.max()))
                mask = np.divide(mask, mask_overlap)
                mask[mask > self._conf_thres * 2] = 1.0
                mask = np.uint8(mask)
                mask = cv2.resize(mask, (source_shape[1], source_shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
        self.mask_num = np.sum(mask)

        return mask


def main():
    import os
    import tifffile
    seg = TissueSegmentationYolo()
    seg.f_init_model(r"D:\code\envs\tissuecut_yolo\tissueseg_yolo_SH_20230131.onnx", )
    img = tifffile.imread(r"D:\stock\dataset\test\fov_stitched.tif")
    mask = seg.f_predict(img, type="matrix")
    tifffile.imwrite(r"D:\stock\dataset\test\1\fov_stitched.tif", mask)


if __name__ == '__main__':
    import sys

    main()
    sys.exit()
