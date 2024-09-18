import os
import numpy as np
import tifffile as tif

from cellbin.utils import clog
from cellbin.modules.mif_calibrate import FFTRegister


class MifCalibrate:
    def __init__(self, dapi_img, if_img):
        """
        Args:
            dapi_img: str | array
            if_img: str | array
        """
        if isinstance(dapi_img, str):
            self.dapi_img = tif.imread(dapi_img)
        elif isinstance(dapi_img, np.ndarray):
            self.dapi_img = dapi_img
        else:
            self.dapi_img = None
            clog.info("DAPI file format error.")

        if isinstance(if_img, str):
            self.if_img = tif.imread(if_img)
        elif isinstance(dapi_img, np.ndarray):
            self.if_img = if_img
        else:
            self.if_img = None
            clog.info("IF file format error.")

        self.calibrate = FFTRegister()

    def calibration(self):
        """
        Returns:
            if_img: 转换后的IF图像
            result: 转换参数
        """
        if self.dapi_img is None or self.if_img is None:
            clog.info("File format error.")
            return

        clog.info("Calibrate start.")
        dapi_img, if_img = self.calibrate.pad_same_image(self.dapi_img, self.if_img)
        result = self.calibrate.calibration(dapi_img, if_img)

        return if_img, result
