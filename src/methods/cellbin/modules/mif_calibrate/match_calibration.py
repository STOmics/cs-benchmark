#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :match_calibration.py
# @Time      :2022/10/9 16:49
# @Author    :kuisu_dgut@163.com

import numpy as np
import cv2
import os

from .imreg_dft import translation, similarity
from .utils import tiff_read, show_3d, affineMatrix3D, tiff_write, mat_channel
from .transform_vips import PyvipsImage


class Register(object):
    def __init__(self):
        self.dst_path = None
        self.src_path = None
        self.output_path = None

        self._dst_img = None
        self._src_img = None
        self.img_dtype = None

        # output
        self.confi = None
        self.offset = None

    def init_img(self, src_path, dst_path, output_path=None):
        self.src_path = src_path
        self.dst_path = dst_path
        self._dst_img = tiff_read(self.dst_path)
        self._src_img = tiff_read(self.src_path)
        self.img_dtype = self._dst_img.dtype
        if output_path is not None:
            self.output_path = output_path
        else:
            self.output_path = os.path.join(os.path.dirname(self.src_path))


class FFTRegister(Register):
    def __init__(self):
        self.debug = False
        self.fft_size = 3000

    @staticmethod
    def pad_same_image(dst, src):
        w0, h0 = dst.shape
        w1, h1 = src.shape

        img_dtype = dst.dtype
        sx = w0 / w1
        sy = h0 / h1
        scale = min(sx, sy)

        im2 = cv2.resize(src, dsize=None, dst=None, fx=scale, fy=scale)
        w2, h2 = im2.shape
        im1_new = np.zeros_like(dst, dtype=img_dtype)  # im1的size小于im0
        im1_new[:w2, :h2] = im2
        return dst, im1_new

    @staticmethod
    def pad_src_image(dst, src):
        w0, h0 = dst.shape
        w1, h1 = src.shape

        img_dtype = dst.dtype

        # w2, h2 = im2.shape
        im1_new = np.zeros_like(dst, dtype=img_dtype)  # im1的size小于im0
        im1_new[:w1, :h1] = src
        # im1_new = src[:w0, :h0]
        return dst, im1_new

    @staticmethod
    def img_scale(dst, std_size=3000):
        # 将图像控制在3000以内
        w, h = dst.shape
        size = max(w, h)
        scale = (std_size / size)
        if scale > 1:
            scale = 1  # 只缩小, 不放大
        return scale

    def calibration(self, dst_img, src_img, similarty = False):
        '''
        IF calibration. The input images are required to be single channel and the size is the same
        :param dst_img: DAPI image
        :param src_img: IF image
        :param similarty: Whether there are differences in scale and angle
        :return: diction
        '''
        assert dst_img.shape == src_img.shape and mat_channel(dst_img) == 1, "dst_img shape: {}, src_img shape: {}. " \
                                                                             "the size is different.".format(
            dst_img.shape, src_img.shape)

        init_scale = self.img_scale(dst_img, self.fft_size)
        dst_img_std = cv2.resize(dst_img, dsize=None, dst=None, fx=init_scale, fy=init_scale)
        src_img_std = cv2.resize(src_img, dsize=None, dst=None, fx=init_scale, fy=init_scale)
        if similarty:
            result = similarity(dst_img_std, src_img_std)
        else:
            result = translation(dst_img_std, src_img_std)
        if "scale" not in result:
            result["scale"] = 1
        if "angle" not in result:
            result["angle"] = 0
        if "dst_shape" not in result:
            result["dst_shape"] = dst_img.shape[:2]
        # result['scale'] *= self.pad_same_scale
        result["tvec"] /= init_scale
        if "tvec" not in result:
            result["tvec"] = [0, 0]
        result["offset"] = [int(result["tvec"][0]), int(result["tvec"][0])]

        if self.debug and "confi_mask" in result:
            show_3d(result['confi_mask'])
        return {"offset": result['tvec'], "scale": result["scale"], "angle": result["angle"],
                "confi": result['success'], "dst_shape": result["dst_shape"]}

    @staticmethod
    def transform_img_vips(img, offset, scale=1, angle=0, dst_shape=None):
        '''
        transform the image
        :param img:
        :param offset: [x,y]
        :param scale:
        :param angle:
        :return: img_t
        '''
        # 对图像进行变换
        # w, h = img.shape
        # center = [int(h / 2), int(w / 2)]
        # H = affineMatrix3D(center, shift=offset, scale=scale, rotation=angle)
        pi = PyvipsImage()
        # pi.load(img)
        pi.mat = img
        # m = [scale * np.cos(angle), -scale * np.sin(angle), scale * np.sin(angle), scale * np.cos(angle)]
        img = pi.calibrate(scale, scale, angle, offset, dst_shape)

        # img_t = cv2.warpPerspective(img, H, dsize=(h, w))
        return img

    @staticmethod
    def transform_img(img, offset, scale=1, angle=0):
        '''
        transform the image
        :param img:
        :param offset: [x,y]
        :param scale:
        :param angle:
        :return: img_t
        '''
        # 对图像进行变换
        w, h = img.shape
        try:
            center = [int(h / 2), int(w / 2)]
            H = affineMatrix3D(center, shift=offset, scale=scale, rotation=angle)
            img_t = cv2.warpPerspective(img, H, dsize=(h, w))
        except:  # 大图采用vips进行图像变换
            from calibration.transform import PyvipsImage
            offset = [int(offset[0]), int(offset[1])]
            pvi = PyvipsImage()
            pvi.mat = img
            img_t = pvi.regist(offset=offset, dst_shape=(w, h))

        return img_t

    def register(self):
        # transform = {'tvec': np.array([500, 300]), 'success': 0.3698420149980571, 'angle': 0, 'scale': 1}
        # src = self.transform_img(self._src_img,offset=transform['tvec'])
        # src = self._src_img
        dst_img, src_img = self.pad_same_image(self._dst_img, self._src_img)
        result = self.calibration(dst_img, src_img)
        self.offset = [int(result['offset'][0]), int(result['offset'][1])]
        self.confi = np.round(result['confi'], 3)
        if self.debug:
            assert os.path.exists(self.output_path), "{} not exist.".format(self.output_path)
            dst, src = self.pad_same_image(dst_img, src_img)
            im_t = self.transform_img(src, offset=result['offset'])
            output = os.path.join(self.output_path, os.path.basename(self.src_path).split(".")[0] + "_registered.tif")
            if self.img_dtype != np.uint8:
                im_t = (im_t / 65535) * 255
                im_t = im_t.astype(np.uint8)
            tiff_write(output, im_t)


if __name__ == "__main__":
    pass
