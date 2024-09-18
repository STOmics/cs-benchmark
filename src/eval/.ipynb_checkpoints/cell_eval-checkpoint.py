# Copyright 2016-2022 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for metrics.py accuracy statistics"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import tqdm
import os


import logging
models_logger = logging.getLogger(__name__)

import numpy as np
import tifffile
import cv2 as cv
from skimage.measure import label

from metrics import Metrics
import argparse
import pandas as pd


def search_files(file_path, exts):
    files_ = list()
    for root, dirs, files in os.walk(file_path):
        if len(files) == 0:
            continue
        for f in files:
            fn, ext = os.path.splitext(f)
            if ext in exts: files_.append(os.path.join(root, f))

    return files_

class CellSegEval(object):
    def __init__(self, method: str = None):
        self._method = method
        self._gt_list = list()
        self._dt_list = list()
        self._object_metrics = None
        self._suitable_shape = None

    def set_method(self, method: str):
        self._method = method

    def _load_image(self, image_path: str):
        arr_ = np.zeros(self._suitable_shape, dtype=np.uint8)
        arr = tifffile.imread(image_path)
        h, w = arr.shape
        arr_[:h, :w] = arr
        arr_ = label(arr_, connectivity=2)
        return arr_

    def evaluation(self, gt_path: str, dt_path: str):
        for i in [gt_path, dt_path]:
            assert os.path.exists(i), '{} is not exists'.format(i)

        if os.path.isfile(gt_path):
            self._gt_list = [gt_path]
        else:
            img_lst = search_files(gt_path, ['.tif'])
            self._gt_list = [i for i in img_lst if 'mask' in i]
        if os.path.isfile(dt_path):
            self._dt_list = [dt_path]
        else:
            self._dt_list = search_files(dt_path, ['.tif'])

        assert len(self._gt_list) == len(self._dt_list), 'Length of list GT {} are not equal to DT {}'.format(len(self._gt_list), len(self._dt_list))

        gt_arr = list()
        dt_arr = list()
        shape_list = list()
        for i in self._dt_list:
            dt = tifffile.imread(i)
            shape_list.append(dt.shape)
        w = np.max(np.array(shape_list)[:, 1])
        h = np.max(np.array(shape_list)[:, 0])
        self._suitable_shape = (h, w)
        models_logger.info('Uniform size {} into {}'.format(list(set(shape_list)), self._suitable_shape))

        for i in tqdm.tqdm(self._dt_list, desc='Load data'):
            tag = os.path.basename(i)
            gt = self._load_image(image_path=i.replace('img', 'mask').replace(dt_path, gt_path))
            dt = self._load_image(image_path=i)
            assert gt.shape == dt.shape, 'Shape of GT are not equal to DT'
            gt_arr.append(gt)
            dt_arr.append(dt)
        gt_arr = np.array(gt_arr)
        dt_arr = np.array(dt_arr)
        pm = Metrics(self._method)
        models_logger.info('Start evaluating the test set, which will take some time.')
        object_metrics = pm.calc_object_stats(gt_arr, dt_arr)
        self._object_metrics = object_metrics.drop(
            labels=['gained_detections', 'missed_det_from_merge', 'gained_det_from_split', 'true_det_in_catastrophe', 'pred_det_in_catastrophe', 'merge', 'split', 'catastrophe', 'seg', 'jaccard', 'dice', 'n_pred', 'n_true', 'correct_detections', 'missed_detections'], axis=1)
        self._object_metrics.index = [os.path.basename(d) for d in self._dt_list]
        models_logger.info('For each piece of data in the test set, the evaluation results are as follows:')
        pd.set_option('expand_frame_repr', False)
        print(self._object_metrics)
        models_logger.info('The statistical indicators for the entire data set are as follows:')
        print(self._object_metrics.mean())

    def dump_info(self, save_path: str):
        import time

        t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        save_path_ = os.path.join(save_path, '{}_cell_segmenatation_{}.xlsx'.format(self._method, t))
        self._object_metrics.to_excel(save_path_)
        models_logger.info('The evaluation results is stored under {}'.format(save_path_))


def main(args, para):
    cse = CellSegEval(args.method)
    cse.evaluation(gt_path=args.gt_path, dt_path=args.dt_path)
    if os.path.exists(args.output_path):
        cse.dump_info(args.output_path)
    else:
        models_logger.warn('Output path not exists, will not dump result')


usage = """ Evaluate cell segmentation """
PROG_VERSION = 'v0.0.1'

"""
python cell_eval.py --gt_path /home/share/huada/home/liuhuanlin/self/data/gt --dt_path /home/share/huada/home/liuhuanlin/self/data/dt --output_path /home/share/huada/home/liuhuanlin/self/data/output --method SAM

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("--version", action="version", version=PROG_VERSION)
    parser.add_argument("-g", "--gt_path", action="store", dest="gt_path", type=str, required=True,
                        help="Input GT path.")
    parser.add_argument("-d", "--dt_path", action="store", dest="dt_path", type=str, required=True,
                        help="Input DT path.")
    parser.add_argument("-o", "--output_path", action="store", dest="output_path", type=str, required=True,
                        help="Output result path.")
    parser.add_argument("-m", "--method", action="store", dest="method", type=str, required=True,
                        help="Segmentation method.")
    parser.set_defaults(func=main)

    (para, args) = parser.parse_known_args()
    print(para, args)
    para.func(para, args)


# methods = ['lt', 'stereocell', 'deepcell', 'sam', 'cellpose'] 
# stain = ['HE', 'FB', 'ssDNA', 'mIF']
# for s in stain:
#     for m in methods:
#         gt_path = os.path.join(image_path, s)
#         dt_path = os.path.join(output_path, m, s)
#         cmd = '{} {} -g {} -d {} -o {} -m {}'.format(py, script, 
#                                      gt_path, dt_path, eval_path, '{}_{}'.format(m, s))
#         os.system(cmd)
