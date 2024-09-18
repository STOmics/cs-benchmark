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
import json


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
    file_path = file_path.replace('.ipynb_checkpoints', '')
    files_ = list()
    for root, dirs, files in os.walk(file_path):
        if '.ipynb_checkpoints' in root: continue
        if len(files) == 0:
            continue
        for f in files:
            if '.ipynb_checkpoints' in f: continue
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
        dt_path = dt_path.replace('.ipynb_checkpoints', '')
        gt_path = gt_path.replace('.ipynb_checkpoints', '')
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

        for i in tqdm.tqdm(self._dt_list, desc='Load data {}'.format(self._method)):
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
        # print(self._object_metrics)
        models_logger.info('The statistical indicators for the entire data set are as follows:')
        return self._object_metrics.mean().to_dict()

    def dump_info(self, save_path: str):
        import time

        t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        save_path_ = os.path.join(save_path, '{}_cell_segmenatation_{}.xlsx'.format(self._method, t))
        self._object_metrics.to_excel(save_path_)
        models_logger.info('The evaluation results is stored under {}'.format(save_path_))


def main(args, para):
    # ['lt', 'stereocell', 'deepcell', 'sam', 'cellpose'] 
    # ['HE', 'FB', 'ssDNA', 'mIF']
    stain = os.listdir(args.gt_path)
    # stain = ['HE']
    methods = os.listdir(args.dt_path)
    # methods = ['stereocell']
    
#     dct_ = {}
#     for m in methods:
#         tissue_dct = {}
#         for s in stain:
#             dt_path_ = os.path.join(args.dt_path, m, s)
#             tissue = os.listdir(dt_path_)
#             for t in tissue:
#                 gt_path = os.path.join(args.gt_path, s, t)
#                 dt_path = os.path.join(dt_path_, t)
#                 cse = CellSegEval(m)
#                 v = cse.evaluation(gt_path=gt_path, dt_path=dt_path)['f1']
#                 tissue_dct['{}/{}'.format(s, t)] = round(v, 5)
#         dct_[m] = tissue_dct
#         print(tissue_dct)
    
#     print('| Species | StereoCell | DeepCell | Cellpose | SAM | LT |')
#     print('|--------: | :---------:|:--------:| :---------:|:--------:|:--------:|')
    
#     for t in dct_['stereocell'].keys():
#         print('| {} | {} | {} |  {} | {} | {} |'.format(t, dct_['stereocell'][t], dct_['deepcell'][t], dct_['cellpose'][t], dct_['sam'][t], dct_['lt'][t]))     
        
    dct = {}
    for s in stain:
        gt_path = os.path.join(args.gt_path, s)
        stain_dct = {}
        for m in methods:
            dt_path = os.path.join(args.dt_path, m, s)

            cse = CellSegEval(m)
            v = cse.evaluation(gt_path=gt_path, dt_path=dt_path)
            stain_dct[m] = v
            if os.path.exists(args.output_path):
                cse.dump_info(args.output_path)
            else:
                models_logger.warn('Output path not exists, will not dump result')
        dct[s] = stain_dct
#     with open(os.path.join(args.output_path, 'stain.json'), 'w') as fd:
#         json.dump(dct, fd, indent=2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    # stains = ("HE", "ssDNA", "FB", "mIF")
    index = ("Precision", "Recall", "F1")
    
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, layout="constrained", figsize=(16, 12))
    
    for i, s in enumerate(dct.keys()):
        row, col = [i // 2, i % 2]
        penguin_means = dct[s]
        # penguin_means = {
        #     'LT': (18.35, 18.43, 14.98),
        #     'SAM': (38.79, 48.83, 47.50),
        #     'Deepcell': (189.95, 195.82, 217.19),
        #     'Cellpose': (189.95, 195.82, 217.19),
        #     'StereoCell': (189.95, 195.82, 217.19),
        # }

        x = np.arange(len(index))  # the label locations
        width = 0.15  # the width of the bars
        multiplier = 0

        # fig, ax = plt.subplots(layout='constrained')

        for attribute, measurement in penguin_means.items():
            offset = width * multiplier
            rects = axs[row, col].bar(x + offset, [round(val,3) for val in measurement.values()], width, label=attribute, alpha=0.62)
            axs[row, col].bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axs[row, col].set_ylabel('Evaluation Index')
        axs[row, col].set_title('Stain type - {}'.format(s))
        axs[row, col].set_xticks(x + width, index)
        axs[row, col].legend(loc='upper left', ncols=3)
        axs[row, col].set_ylim(0, 1)

    # plt.show()
    plt.savefig(os.path.join(args.output_path, 'benchmark.png'))


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
    # parser.add_argument("-m", "--method", action="store", dest="method", type=str, required=True,
    #                     help="Segmentation method.")
    parser.set_defaults(func=main)

    (para, args) = parser.parse_known_args()
    print(para, args)
    para.func(para, args)
