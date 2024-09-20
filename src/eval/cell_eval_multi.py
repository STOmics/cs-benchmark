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
from skimage import io
from collections import OrderedDict
from metrics import Metrics
import argparse
import pandas as pd
import subprocess
# Constants
work_path = os.path.abspath('.')
cellmorphology_PY = os.path.join(work_path,'src/eval/cellmorphology/maskanalysis.py')

def sub_run(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while p.poll() is None:
        line = p.stdout.readline()
        line = line.strip()
        line = str(line, encoding='utf-8')
        if line:
            print('Subprogram output: [{}]'.format(line))
    if p.returncode == 0:
        print('Subprogram success')
    else:
        print('Subprogram failed')
    return
def cellmorphology(input, output):
    cmd = f"python {cellmorphology_PY} -g {input} -o {output}"
    sub_run(cmd)
    return
def draw_boxplot(directory,output_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
  
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.xlsx')]
    
    # initialize
    data = {}
    methods = []
    eval_indexs = []
    # Read each Excel file and extract metrics data
    for i, file_path in enumerate(file_paths):
        df = pd.read_excel(file_path)
        methods.append(os.path.basename(file_paths[i]).split('_')[0])
        data[os.path.basename(file_paths[i]).split('_')[0]]= df
        
    eval_indexs = [index for index in data[methods[0]].columns[1:]]  # get evaluation index
    
    eval_pd = dict([(eval_index,pd.DataFrame()) for eval_index in eval_indexs])
    for i, eval_index in enumerate(eval_indexs):
        for j , method in enumerate(methods):
            eval_pd[eval_index][method] = pd.DataFrame(data[method][eval_index])
    
    # Draw boxplot
    fig, axes = plt.subplots(1, len(eval_indexs), figsize=(5*len(eval_indexs), 6))
    
    for i,key in enumerate(eval_pd):
        sns.boxplot(data=eval_pd[key], ax=axes[i])
        axes[i].set_title( key+' Comparison')
        axes[i].set_xlabel('Algorithm')
        axes[i].set_ylabel(key)

    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'benchmark-boxplot.png'))
    
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
        #arr = io.imread(image_path) 
        arr = tifffile.imread(image_path,key=0) ##读取第一帧，不然deepcell的会变成(1,512,512,1)的shape
        h, w = arr.shape
        arr_[:h, :w] = arr
        arr_ = label(arr_, connectivity=2)
        return arr_

    def evaluation(self, gt_path: str, dt_path: str):
        dt_path = dt_path.replace('.ipynb_checkpoints', '')
        gt_path = gt_path.replace('.ipynb_checkpoints', '')
        for i in [gt_path, dt_path]:
            assert os.path.exists(i), '{} is not exists'.format(i)
        
        print(f'gt:{gt_path}\ndt:{dt_path}')
        if os.path.isfile(gt_path):
            self._gt_list = [gt_path]
        else:
            img_lst = search_files(gt_path, ['.tif','.png','.jpg'])
            self._gt_list = [i for i in img_lst if 'mask' in i]
        if os.path.isfile(dt_path):
            self._dt_list = [dt_path]
        else:
            self._dt_list = search_files(dt_path, ['.tif','.png','.jpg'])
        self._gt_list = [imgpath for imgpath in self._dt_list if imgpath.replace('mask','img').replace(gt_path,dt_path) in self._dt_list] #只读取DT中有的GT对应的图片
        assert len(self._gt_list) == len(self._dt_list), 'Length of list GT {} are not equal to DT {}'.format(len(self._gt_list), len(self._dt_list))

        gt_arr = list()
        dt_arr = list()
        shape_list = list()
        for i in self._dt_list:
            dt = tifffile.imread(i,key=0) ##读取第一帧，不然deepcell的会变成(1,512,512,1)的shape
            #dt = io.imread(i)
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
        pm = Metrics(self._method, cutoff1=0.55)
        models_logger.info('Start evaluating the test set, which will take some time.')
        object_metrics = pm.calc_object_stats(gt_arr, dt_arr)
        self._object_metrics = object_metrics.drop(
            labels=['gained_detections', 'missed_det_from_merge', 'gained_det_from_split', 'true_det_in_catastrophe', 'pred_det_in_catastrophe', 'merge', 'split', 'catastrophe', 'seg', 'n_pred', 'n_true', 'correct_detections','missed_detections'], axis=1)
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
    #stain = os.listdir(args.gt_path)
    dataset_name = os.path.basename(os.path.dirname(args.gt_path))
    #stain = [img for img in os.listdir(args.gt_path) if os.path.splitext(img)[1] == '.jpg' or os.path.splitext(img)[1] == '.tif' or os.path.splitext(img)[1] == '.png' ]
    print(f'dataset_name:{dataset_name}')
    # stain = ['HE']
    visible_folders = [folder for folder in os.listdir(args.dt_path) if not folder.startswith('.')]
    methods = visible_folders
    print(f'methods:{methods}')
    
    # methods = ['stereocell']
        
    dct = {}
    gt_path = os.path.join(args.gt_path)
    dataset_dct = {}
    for m in methods:

        dt_path = os.path.join(args.dt_path, m)
        cse = CellSegEval(m)
        v = cse.evaluation(gt_path=gt_path, dt_path=dt_path)
        dataset_dct[m] = v
        if os.path.exists(args.output_path):
            cse.dump_info(args.output_path)
        else:
            models_logger.warn('Output path not exists, will not dump result')

    dct[dataset_name] = dataset_dct

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    # stains = ("HE", "ssDNA", "FB", "mIF")
    index = ('Precision', 'Recall', "F1",'jaccrd','dice')
    

    fig, axs = plt.subplots(figsize=(16, 12))

    x = np.arange(len(index))  # the label locations
    width = 0.1  # the width of the bars
    multiplier = 0
    
    # fig, ax = plt.subplots(layout='constrained')
    penguin_means = dct[dataset_name]
    print(penguin_means)
        

    # 原始字典（未排序）
    colors = {
        'cellprofiler': '#ff7f0e', 
        'MEDIAR': '#d62728',
        'cellpose': '#1f77b4', 
        'cellpose3': '#2ca02c',
        'sam': '#8c564b',
        'stardist': '#9467bd',
        'deepcell': '#17becf',
        'v3': '#bcbd22',
        'lt': '#e377c2',
        'cellpose2': '#7f7f7f'
    }

    # 需要的排序顺序
    order = [
        'cellprofiler', 
        'MEDIAR', 
        'cellpose', 
        'cellpose3', 
        'sam', 
        'stardist', 
        'deepcell', 
        'v3', 
        'lt', 
        'cellpose2'
    ]

    # 根据 order 列表排序字典
    order_means = penguin_means
    order_means = OrderedDict((key, order_means[key]) for key in order if key in order_means)
    for attribute, measurement in order_means.items():
        
        print(attribute, measurement)
        offset = width * multiplier
        rects = axs.bar(x + offset, [round(val,2) for val in measurement.values()], width, label=attribute, color=colors[attribute],alpha=0.62)
        axs.bar_label(rects, padding=3)
        multiplier += 1
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs.set_ylabel('Evaluation Index')
    axs.set_title('dataset - {}'.format(dataset_name))
    axs.set_xticks(x + width, index)
    axs.legend(loc='upper left', ncols=3)
    axs.set_ylim(0, 1)

    # plt.show()
    plt.savefig(os.path.join(args.output_path, '{}_benchmark.png'.format(dataset_name)))
    
    # box plot
    try:
        draw_boxplot(args.output_path, args.output_path)
    except:
        print("no module name seaborn")
usage = """ Evaluate cell segmentation """
PROG_VERSION = 'v0.0.1'

"""
python cell_eval_multi.py --gt_path /home/share/huada/home/liuhuanlin/self/data/gt --dt_path /home/share/huada/home/liuhuanlin/self/data/dt --output_path /home/share/huada/home/liuhuanlin/self/data/output --method SAM

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
    cellmorphology(para.gt_path,para.output_path)
