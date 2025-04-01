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
cellmorphology_PY = os.path.join(work_path, 'src/eval/cellmorphology/maskanalysis.py')

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

def draw_boxplot(directory, output_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
  
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.xlsx')]
    
    # initialize
    data = {}
    methods = []
    # Read each Excel file and extract metrics data
    for i, file_path in enumerate(file_paths):
        df = pd.read_excel(file_path)
        methods.append(os.path.basename(file_paths[i]).split('_')[0])
        data[os.path.basename(file_paths[i]).split('_')[0]] = df
        
    eval_indexs = [index for index in data[methods[0]].columns[1:]]  # get evaluation index
    
    eval_pd = dict([(eval_index, pd.DataFrame()) for eval_index in eval_indexs])
    for eval_index in eval_indexs:
        for method in methods:
            eval_pd[eval_index][method] = pd.DataFrame(data[method][eval_index])
    
    # Draw boxplot
    fig, axes = plt.subplots(1, len(eval_indexs), figsize=(5*len(eval_indexs), 6))
    
    for i, key in enumerate(eval_pd):
        sns.boxplot(data=eval_pd[key], ax=axes[i])
        axes[i].set_title(key + ' Comparison')
        axes[i].set_xlabel('Algorithm')
        axes[i].set_ylabel(key)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'benchmark-boxplot.png'))

def search_files(file_path, exts):
    file_path = file_path.replace('.ipynb_checkpoints', '')
    files_ = list()
    for root, dirs, files in os.walk(file_path):
        if '.ipynb_checkpoints' in root: 
            continue
        if len(files) == 0:
            continue
        for f in files:
            if '.ipynb_checkpoints' in f: 
                continue
            fn, ext = os.path.splitext(f)
            if ext in exts: 
                files_.append(os.path.join(root, f))
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
        # 使用 tifffile 读取第一帧，避免 deepcell 返回 (1,512,512,1) 的 shape
        arr = tifffile.imread(image_path, key=0)
        h, w = arr.shape
        arr_[:h, :w] = arr
        arr_ = label(arr_, connectivity=2)
        return arr_

    def evaluation(self, gt_path: str, dt_path: str, cutoff: float = 0.55):
        dt_path = dt_path.replace('.ipynb_checkpoints', '')
        gt_path = gt_path.replace('.ipynb_checkpoints', '')
        for i in [gt_path, dt_path]:
            assert os.path.exists(i), '{} is not exists'.format(i)
        
        print(f'gt:{gt_path}\ndt:{dt_path}')
        if os.path.isfile(gt_path):
            self._gt_list = [gt_path]
        else:
            img_lst = search_files(gt_path, ['.tif', '.png', '.jpg'])
            self._gt_list = [i for i in img_lst if 'mask' in i]
        if os.path.isfile(dt_path):
            self._dt_list = [dt_path]
        else:
            self._dt_list = search_files(dt_path, ['.tif', '.png', '.jpg'])
        self._gt_list = [imgpath for imgpath in self._dt_list if imgpath.replace('mask', 'img').replace(gt_path, dt_path) in self._dt_list]  # 只读取 DT 中有的 GT 对应的图片
        assert len(self._gt_list) == len(self._dt_list), 'Length of list GT {} are not equal to DT {}'.format(len(self._gt_list), len(self._dt_list))

        gt_arr = list()
        dt_arr = list()
        shape_list = list()
        for i in self._dt_list:
            dt = tifffile.imread(i, key=0)
            shape_list.append(dt.shape)
        w = np.max(np.array(shape_list)[:, 1])
        h = np.max(np.array(shape_list)[:, 0])
        self._suitable_shape = (h, w)
        models_logger.info('Uniform size {} into {}'.format(list(set(shape_list)), self._suitable_shape))
        for i in tqdm.tqdm(self._dt_list, desc='Load data {}'.format(self._method)):
            gt = self._load_image(image_path=i.replace('img', 'mask').replace(dt_path, gt_path))
            dt = self._load_image(image_path=i)
            assert gt.shape == dt.shape, 'Shape of GT are not equal to DT'
            gt_arr.append(gt)
            dt_arr.append(dt)
        gt_arr = np.array(gt_arr)
        dt_arr = np.array(dt_arr)
        # 使用传入的 cutoff 参数
        pm = Metrics(self._method, cutoff1=cutoff)
        models_logger.info('Start evaluating the test set, which will take some time.')
        object_metrics = pm.calc_object_stats(gt_arr, dt_arr)
        self._object_metrics = object_metrics.drop(
            labels=['jaccard','missed_det_from_merge', 'gained_det_from_split', 
                    'true_det_in_catastrophe', 'pred_det_in_catastrophe', 'merge', 'split', 
                    'catastrophe', 'seg', 'n_pred', 'n_true', 'correct_detections', 'missed_detections'], 
            axis=1)
        self._object_metrics.index = [os.path.basename(d) for d in self._dt_list]
        models_logger.info('For each piece of data in the test set, the evaluation results are as follows:')
        pd.set_option('expand_frame_repr', False)
        models_logger.info('The statistical indicators for the entire data set are as follows:')
        return self._object_metrics.mean().to_dict()

    def dump_info(self, save_path: str):
        import time
        t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        save_path_ = os.path.join(save_path, '{}_cell_segmenatation_{}.xlsx'.format(self._method, t))
        self._object_metrics.to_excel(save_path_)
        models_logger.info('The evaluation results is stored under {}'.format(save_path_))

def main(args, para):
    from decimal import Decimal
    # 示例中方法列表可以为：['lt', 'stereocell', 'deepcell', 'sam', 'cellpose'] 
    # 数据集名称例如：['HE', 'FB', 'ssDNA', 'mIF']
    dataset_name = os.path.basename(os.path.dirname(args.gt_path))
    print(f'dataset_name:{dataset_name}')
    
    visible_folders = [folder for folder in os.listdir(args.dt_path) if not folder.startswith('.')]
    methods = visible_folders
    print(f'methods:{methods}')
    
    # 判断是否启用多阈值评估
    if args.multi_threshold:
        thresholds = [0.2, 0.4, 0.5 ,0.55, 0.6, 0.8]
    else:
        thresholds = [0.55]
    
    gt_path = os.path.join(args.gt_path)
    
    # 对每个阈值进行循环评估
    for cutoff in thresholds:
        print(f"\nEvaluating with IoU threshold: {round(1-cutoff,2)}")
        # 如果多阈值评估，则在输出路径下创建子文件夹，例如 eval@0.2
        if args.multi_threshold:
            out_dir = os.path.join(args.output_path, f"eval@{round(1-cutoff,2)}")
        else:
            out_dir = args.output_path
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        dataset_dct = {}
        for m in methods:
            dt_path = os.path.join(args.dt_path, m)
            cse = CellSegEval(m)
            v = cse.evaluation(gt_path=gt_path, dt_path=dt_path, cutoff=cutoff)
            dataset_dct[m] = v
            if os.path.exists(out_dir):
                cse.dump_info(out_dir)
            else:
                models_logger.warn('Output path not exists, will not dump result')
        
        # 绘制柱状图
        print(dataset_dct)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        index = ('Precision', 'Recall', "F1",  'dice', 'PQ')
        fig, axs = plt.subplots(figsize=(16, 12))
        x = np.arange(len(index))  # 标签位置
        width = 0.1  # 每个条形图宽度
        multiplier = 0
        
        colors = {
            'cellprofiler': '#ff7f0e', 
            'MEDIAR': '#d62728',
            'cellpose': '#1f77b4', 
            'cellpose3': '#2ca02c',
            'sam': '#8c564b',
            'stardist': '#9467bd',
            'deepcell': '#17becf',
            'cellbin2': '#bcbd22',
            'hovernet': '#e377c2',
            'cyto3_train_at_cellbinDB': '#7f7f7f'
        }
        order = [
            'cellprofiler', 
            'MEDIAR', 
            'cellpose', 
            'cellpose3', 
            'sam', 
            'stardist', 
            'deepcell', 
            'cellbin2', 
            'hovernet', 
            'cyto3_train_at_cellbinDB'
        ]
        # 对结果按照指定顺序排序
        order_means = OrderedDict((key, dataset_dct[key]) for key in order if key in dataset_dct)
        for key in dataset_dct:
            if 'gained_detections' in dataset_dct[key]:
                del dataset_dct[key]['gained_detections']
        for attribute, measurement in order_means.items():
            offset = width * multiplier
            rects = axs.bar(x + offset, [round(val, 2) for val in measurement.values()], width, label=attribute, color=colors.get(attribute, None), alpha=0.62)
            axs.bar_label(rects, padding=3)
            multiplier += 1
        
        axs.set_ylabel('Evaluation Index')
        axs.set_title(f'dataset - {dataset_name} (IoU threshold={round(1-cutoff,2)})')
        axs.set_xticks(x + width, index)
        axs.legend(loc='upper left', ncols=3)
        axs.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{dataset_name}_benchmark.png'))
        
        # 绘制箱线图
        try:
            draw_boxplot(out_dir, out_dir)
        except Exception as e:
            print("no module named seaborn or error in boxplot:", e)

usage = """ Evaluate cell segmentation """
PROG_VERSION = 'v0.0.1'

"""
示例：
python cell_eval_multi.py --gt_path /home/share/gt --dt_path /home/share/dt --output_path /home/share/output --multi_threshold
如果不需要多阈值评估，则不传入 --multi_threshold 参数（默认使用阈值 0.55）
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
    parser.add_argument("--multi_threshold", action="store_true", 
                        help="开启多阈值评估功能，依次使用0.2、0.6、0.8进行评估并分别保存结果。")
    parser.set_defaults(func=main)

    (para, args) = parser.parse_known_args()
    print(para, args)
    para.func(para, args)
    # 对 cellmorphology 的调用，这里保持不变
    cellmorphology(para.gt_path, para.output_path)
