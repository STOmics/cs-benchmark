# benchmark（readme is under reconstruction）

## Introduction

We have deployed the following 7 cell segmentation methods:

## Installation
Supported Cell Segmentation Algorithms: MEDIAR, Cellpose, Cellpose3, SAM,  Stardist, Deepcell

```
git clone https://github.com/STOmics/cs-benchmark.git  
cd cs-benchmark 
```
```
# python3.8 in conda env
conda create --name=cs-benchmark python=3.8
conda activate cs-benchmark
```
#### Environment configuration
Use the following command to install the environment for the methods you need, and add the path in the cellsegmentation_benchmark.ipynb or cell_seg.py  **\_py_**  
```
conda env create -f src/methods/method_name/enviroment.yaml
```
#### Download the necessary model file

[sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)  
[MEDIAR](https://drive.google.com/drive/folders/1eZLGuQkxF5ouBgTA2UuH0beLcm635ADS)  
After downloading the model file, place it in the **src/methods/models** directory of the project.

## Getting Started
Modify the parameters in the following command and input it into the command line:  
```
python cell_seg.py -i your_inputpath -o your_outputpath -m  cellpose3 sam -t ss/he -g 0  
```
Where:

-i is the input image path  
-o is the output mask path  
-m is the algorithm(s) to be used (can specify multiple)  
-t is the image type (use "ss" if not HE)  
-g is the GPU index (num or False)  
## How to use cellprofiler

## Segmentation Evaluation Usage
Ensure that the images in the gt folder have filenames with "_mask" and the images in the algorithm output mask folder have filenames with "_img", with only this difference in their names.
### Environment configuration
pip install tqdm numpy tifffile opencv-python scikit-image pandas scikit-learn matplotlib seaborn cython six openpyxl  
cd src/eval  
python setup.py install --user  

### Getting Started

Modify the parameters in the following command and input it into the command line:
```
python src/eval/cell_eval_multi.py -g gt_path -d dt_path -o result_path
```
Where:

-g is the path to the ground truth (GT) folder  
-d is the path to the algorithm output mask folder  
-o is the output path for the results  

## Acknowledgements
> [cellpose](https://github.com/MouseLand/cellpose)  
> [cellpose3](https://github.com/MouseLand/cellpose)  
> [deepcell](https://github.com/vanvalenlab/deepcell-tf)   
> [sam](https://github.com/facebookresearch/segment-anything)   
> [mediar](https://github.com/Lee-Gihun/MEDIAR)   
> [stardist](https://github.com/stardist/stardist)   
> [cellprofiler](https://github.com/CellProfiler)   