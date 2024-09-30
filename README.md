# benchmark（readme is under reconstruction）

## Introduction

This project is a benchmark for general cell segmentation models. We have deployed the following 7 cell segmentation methods: MEDIAR, Cellpose, Cellpose3, SAM, Stardist, Deepcell, Cellprofiler, along with code to evaluate the model segmentation performance. Through the command line or notebook, you can run 7 segmentation models or evaluate the performance of 7 models with one click.



## Installation


```
git clone https://github.com/STOmics/cs-benchmark.git  
cd cs-benchmark 
```
- Create an environment for Cellpose, SAM, StarDist, and DeepCell.
```
# python3.8 in conda env
conda create --name=cs-benchmark python=3.8
conda activate cs-benchmark
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

- Use the following command to install the environment for **mediar** and **cellprofiler**, and add the path in the cellsegmentation_benchmark.ipynb or cell_seg.py  **\_py_**  
```
conda env create -f src/methods/MEDIAR/environment.yaml
conda env create -f src/methods/cellprofiler/environment.yaml
```
- Download the necessary model file

    [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)  
[MEDIAR](https://drive.google.com/drive/folders/1eZLGuQkxF5ouBgTA2UuH0beLcm635ADS)  
After downloading the model file, place it in the **src/methods/models** directory of the project.

## Tutorials
### Data
[Data download link](https://bgipan.genomics.cn/#/link/v2dKKUZf8M3YFpGWvB5g)  
Contact fanjinghong@genomics.cn or shican@genomics.cn to obtain the password.  
![figure1](docs/figure1.png)
### Use via command line
#### Cell segmentation
- Modify the parameters in the following command and input it into the command line:  
```
python cell_seg.py -i your_inputpath -o your_outputpath -m  cellpose3 sam -t ss -g 0  
```
- Where:

- -i is the input image path  
-o is the output mask path  
-m is the algorithm(s) to be used (can specify multiple)  
-t is the image type (ss/he/dapi/mif)  
-g is the GPU index (num)  
#### Segmentation evaluation
- Ensure that the images in the gt folder have filenames with "**_mask**" and the images in the algorithm output mask folder have filenames with "**_img**", with only this difference in their names.   
  
- Modify the parameters in the following command and input it into the command line:
```
python src/eval/cell_eval_multi.py -g gt_path -d dt_path -o result_path
```
- Where:

- -g is the path to the ground truth (GT) folder  
-d is the path to the algorithm output mask folder  
-o is the output path for the results  

### Use via Notebook
[cellsegmentation_benchmark.ipynb](https://github.com/STOmics/cs-benchmark/blob/main/tutorial/cellsegmentation_benchmark.ipynb)

## License and Citation


## Reference
> [cellpose](https://github.com/MouseLand/cellpose)  
> [cellpose3](https://github.com/MouseLand/cellpose)  
> [deepcell](https://github.com/vanvalenlab/deepcell-tf)   
> [sam](https://github.com/facebookresearch/segment-anything)   
> [mediar](https://github.com/Lee-Gihun/MEDIAR)   
> [stardist](https://github.com/stardist/stardist)   
> [cellprofiler](https://github.com/CellProfiler)   