# Cell Segmentation and Evaluation Platform  
Usage on linux  

## Cell Segmentation Platform Usage  

### Environment configuration

### Supported Cell Segmentation Algorithms  
MEDIAR, Cellpose, Cellpose3, SAM,  Stardist  

### Usage via Command Line  
Modify the parameters in the following command and input it into the command line:  
```
python cell_seg.py -i your_inputpath -o your_outputpath -m  cellpose3 sam v3  -t ss/he -g 0  
```
Where:

-i is the input image path
-o is the output mask path
-m is the algorithm(s) to be used (can specify multiple)
-t is the image type (required for V3 and Stardist; for Stardist, use "ss" if not HE)
-g is the GPU index

## Segmentation Evaluation Usage
Ensure that the images in the gt folder have filenames with "_mask" and the images in the algorithm output mask folder have filenames with "_img", with only this difference in their names.

Usage via Command Line
Modify the parameters in the following command and input it into the command line:
```
python src/eval/cell_eval_multi.py -g gt_path -d dt_path -o result_path
```
Where:

-g is the path to the ground truth (GT) folder
-d is the path to the algorithm output mask folder
-o is the output path for the results

## How to use cellprofiler
