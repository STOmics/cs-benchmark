# 细胞分割和评估平台
在ztron使用
## 细胞分割平台使用方式

### 支持的细胞分割算法
MEDIAR cellpose cellpose3 sam v3 stardist

### 通过命令行使用
在命令行修改以下命令的参数，并输入
```
/storeData/USER/data/01.CellBin/00.user/fanjinghong/home/anaconda3/envs/benchmark/bin/python /storeData/USER/data/01.CellBin/00.user/fanjinghong/code/benchmark2/cell_seg.py -i your_inputpath -o your_outputpath -m  cellpose3 sam v3  -t ss/he -g 0
```
其中：  
-i 为图片输入路径  
-o 为mask输出路径  
-m 为需要使用的算法 (可以多个)   
-t 为图片类型（v3、stardist需要）,stardist不是he就填ss  
-g 为GPU序号

## 分割评估使用方式
确保gt文件夹下的图片名称带_mask,算法输出的mask文件夹下的图片名称带_img，且只有这一点不同

### 通过命令行使用
在命令行修改以下命令的参数，并输入
```
/storeData/USER/data/01.CellBin/00.user/fanjinghong/home/anaconda3/envs/benchmark/bin/python /storeData/USER/data/01.CellBin/00.user/fanjinghong/code/benchmark2/src/eval/cell_eval_multi.py -g gt_path -d dt_path -o result_path
```
其中：  
-g 为gt文件夹路径  
-d 为算法输出的mask文件夹路径  
-o 为结果输出路径  


# Cell Segmentation and Evaluation Platform  
Usage on ztron  

## Cell Segmentation Platform Usage  

### Supported Cell Segmentation Algorithms  
MEDIAR, Cellpose, Cellpose3, SAM, V3, Stardist  

### Usage via Command Line  
Modify the parameters in the following command and input it into the command line:  
```
/storeData/USER/data/01.CellBin/00.user/fanjinghong/home/anaconda3/envs/benchmark/bin/python /storeData/USER/data/01.CellBin/00.user/fanjinghong/code/benchmark2/cell_seg.py -i your_inputpath -o your_outputpath -m  cellpose3 sam v3  -t ss/he -g 0  
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
/storeData/USER/data/01.CellBin/00.user/fanjinghong/home/anaconda3/envs/benchmark/bin/python /storeData/USER/data/01.CellBin/00.user/fanjinghong/code/benchmark2/src/eval/cell_eval_multi.py -g gt_path -d dt_path -o result_path
```
Where:

-g is the path to the ground truth (GT) folder
-d is the path to the algorithm output mask folder
-o is the output path for the results