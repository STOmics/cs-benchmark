# Tutorial

## methods

We have deployed the following 7 cell segmentation methods:

> [Cellpose](https://github.com/MouseLand/cellpose) is a generalist algorithm for cellular segmentation.<br>
> [DeepCell](https://github.com/vanvalenlab/deepcell-tf) is a deep learning library for single-cell analysis of biological images. Here, pre-trained DeepCell models are used for cell/nuclei segmentation from raw image data.<br><br>
> [StereoCell](https://github.com/STOmics/StereoCell/tree/dev) is an open-source software for measuring and analyzing cell images. Here, CellProfiler is used for object detection and region growth-based object segmentation.<br><br>
> [SAM](https://github.com/facebookresearch/segment-anything) is an open-source software for measuring and analyzing cell images. Here, CellProfiler is used for object detection and region growth-based object<br><br>
> [MEDIAR]()  <br><br>
> [stardist]()  <br><br>
> [cellprofiler]()  <br><br>

## Data

Here, we present the StereoCell cell segmentation test dataset to compare the performance of different segmentation methods. Recent studies have shown that the diversity of data modalities, complex differences in image backgrounds, and cell distribution and morphology pose great challenges to segmentation methods. Therefore, we chose imaging data under [stereo-seq]() technology to construct a test set, covering 4 staining methods, namely: ssDNA, [HE](), [FB]() and [mIF](); all 42 ROIs in the test set come from 11 animal sections and 1 plant tissue sample. The test dataset is available at https://datasets.deepcell.org/ for noncommercial use.

<div align="center">
    <img src="docs/slice.png" width=60% height=60% alt="Fig StereoCell benchmarking" />
    <h6>
      Fig 1 Benchmarking for stereo-seq Image Data
    </h6>
</div>

# Benchmarking
## Index
<div align="center">
    <img src="docs/seg.png" width=50% height=50% alt="Single-cell Stereo-seq reveals induced progenitor cells involved in axolotl brain regeneration" />
    <h6>
      Fig 2 precision and recall for cell segmentation
    </h6>
</div>

To evaluate the relative performance of different deep learning architectures, we compared several alternatives: StereoCell (kernel), Deepcell (whole-cell), Cellpose (whole-cell), SAM (whole-cell), and LT. All methods are evaluated on the StereoCell test set.
 - precision,
 - recall,
 - F1,

## Cell Segmentation Platform Usage  
```
git clone https://github.com/STOmics/cs-benchmark.git  
cd cs-benchmark 
```
### Supported Cell Segmentation Algorithms  
MEDIAR, Cellpose, Cellpose3, SAM,  Stardist, Deepcell
### Environment configuration
Use the following command to install the environment for the methods you need, and add the path in the cellsegmentation_benchmark.ipynb **\_py_**  
```
conda env create -f src/methods/method_name/enviroment.yaml
```
### Usage via Command Line  
Modify the parameters in the following command and input it into the command line:  
```
python cell_seg.py -i your_inputpath -o your_outputpath -m  cellpose3 sam v3  -t ss/he -g 0  
```
Where:

-i is the input image path  
-o is the output mask path  
-m is the algorithm(s) to be used (can specify multiple)  
-t is the image type (use "ss" if not HE)  
-g is the GPU index (num or False)  
### How to use cellprofiler

## Segmentation Evaluation Usage
Ensure that the images in the gt folder have filenames with "_mask" and the images in the algorithm output mask folder have filenames with "_img", with only this difference in their names.
### Environment configuration
pip install tqdm  
pip install numpy  
pip install tifffile  
pip install opencv-python  
pip install scikit-image  
pip install pandas  
pip install scikit-learn  
pip install matplotlib  
pip install seaborn  
pip install cython six openpyxl  
cd src/eval  
python setup.py install --user  



### Usage via Command Line

Modify the parameters in the following command and input it into the command line:
```
python src/eval/cell_eval_multi.py -g gt_path -d dt_path -o result_path
```
Where:

-g is the path to the ground truth (GT) folder  
-d is the path to the algorithm output mask folder  
-o is the output path for the results  

## Acknowledgements

We thank: 

- [Cellpose_Cell_Segmentation_Tutorial](https://cloud.stomics.tech/#/public/tool/app-detail/notebook/224/--)
- [DeepCell_Cell_Segmentation](https://cloud.stomics.tech/#/public/tool/app-detail/notebook/233/--)
- [StereoCell_Cell_Segmentation](https://cloud.stomics.tech/#/public/tool/app-detail/notebook/222/--)
- [SAM_Cell_Segmentation](https://cloud.stomics.tech/#/public/tool/app-detail/notebook/206/--)