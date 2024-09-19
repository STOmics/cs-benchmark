# Tutorial

## methods

We have deployed the following 7 cell segmentation methods:

> [Cellpose](https://github.com/MouseLand/cellpose) is a generalist algorithm for cell and nucleus segmentation. Cellpose can segment many types of cell without requiring parameter adjustments, new training data or further model retraining. Cellpose uses the 'cyto' model, with the channels for grayscale images set to [0,0] and for H&E images set to [1,3]. The diameter is set to None. <br><br>
> [Cellpose3](https://github.com/MouseLand/cellpose) is a generalist algorithm for cell and nucleus segmentation that can perform image restoration. Cellpose 3 uses the 'cyto3' model, with the channels for grayscale images set to [0,0] and for HE images set to [1,3]. The diameter is set to None, and the 'denoise_cyto3' model is used for noisy images. <br><br>
> [DeepCell](https://github.com/vanvalenlab/deepcell-tf) is a deep learning library for single-cell analysis of biological images. Here, pre-trained DeepCell models are used for cell/nuclei segmentation from raw image data. Deepcell uses the 'Mesmer' model, with image_map set to 0.5 and compartment set to 'nuclear'. <br><br>
> [SAM](https://github.com/facebookresearch/segment-anything) is an open-source software for measuring and analyzing cell images. SAM claims that it has capability on Zero-shot generalization of new image distributions and tasks. The model_type of SAM uses 'vit_b' and utilizes SamAutomaticMaskGenerator to automatically generate masks without the need for external prompts. <br><br>
> [MEDIAR](https://github.com/Lee-Gihun/MEDIAR) is a framework for efficient cell instance segmentation of multi-modality microscopy images. MEDAIR stood out in the NeurIPS 2022 cell segmentation competition and achieved state of the art (SOTA). MEDIAR uses the provided from_phase2.pth model, with model_args configured as follows: 'classes' is set to 3, 'decoder_channels' is set to [1024, 512, 256, 128, 64], 'decoder_pab_channels' is set to 256, 'encoder_name' is set to 'mit_b5', and 'in_channels' is set to 3. The algo_params has 'use_tta' set to True. <br><br>
> [stardist](https://github.com/stardist/stardist) uses star-convex polygons to represent cell shapes, allowing accurate cell localization even under challenging conditions. StarDist uses the '2D_versatile_he' model for HE images and the '2D_demo' model for non-HE images. <br><br>
> [cellprofiler](https://github.com/CellProfiler) is a free open-source software designed to enable biologists without training in computer vision or programming to quantitatively measure phenotypes from thousands of images automatically. <br><br>

## Data

Here, we present a new dataset to compare the performance of different segmentation methods. Recent studies have shown that the diversity of data modalities, complex differences in image backgrounds, and cell distribution and morphology pose great challenges to segmentation methods. 
The dataset consists of 1,044 annotated microscopy images, with a total of 109,083 cell annotations, including four staining types: DAPI, ssDNA, H&E, and mIF. The dataset contains two species, human and mouse, and includes more than 30 types of histologically normal tissues and diseased tissue (such as skin melanoma). The images in the dataset come from two sources: unpublished experimental data and open-source platform [10×Genomics](https://www.10xgenomics.com/datasets). The test dataset is available for noncommercial use.

<div align="center">
    <img src="docs/slice.png" width=60% height=60% alt="Fig 1" />
    <h6>
      Fig 1 Image Data
    </h6>
</div>

# Benchmarking

## Cell Segmentation Platform Usage  
```
git clone https://github.com/STOmics/cs-benchmark.git  
cd cs-benchmark 
```
### Supported Cell Segmentation Algorithms  
MEDIAR, Cellpose, Cellpose3, SAM,  Stardist, Deepcell
### Environment configuration
Use the following command to install the environment for the methods you need, and add the path in the cellsegmentation_benchmark.ipynb or cell_seg.py  **\_py_**  
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