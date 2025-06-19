# RUN CELLSAM
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tifffile
import imageio.v3 as iio
from skimage.color import gray2rgb

from cellSAM.cellsam_pipeline import cellsam_pipeline
from utils import cell_dataset  # 建议你在 utils.py 中复用 cell_dataset 函数

def cellsam_method(para, args):
    input_path = para.image_path
    output_path = para.output
    use_gpu = para.is_gpu
    img_type = para.img_type
    model_path = para.model_path

    os.makedirs(output_path, exist_ok=True)

    # 获取图像列表
    if os.path.isdir(input_path):
        files = cell_dataset(input_path, ['.tif', '.jpg', '.png'])
    else:
        files = [input_path]

    print(f"正在处理 {len(files)} 张图像")
    for f in tqdm(files, desc='CellSAM'):
        img = iio.imread(f)

        # 转为 RGB 格式以确保兼容
        if img.ndim == 2:
            img = gray2rgb(img)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 3:
            pass  # 已经是 RGB
        else:
            raise ValueError(f"不支持的图像维度: {img.shape}")

        # 执行 CellSAM 推理
        mask = cellsam_pipeline(
            img,
            model_path=model_path,
            use_wsi=False,
            gauge_cell_size=True,
            iou_threshold=0.5,
        )

        # 保存掩膜图
        name = os.path.basename(f)
        save_path = os.path.join(output_path, name)
        tifffile.imwrite(save_path, mask.astype(np.uint16), compression='zlib')

USAGE = 'CellSAM'
PROG_VERSION = 'v0.0.1'

def main():
    arg_parser = argparse.ArgumentParser(usage=USAGE)
    arg_parser.add_argument("--version", action="version", version=PROG_VERSION)
    arg_parser.add_argument("-o", "--output", type=str, default=None, help="保存分割结果的路径")
    arg_parser.add_argument("-i", "--image_path", type=str, default=None, help="输入图像文件或文件夹")
    arg_parser.add_argument("-g", "--is_gpu", type=bool, default=True, help="是否使用GPU（未启用，保留接口）")
    arg_parser.add_argument("-t", "--img_type", type=str, help="图像类型，例如 he、ss")
    arg_parser.add_argument("-m", "--model_path", type=str, default=None, help="可选：CellSAM 自定义模型路径")
    arg_parser.set_defaults(func=cellsam_method)
    (para, args) = arg_parser.parse_known_args()
    para.func(para, args)

if __name__ == '__main__':
    import sys
    return_code = main()
    sys.exit(return_code)
