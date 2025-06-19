# RUN CELLPOSE4
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm, trange
import tifffile
from cellpose import models, io, core
from utils import cell_dataset, instance2semantics

def cellpose4_method(para, args):
    input_path = para.image_path
    output_path = para.output
    use_gpu = para.is_gpu
    img_type = para.img_type

    # 检查GPU是否可用
    if use_gpu and not core.use_gpu():
        raise RuntimeError("GPU 不可用，请检查CUDA环境。")

    # 加载模型
    model = models.CellposeModel(gpu=use_gpu)

    # 获取输入图像列表
    if os.path.isdir(input_path):
        files = cell_dataset(input_path, ['.tif', '.jpg', '.png'])
    else:
        files = [input_path]

    # 参数设置
    flow_threshold = 0.4
    cellprob_threshold = 0.0
    tile_norm_blocksize = 0
    chan = [0, 1, 2]  # 可按需修改，使用哪些通道

    os.makedirs(output_path, exist_ok=True)

    print(f"正在处理 {len(files)} 张图像")
    for f in tqdm(files, desc='Cellpose4'):
        img = io.imread(f)

        # 确保图像是 (H, W, C) 形式
        if img.ndim == 2:  # 灰度图
            img = np.expand_dims(img, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            pass  # 单通道灰度图
        elif img.ndim == 3 and img.shape[2] == 3:
            pass  # RGB 图像
        else:
            raise ValueError(f"不支持的图像形状: {img.shape}")

        # 自适应设置通道
        if img.shape[2] == 1:
            chan = [0, 0]  # 用一个灰度通道做核+细胞
        elif img.shape[2] == 3:
            chan = [0, 2]  # R 通道为核，B 为细胞
        else:
            raise ValueError(f"图像通道数不支持: {img.shape[2]}")

        # 提取所需通道
        img_selected = img[:, :, [chan[0]]] if chan[0] == chan[1] else img[:, :, chan]


        # 推理
        masks, flows, styles = model.eval(
            img_selected,
            batch_size=32,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            normalize={"tile_norm_blocksize": tile_norm_blocksize}
        )

        # 转语义分割图
        semantics = instance2semantics(masks)
        semantics[semantics > 0] = 255

        # 保存掩膜图像
        name = os.path.basename(f)
        #tifffile.imwrite(os.path.join(output_path, name), masks.astype(np.uint8), compression='zlib')
        tifffile.imwrite(os.path.join(output_path, name), semantics.astype(np.uint8), compression='zlib')

USAGE = 'Cellpose4'
PROG_VERSION = 'v0.0.1'

def main():
    arg_parser = argparse.ArgumentParser(usage=USAGE)
    arg_parser.add_argument("--version", action="version", version=PROG_VERSION)
    arg_parser.add_argument("-o", "--output", action="store", dest="output",
                            type=str, default=None, help="保存分割结果的路径")
    arg_parser.add_argument("-i", "--image_path", action="store", dest="image_path",
                            type=str, default=None, help="输入图像文件或文件夹")
    arg_parser.add_argument("-g", "--is_gpu", action="store", dest="is_gpu",
                            type=bool, default=True, help="是否使用GPU")
    arg_parser.add_argument("-t", "--img_type", help="图像类型，例如he、ss")
    arg_parser.set_defaults(func=cellpose4_method)
    (para, args) = arg_parser.parse_known_args()
    para.func(para, args)

if __name__ == '__main__':
    import sys
    return_code = main()
    sys.exit(return_code)
