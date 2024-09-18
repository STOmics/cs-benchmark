import sys

sys.path.append("/storeData/USER/data/01.CellBin/00.user/cenweixuan/cellseg3rd/cellpose")
import copy
import cv2
import os
from cellpose import models
import numpy as np
from math import ceil
import patchify
import tqdm
import tifffile
import glob
import traceback
import time
from skimage.morphology import remove_small_objects


def f_ij_16_to_8(img, chunk_size=1000):
    """
    16 bits img to 8 bits

    :param img: (CHANGE) np.array
    :param chunk_size: chunk size (bit)
    :return: np.array
    """

    if img.dtype == 'uint8':
        return img
    dst = np.zeros(img.shape, np.uint8)
    p_max = np.max(img)
    p_min = np.min(img)
    scale = 256.0 / (p_max - p_min + 1)
    for idx in range(img.shape[0] // chunk_size + 1):
        sl = slice(idx * chunk_size, (idx + 1) * chunk_size)
        win_img = copy.deepcopy(img[sl])
        win_img = np.int16(win_img)
        win_img = (win_img & 0xffff)
        win_img = win_img - p_min
        win_img[win_img < 0] = 0
        win_img = win_img * scale + 0.5
        win_img[win_img > 255] = 255
        dst[sl] = np.array(win_img).astype(np.uint8)
    return dst


def f_rgb2gray(img, need_not=False):
    """
    rgb2gray

    :param img: (CHANGE) np.array
    :param need_not: if need bitwise_not
    :return: np.array
    """
    if img.ndim == 3:
        if img.shape[0] == 3 and img.shape[1] > 3 and img.shape[2] > 3:
            img = img.transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if need_not:
            img = cv2.bitwise_not(img)
    return img


def f_instance2semantics(ins):
    h, w = ins.shape[:2]
    tmp0 = ins[1:, 1:] - ins[:h - 1, :w - 1]
    ind0 = np.where(tmp0 != 0)

    tmp1 = ins[1:, :w - 1] - ins[:h - 1, 1:]
    ind1 = np.where(tmp1 != 0)
    ins[ind1] = 0
    ins[ind0] = 0
    ins[np.where(ins > 0)] = 1
    return np.array(ins, dtype=np.uint8)


def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    '''
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape
    view_shape = (1 + (m1 - m2) // stride[0], 1 + (n1 - n2) // stride[1], m2, n2) + arr.shape[2:]
    strides = (stride[0] * s0, stride[1] * s1, s0, s1) + arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides)
    return subs


def poolingOverlap(mat, ksize, stride=None, method='max', pad=False):
    '''Overlapping pooling on 2D or 3D data.
    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).
    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky, kx = ksize
    if stride is None:
        stride = (ky, kx)
    sy, sx = stride

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    mat = np.where(mat == 0, np.nan, mat)

    if pad:
        ny = _ceil(m, sy)
        nx = _ceil(n, sx)
        size = ((ny - 1) * sy + ky, (nx - 1) * sx + kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[:(m - ky) // sy * sy + ky, :(n - kx) // sx * sx + kx, ...]

    view = asStride(mat_pad, ksize, stride)
    if method == 'max':
        result = np.nanmax(view, axis=(2, 3))
    else:
        result = np.nanmean(view, axis=(2, 3))
    result = np.nan_to_num(result)
    return result


def f_instance2semantics_max(ins):
    ins_m = poolingOverlap(ins, ksize=(2, 2), stride=(1, 1), pad=True, method='mean')
    mask = np.uint8(np.subtract(np.float64(ins), ins_m))
    ins[mask != 0] = 0
    ins = f_instance2semantics(ins)
    return ins


def cellseg(file_lst, save_path, photo_size, photo_step, model_name='cyto'):
    '''
    open_path 待处理图像存储为文件夹路径
    save_path 保存路径
    photo_size 剪裁成小图时，小图的大小
    photo_step  剪裁图像时的步长
    dmin 设置细胞直径的最小值
    dmax 设置细胞直径的最大值
    step 设置遍历细胞直径的步长
    '''
    overlap = photo_size - photo_step
    if (overlap % 2) == 1:
        overlap = overlap + 1
    act_step = ceil(overlap / 2)

    # file_lst = glob.glob(os.path.join(open_path, "*.tif"))
    model = models.Cellpose(gpu=True, model_type=model_name)
    for file in file_lst:
        try:
            name = os.path.split(file)[-1]
            if os.path.exists(os.path.join(save_path, name)):
                continue

            print(file)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            img = tifffile.imread(file,key=0)
            img = f_ij_16_to_8(img, chunk_size=1000000)
            img = f_rgb2gray(img, True)

            res_image = np.pad(img, ((act_step, act_step), (act_step, act_step)), 'constant')
            res_a = res_image.shape[0]
            res_b = res_image.shape[1]
            re_length = ceil((res_a - (photo_size - photo_step)) / photo_step) * photo_step + (
                    photo_size - photo_step)
            re_width = ceil((res_b - (photo_size - photo_step)) / photo_step) * photo_step + (
                    photo_size - photo_step)
            regray_image = np.pad(res_image, ((0, re_length - res_a), (0, re_width - res_b)), 'constant')
            patches = patchify.patchify(regray_image, (photo_size, photo_size), step=photo_step)
            wid = patches.shape[0]
            high = patches.shape[1]
            a_patches = np.full((wid, high, (photo_size - overlap), (photo_size - overlap)), 255, dtype=np.uint8)

            for i in tqdm.tqdm(range(wid)):
                for j in range(high):
                    img_data = patches[i, j, :, :]
                    masks, flows, styles, diams = model.eval(img_data, diameter=None, channels=[0, 0])
                    masks = f_instance2semantics_max(masks)
                    a_patches[i, j, :, :] = masks[act_step:(photo_size - act_step),
                                            act_step:(photo_size - act_step)]

            patch_nor = patchify.unpatchify(a_patches,
                                            ((wid) * (photo_size - overlap), (high) * (photo_size - overlap)))
            nor_imgdata = np.array(patch_nor)
            after_wid = patch_nor.shape[0]
            after_high = patch_nor.shape[1]
            cropped_1 = nor_imgdata[0:(after_wid - (re_length - res_a)), 0:(after_high - (re_width - res_b))]
            cropped_1 = np.uint8(remove_small_objects(cropped_1 > 0, min_size=2))
            tifffile.imwrite(os.path.join(save_path, name), cropped_1, compression='zlib')
            #tifffile.imwrite(save_path, cropped_1, compression='zlib')
        except:
            traceback.print_exc()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="you should add those parameter")
    parser.add_argument('-i', "--input", help="the input img path")
    parser.add_argument('-o', "--output", help="the output file")
    parser.add_argument("-m", "--model_name", help="model name", default="cyto")
    parser.add_argument("-g", "--gpu", help="the gpu index", default="-1")

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    model_name = args.model_name
    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    file_lst = []
    if os.path.isdir(input_path):
        file_lst = glob.glob(os.path.join(input_path, "*.tif"))
    else:
        file_lst = [input_path]

    cellseg(file_lst, output_path, 2048, 2000, model_name=model_name)
    sys.exit()
