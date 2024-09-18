import sys

sys.path.append("")
import os
import copy
import traceback
import glob

import numpy as np
import cv2
import tifffile

from stardist.models import StarDist2D
from csbdeep.utils import normalize
from tqdm import tqdm

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


def f_pad(img, top, bot, left, right, mode='constant', value=0):
    """
    update by dengzhonghan on 2023/2/23
    1. support 3d array padding.
    2. not support 1d array padding.

    Args:
        img (): numpy ndarray (2D or 3D).
        top (): number of values padded to the top direction.
        bot (): number of values padded to the bottom direction.
        left (): number of values padded to the left direction.
        right (): number of values padded to the right direction.
        mode (): padding mode in numpy, default is constant.
        value (): constant value when using constant mode, default is 0.

    Returns:
        pad_img: padded image.

    """

    if mode == 'constant':
        if img.ndim == 2:
            pad_img = np.pad(img, ((top, bot), (left, right)), mode, constant_values=value)
        elif img.ndim == 3:
            pad_img = np.pad(img, ((top, bot), (left, right), (0, 0)), mode, constant_values=value)
    else:
        if img.ndim == 2:
            pad_img = np.pad(img, ((top, bot), (left, right)), mode)
        elif img.ndim == 3:
            pad_img = np.pad(img, ((top, bot), (left, right), (0, 0)), mode)
    return pad_img


def f_padding(img, shape, mode='constant'):
    h, w = img.shape[:2]
    win_h, win_w = shape[:2]
    img = f_pad(img, 0, abs(win_h - h), 0, abs(win_w - w), mode)
    return img


def f_fusion(img1, img2):
    img1 = cv2.bitwise_or(img1, img2)
    return img1


def f_instance2semantics(ins):
    """
    instance to semantics
    Args:
        ins(ndarray):labeled instance

    Returns(ndarray):mask
    """
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


def pre(img):
    axis_norm = (0, 1)  # normalize channels independently
    img = normalize(img, 1, 99.8, axis=axis_norm)
    return img


def stardist_seg(img, model):
    pred, _ = model.predict_instances(img)
    pred = f_instance2semantics_max(pred)
    return pred

def run(file_lst, out_path,img_type):
    os.makedirs(out_path, exist_ok=True)
    if img_type.lower()=='he':
        model = StarDist2D.from_pretrained('2D_versatile_he')
    else:
        model = StarDist2D.from_pretrained('2D_demo')

    for i,file in enumerate(tqdm(file_lst, desc='stardist')):
        try:
            name = os.path.split(file)[-1]
            if os.path.exists(os.path.join(out_path, name)):
                continue
            img = tifffile.imread(file)
            #img = cv2.imread(file)
            if img_type.lower() != 'he':
                img = f_ij_16_to_8(img, chunk_size=1000000)
                if img.ndim == 3:
                    #img = f_rgb2gray(img, True)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = pre(img)
            mask = stardist_seg(img, model)
            mask[mask > 0] = 255
            tifffile.imwrite(os.path.join(out_path, name), mask, compression='zlib')
        except:
            traceback.print_exc()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="you should add those parameter")
    parser.add_argument('-i', "--input", help="the input img path")
    parser.add_argument('-o', "--output", help="the output file")
    parser.add_argument("-g", "--gpu", help="the gpu index", default="-1")
    parser.add_argument("-t", "--img_type", help="ss/he")
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    img_type = args.img_type
    gpu = args.gpu
    if gpu == True:
        gpu = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    file_lst = []
    if os.path.isdir(input_path):
        file_lst = glob.glob(os.path.join(input_path, "*.tif")) 
    else:
        file_lst = [input_path]

    run(file_lst, output_path,img_type)
    sys.exit()
