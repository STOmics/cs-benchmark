import os
import numpy as np
import cv2


def cell_filter(mask, area_thr: int):
    mask_ = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    for c in contours:
        area = cv2.contourArea(c)
        if area >= area_thr: cv2.fillPoly(mask_, c)
    return mask_


def instance2semantics(ins):
    h, w = ins.shape[:2]
    tmp0 = ins[1:, 1:] - ins[:h - 1, :w - 1]
    ind0 = np.where(tmp0 != 0)

    tmp1 = ins[1:, :w - 1] - ins[:h - 1, 1:]
    ind1 = np.where(tmp1 != 0)
    ins[ind1] = 0
    ins[ind0] = 0
    ins[np.where(ins > 0)] = 1
    return np.array(ins, dtype=np.uint8)


def search_files(file_path, exts):
    files_ = list()
    for root, dirs, files in os.walk(file_path):
        if len(files) == 0:
            continue
        for f in files:
            fn, ext = os.path.splitext(f)
            if ext in exts: files_.append(os.path.join(root, f))

    return files_


def cell_dataset(file_path: str, exts: list):
    img_lst = search_files(file_path, exts)
    return img_lst
    #return [i for i in img_lst if 'img' in i]


def auto_make_dir(file_path: str, src: str, output: str):
    i_output_path = os.path.dirname(file_path.replace(src, output))
    if not os.path.exists(i_output_path): os.makedirs(i_output_path)

    return os.path.join(i_output_path, os.path.basename(file_path))


def cellpose_channel_detect(mat):
    if mat.ndims == 2:
        return [0, 0]
    else:
        h, w, c = mat.shape
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)