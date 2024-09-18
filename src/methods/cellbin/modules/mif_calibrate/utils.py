#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :utils.py
# @Time      :2022/10/9 16:50
# @Author    :kuisu_dgut@163.com
import numpy as np
import tifffile
import scipy.ndimage as ndi
import os
import cv2
import math


def mat_channel(mat):
    if mat.ndim == 2:
        return 1
    elif mat.shape[0] == 2:
        return 2
    else:
        return 3


def mat_deepth(mat):
    c = mat_channel(mat)
    if c == 1:
        item = mat[0, 0]
    else:
        item = mat[0, 0, 0]
    tp = type(item)
    if tp == np.uint8:
        return 8
    elif tp == np.uint16:
        return 16
    else:
        return 32


def tiff_write(file_name, mat):
    # opencv BGR -> RGB.
    if mat.ndim == 3:
        if mat.shape[2] == 3:
            mat = mat[:, :, (2, 1, 0)]
        mat = np.transpose(mat, (2, 0, 1))

    tifffile.imwrite(file_name, mat)


def tiff_read(file_name):
    if not os.path.exists(file_name):
        return None
    else:
        return tifffile.imread(file_name)


def imageEnhance(image):
    # todo 增加图像增强
    clahe = cv2.createCLAHE(clipLimit=image.max())
    if image.ndim == 3:
        image = image[0]  # 取第一个通道
    image = clahe.apply(image)
    image = (image / image.max()) * 255
    image = image.astype(np.uint8)
    return image


def affineMatrix3D(center, shift=[0, 0], scale=1, rotation=0):
    '''
    Compute the affine transformation matrix by the angle and scale
    :param center: (cx,cy)
    :param angle: rotation
    :param scale: scale
    :return: matrix 3x3
    '''
    w, h = center[0], center[1]
    # 1. 旋转
    angle = math.pi * (rotation / 180)
    # 2. 绕中心点旋转
    # w1 = w * (1 - np.cos(angle)) - h * np.sin(angle)
    # h1 = w * np.sin(angle) + h * (1 - np.cos(angle))
    alpha = scale * math.cos(angle)
    belta = scale * math.sin(angle)
    w1 = (1 - alpha) * center[0] - belta * center[1]
    h1 = (belta * center[0]) + (1 - alpha) * center[1]
    M_rotation = np.array([[alpha, belta, w1],
                           [-belta, alpha, h1],
                           [0, 0, 1]], dtype=np.float32)
    # 尺度缩放
    # M_scale = np.array([[scale[0], 0, 0],
    #                     [0, scale[1], 0],
    #                     [0, 0, 1]], dtype=np.float32)
    # 平移
    M_shift = np.array([[1, 0, shift[1]],
                        [0, 1, shift[0]],
                        [0, 0, 1]], dtype=np.float32)

    M = M_shift @ M_rotation
    return M


def show_3d(image):
    from matplotlib import cm
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    h, w = image.shape
    X = np.arange(0, w, 1)
    Y = np.arange(0, h, 1)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, image, cmap=cm.coolwarm)
    plt.show()


def rot180(arr):
    """
    Rotate the input array over 180°
    """
    ret = np.rot90(arr, 2)
    return ret


def wrap_angle(angles, ceil=2 * np.pi):
    """
    Args:
        angles (float or ndarray, unit depends on kwarg ``ceil``)
        ceil (float): Turnaround value
    """
    angles += ceil / 2.0
    angles %= ceil
    angles -= ceil / 2.0
    return angles


def embed_to(where, what):
    """
    Given a source and destination arrays, put the source into
    the destination so it is centered and perform all necessary operations
    (cropping or aligning)

    Args:
        where: The destination array (also modified inplace)
        what: The source array

    Returns:
        The destination array
    """
    slices_from, slices_to = _get_emslices(where.shape, what.shape)

    where[slices_to[0], slices_to[1]] = what[slices_from[0], slices_from[1]]
    return where


# todo: imreg_dft utils

def argmax_translation(array, filter_pcorr=0, constraints=None, reports=None):
    if constraints is None:
        constraints = dict(tx=(0, None), ty=(0, None))

    # We want to keep the original and here is obvious that
    # it won't get changed inadvertently
    array_orig = array.copy()
    # if filter_pcorr > 0:#median filtering
    #     array = ndi.minimum_filter(array, filter_pcorr)

    ashape = np.array(array.shape, int)
    mask = np.ones(ashape, float)
    # first goes Y, then X
    for dim, key in enumerate(("ty", "tx")):
        if constraints.get(key, (0, None))[1] is None:
            continue
        pos, sigma = constraints[key]
        alen = ashape[dim]
        dom = np.linspace(-alen // 2, -alen // 2 + alen, alen, False)
        if sigma == 0:
            # generate a binary array closest to the position
            idx = np.argmin(np.abs(dom - pos))
            vals = np.zeros(dom.size)
            vals[idx] = 1.0
        else:
            vals = np.exp(- (dom - pos) ** 2 / sigma ** 2)
        if dim == 0:
            mask *= vals[:, np.newaxis]
        else:
            mask *= vals[np.newaxis, :]

    array *= mask

    # WE ARE FFTSHIFTED already.
    # ban translations that are too big
    aporad = (ashape // 6).min()
    mask2 = get_apofield(ashape, aporad, corner=True)
    array *= mask2
    # Find what we look for
    tvec = _argmax_ext(array, 'inf')
    tvec = _interpolate(array_orig, tvec)

    # If we use constraints or min filter,
    # array_orig[tvec] may not be the maximum
    success = _get_success(array_orig, tuple(tvec), 2)

    if reports is not None and reports.show("translation"):
        reports["amt-orig"] = array_orig.copy()
        reports["amt-postproc"] = array.copy()

    return tvec, success


def argmax_angscale(array, log_base, exponent, constraints=None, reports=None):
    """
    Given a power spectrum, we choose the best fit.

    The power spectrum is treated with constraint masks and then
    passed to :func:`_argmax_ext`.
    """
    mask = _get_constraint_mask(array.shape, log_base, constraints)
    array_orig = array.copy()

    array *= mask
    ret = _argmax_ext(array, exponent)
    ret_final = _interpolate(array, ret)

    if reports is not None and reports.show("scale_angle"):
        reports["amas-orig"] = array_orig.copy()
        reports["amas-postproc"] = array.copy()

    success = _get_success(array_orig, tuple(ret_final), 0)
    return ret_final, success


def _get_emslices(shape1, shape2):
    """
    Common code used by :func:`embed_to` and :func:`undo_embed`
    """
    slices_from = []
    slices_to = []
    for dim1, dim2 in zip(shape1, shape2):
        diff = dim2 - dim1
        # In fact: if diff == 0:
        slice_from = slice(None)
        slice_to = slice(None)

        # dim2 is bigger => we will skip some of their pixels
        if diff > 0:
            # diff // 2 + rem == diff
            rem = diff - (diff // 2)
            slice_from = slice(diff // 2, dim2 - rem)
        if diff < 0:
            diff *= -1
            rem = diff - (diff // 2)
            slice_to = slice(diff // 2, dim1 - rem)
        slices_from.append(slice_from)
        slices_to.append(slice_to)
    return slices_from, slices_to


def _get_constraint_mask(shape, log_base, constraints=None):
    """
    Prepare mask to apply to constraints to a cross-power spectrum.
    """
    if constraints is None:
        constraints = {}

    mask = np.ones(shape, float)

    # Here, we create masks that modulate picking the best correspondence.
    # Generally, we look at the log-polar array and identify mapping of
    # coordinates to values of quantities.
    if "scale" in constraints:
        scale, sigma = constraints["scale"]
        scales = np.fft.ifftshift(_get_lograd(shape, log_base))
        # vvv This issome kind of transformation of result of _get_lograd
        # vvv (log radius in pixels) to the linear scale.
        scales *= log_base ** (- shape[1] / 2.0)
        # This makes the scales array low near where scales is near 'scale'
        scales -= 1.0 / scale
        if sigma == 0:
            # there isn't: ascales = np.abs(scales - scale)
            # because scales are already low for values near 'scale'
            ascales = np.abs(scales)
            scale_min = ascales.min()
            mask[ascales > scale_min] = 0
        elif sigma is None:
            pass
        else:
            mask *= np.exp(-scales ** 2 / sigma ** 2)

    if "angle" in constraints:
        angle, sigma = constraints["angle"]
        angles = _get_angles(shape)
        # We flip the sign on purpose
        # TODO: ^^^ Why???
        angles += np.deg2rad(angle)
        # TODO: Check out the wrapping. It may be tricky since pi+1 != 1
        wrap_angle(angles, np.pi)
        angles = np.rad2deg(angles)
        if sigma == 0:
            aangles = np.abs(angles)
            angle_min = aangles.min()
            mask[aangles > angle_min] = 0
        elif sigma is None:
            pass
        else:
            mask *= np.exp(-angles ** 2 / sigma ** 2)

    mask = np.fft.fftshift(mask)
    return mask


def _interpolate(array, rough, rad=2):
    """
    Returns index that is in the array after being rounded.

    The result index tuple is in each of its components between zero and the
    array's shape.
    """
    rough = np.round(rough).astype(int)
    surroundings = _get_subarr(array, rough, rad)
    com = _argmax_ext(surroundings, 1)
    offset = com - rad
    ret = rough + offset
    # similar to win.wrap, so
    # -0.2 becomes 0.3 and then again -0.2, which is rounded to 0
    # -0.8 becomes - 0.3 -> len() - 0.3 and then len() - 0.8,
    # which is rounded to len() - 1. Yeah!
    ret += 0.5
    ret %= np.array(array.shape).astype(int)
    ret -= 0.5
    return ret


def _get_success(array, coord, radius=2):
    """
    Given a coord, examine the array around it and return a number signifying
    how good is the "match".

    Args:
        radius: Get the success as a sum of neighbor of coord of this radius
        coord: Coordinates of the maximum. Float numbers are allowed
            (and converted to int inside)

    Returns:
        Success as float between 0 and 1 (can get slightly higher than 1).
        The meaning of the number is loose, but the higher the better.
    """
    coord = np.round(coord).astype(int)
    coord = tuple(coord)

    subarr = _get_subarr(array, coord, 2)

    theval = subarr.sum()
    theval2 = array[coord]
    # bigval = np.percentile(array, 97)
    # success = theval / bigval
    # TODO: Think this out
    success = np.sqrt(theval * theval2)
    return success


def _get_subarr(array, center, rad):
    """
    Args:
        array (ndarray): The array to search
        center (2-tuple): The point in the array to search around
        rad (int): Search radius, no radius (i.e. get the single point)
            implies rad == 0
    """
    dim = 1 + 2 * rad
    subarr = np.zeros((dim,) * 2)
    corner = np.array(center) - rad
    for ii in range(dim):
        yidx = corner[0] + ii
        yidx %= array.shape[0]
        for jj in range(dim):
            xidx = corner[1] + jj
            xidx %= array.shape[1]
            subarr[ii, jj] = array[yidx, xidx]
    return subarr


def _get_angles(shape):
    """
    In the log-polar spectrum, the (first) coord corresponds to an angle.
    This function returns a mapping of (the two) coordinates
    to the respective angle.
    """
    ret = np.zeros(shape, dtype=np.float64)
    ret -= np.linspace(0, np.pi, shape[0], endpoint=False)[:, np.newaxis]
    return ret


def _get_lograd(shape, log_base):
    """
    In the log-polar spectrum, the (second) coord corresponds to an angle.
    This function returns a mapping of (the two) coordinates
    to the respective scale.

    Returns:
        2D np.ndarray of shape ``shape``, -1 coord contains scales
            from 0 to log_base ** (shape[1] - 1)
    """
    ret = np.zeros(shape, dtype=np.float64)
    ret += np.power(log_base, np.arange(shape[1], dtype=float))[np.newaxis, :]
    return ret


def get_apofield(shape, aporad, corner=False):
    """
    Returns an array between 0 and 1 that goes to zero close to the edges.
    """
    if aporad == 0:
        return np.ones(shape, dtype=float)
    apos = np.hanning(aporad * 2)
    vecs = []
    for dim in shape:
        assert dim > aporad * 2, \
            "Apodization radius %d too big for shape dim. %d" % (aporad, dim)
        toapp = np.ones(dim)
        toapp[:aporad] = apos[:aporad]
        toapp[-aporad:] = apos[-aporad:]
        vecs.append(toapp)
    apofield = np.outer(vecs[0], vecs[1])
    if corner:
        apofield[aporad:-aporad] = 1
        apofield[:, aporad:-aporad] = 1
    return apofield


def _argmax_ext(array, exponent):
    """
    Calculate coordinates of the COM (center of mass) of the provided array.

    Args:
        array (ndarray): The array to be examined.
        exponent (float or 'inf'): The exponent we power the array with. If the
            value 'inf' is given, the coordinage of the array maximum is taken.

    Returns:
        np.ndarray: The COM coordinate tuple, float values are allowed!
    """

    # When using an integer exponent for _argmax_ext, it is good to have the
    # neutral rotation/scale in the center rather near the edges

    ret = None
    if exponent == "inf":
        ret = _argmax2D(array)
    else:
        col = np.arange(array.shape[0])[:, np.newaxis]
        row = np.arange(array.shape[1])[np.newaxis, :]

        arr2 = array ** exponent
        arrsum = arr2.sum()
        if arrsum == 0:
            # We have to return SOMETHING, so let's go for (0, 0)
            return np.zeros(2)
        arrprody = np.sum(arr2 * col) / arrsum
        arrprodx = np.sum(arr2 * row) / arrsum
        ret = [arrprody, arrprodx]
        # We don't use it, but it still tells us about value distribution

    return np.array(ret)


def _argmax2D(array, reports=None):
    """
    Simple 2D argmax function with simple sharpness indication
    """
    amax = np.argmax(array)
    ret = list(np.unravel_index(amax, array.shape))

    return np.array(ret)


def _apodize(what, aporad=None, ratio=None):
    """
    Given an image, it apodizes it (so it becomes quasi-seamless).
    When ``ratio`` is None, color near the edges will converge
    to the same colour, whereas when ratio is a float number, a blurred
    original image will serve as background.

    Args:
        what: The original image
        aporad (int): Radius [px], width of the band near the edges
            that will get modified
        ratio (float or None): When None, the apodization background will
            be a flat color.
            When a float number, the background will be the image itself
            convolved with Gaussian kernel of sigma (aporad / ratio).

    Returns:
        The apodized image
    """
    if aporad is None:
        mindim = min(what.shape)
        aporad = int(mindim * 0.12)
    apofield = get_apofield(what.shape, aporad)
    res = what * apofield
    if ratio is not None:
        ratio = float(ratio)
        bg = ndi.gaussian_filter(what, aporad / ratio, mode='wrap')
    else:
        bg = get_borderval(what, aporad // 2)
    res += bg * (1 - apofield)
    return res


def get_borderval(img, radius=None):
    """
    Given an image and a radius, examine the average value of the image
    at most radius pixels from the edge
    """
    if radius is None:
        mindim = min(img.shape)
        radius = max(1, mindim // 20)
    mask = np.zeros_like(img, dtype=np.bool_)
    mask[:, :radius] = True
    mask[:, -radius:] = True
    mask[radius, :] = True
    mask[-radius:, :] = True

    mean = np.median(img[mask])
    return mean
