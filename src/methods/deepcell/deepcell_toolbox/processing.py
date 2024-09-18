# Copyright 2016-2022 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-toolbox/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Functions for pre- and post-processing image data"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from scipy import ndimage
from skimage import morphology
from skimage import segmentation
from skimage.feature import peak_local_max
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity
from skimage.measure import label


def normalize(image, epsilon=1e-07):
    """Normalize image data by dividing by the maximum pixel value

    Args:
        image (numpy.array): numpy array of image data
        epsilon (float): fuzz factor used in numeric expressions.

    Returns:
        numpy.array: normalized image data
    """
    if not np.issubdtype(image.dtype, np.floating):
        logging.info('Converting image dtype to float')
    image = image.astype('float32')

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            normal_image = (img - img.mean()) / (img.std() + epsilon)
            image[batch, ..., channel] = normal_image
    return image


def histogram_normalization(image, kernel_size=None):
    """Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).

    If one of the inputs is a constant-value array, it will
    be normalized as an array of all zeros of the same shape.

    Args:
        image (numpy.array): numpy array of phase image data.
        kernel_size (integer): Size of kernel for CLAHE,
            defaults to 1/8 of image size.

    Returns:
        numpy.array: Pre-processed image data with dtype float32.
    """
    if not np.issubdtype(image.dtype, np.floating):
        logging.info('Converting image dtype to float')
    image = image.astype('float32')

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            X = image[batch, ..., channel]
            sample_value = X[(0,) * X.ndim]
            if (X == sample_value).all():
                # TODO: Deal with constant value arrays
                # https://github.com/scikit-image/scikit-image/issues/4596
                logging.warning('Found constant value array in batch %s and '
                                'channel %s. Normalizing as zeros.',
                                batch, channel)
                image[batch, ..., channel] = np.zeros_like(X)
                continue

            # X = rescale_intensity(X, out_range='float')
            X = rescale_intensity(X, out_range=(0.0, 1.0))
            X = equalize_adapthist(X, kernel_size=kernel_size)
            image[batch, ..., channel] = X
    return image


def percentile_threshold(image, percentile=99.9):
    """Threshold an image to reduce bright spots

    Args:
        image: numpy array of image data
        percentile: cutoff used to threshold image

    Returns:
        np.array: thresholded version of input image
    """

    processed_image = np.zeros_like(image)
    for img in range(image.shape[0]):
        for chan in range(image.shape[-1]):
            current_img = np.copy(image[img, ..., chan])
            non_zero_vals = current_img[np.nonzero(current_img)]

            # only threshold if channel isn't blank
            if len(non_zero_vals) > 0:
                img_max = np.percentile(non_zero_vals, percentile)

                # threshold values down to max
                threshold_mask = current_img > img_max
                current_img[threshold_mask] = img_max

                # update image
                processed_image[img, ..., chan] = current_img

    return processed_image


def mibi(prediction, edge_threshold=.25, interior_threshold=.25):
    """Post-processing for MIBI data. Uniquely segments every cell by
    repeatedly eroding and dilating the cell interior prediction  until a
    boundary is reached.
    Args:
        prediction: output from a pixelwise transform (edge, interior, bg)
        edge_threshold: confidence threshold to determine edge pixels
        interior_threshold: confidence threshold to determine interior pixels
    Returns:
        transformed data where each cell is labeled uniquely
    """

    def dilate(array, mask, num_dilations):
        copy = np.copy(array)
        for _ in range(0, num_dilations):
            dilated = morphology.dilation(copy)
            # dilate if still in mask range not in another cell
            copy = np.where((mask != 0) & (dilated != copy) & (copy == 0),
                            dilated, copy)
        return copy

    def dilate_nomask(array, num_dilations):
        copy = np.copy(array)
        for _ in range(0, num_dilations):
            dilated = morphology.dilation(copy)
            # if one cell not eating another, dilate
            copy = np.where((dilated != copy) & (copy == 0), dilated, copy)
        return copy

    def erode(array, num_erosions):
        original = np.copy(array)
        for _ in range(0, num_erosions):
            eroded = morphology.erosion(np.copy(original))
            original[original != eroded] = 0
        return original

    # edge = (prediction[..., 0] >= edge_threshold).astype('int')
    edge = np.copy(prediction[..., 0])
    edge[edge < edge_threshold] = 0
    edge[edge >= edge_threshold] = 1

    # interior = (prediction[..., 1] >= interior_threshold).astype('int')
    interior = np.copy(prediction[..., 1])
    interior[interior >= interior_threshold] = 1
    interior[interior < interior_threshold] = 0

    # define foreground as the interior bounded by edge
    fg_thresh = np.logical_and(interior == 1, edge == 0)

    # remove small objects from the foreground segmentation
    fg_thresh = morphology.remove_small_objects(
        fg_thresh, min_size=50, connectivity=1)

    fg_thresh = np.expand_dims(fg_thresh, axis=-1)

    segments = label(np.squeeze(fg_thresh), connectivity=2)

    for _ in range(8):
        segments = dilate(segments, interior, 2)
        segments = erode(segments, 1)

    segments = dilate(segments, interior, 2)

    for _ in range(2):
        segments = dilate_nomask(segments, 1)
        segments = erode(segments, 2)

    segments = np.expand_dims(segments, axis=-1)
    return segments.astype('uint16')


def watershed(image, min_distance=10, min_size=50, threshold_abs=0.05):
    """Use the watershed method to identify unique cells based
    on their distance transform.

    # TODO: labels should be the fgbg output, NOT the union of distances

    Args:
        image (numpy.array): distance transform of image (model output)
        min_distance (int): minimum number of pixels separating peaks
        min_size (int): removes small objects if smaller than min_size.
        threshold_abs (float): minimum intensity of peaks

    Returns:
        numpy.array: image mask where each cell is annotated uniquely
    """
    distance = np.argmax(image, axis=-1)
    labels = (distance > 0).astype('int')

    local_maxi_coords = peak_local_max(
        image[..., -1],
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        labels=labels,
        exclude_border=False)

    # Construct mask from local_maxi coords
    local_maxi = np.zeros_like(image[..., -1], dtype=bool)
    local_maxi[tuple(local_maxi_coords.T)] = True

    # markers = label(local_maxi)
    markers = ndimage.label(local_maxi)[0]
    segments = segmentation.watershed(-distance, markers, mask=labels)
    results = np.expand_dims(segments, axis=-1)
    results = morphology.remove_small_objects(
        results, min_size=min_size, connectivity=1)
    return results


def pixelwise(prediction, threshold=.8, min_size=50, interior_axis=-2):
    """Post-processing for pixelwise transform predictions.
    Uses the interior predictions to uniquely label every instance.

    Args:
        prediction (numpy.array): pixelwise transform prediction
        threshold (float): confidence threshold for interior predictions
        min_size (int): removes small objects if smaller than min_size.

    Returns:
        numpy.array: post-processed data with each cell uniquely annotated
    """
    # instantiate array to be returned
    labeled_prediction = np.zeros(prediction.shape[:-1] + (1,))

    for batch in range(prediction.shape[0]):
        interior = prediction[[batch], ..., interior_axis] > threshold
        labeled = ndimage.label(interior)[0]
        labeled = morphology.remove_small_objects(
            labeled, min_size=min_size, connectivity=1)

        labeled_prediction[batch, ..., 0] = labeled

    return labeled_prediction


# alias for backwards compatibility
phase_preprocess = histogram_normalization
