# Copyright 2016-2022 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
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
# ==============================================================================
"""Tests for metrics.py accuracy statistics"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import datetime
import pytest

from random import sample

import numpy as np
import pandas as pd

from numpy import testing
from skimage.measure import label
from skimage.segmentation import relabel_sequential

from deepcell_toolbox import metrics


def _get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias
    return img


def _get_image_multichannel(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h, 2) * 64
    variance = np.random.rand(img_w, img_h, 2) * (255 - 64)
    img = np.random.rand(img_w, img_h, 2) * variance + bias
    return img


def _generate_test_masks():
    img_w = img_h = 30
    mask_images = []
    for _ in range(8):
        imarray = np.random.randint(2, size=(img_w, img_h, 1))
        mask_images.append(imarray)
    return mask_images


def _generate_stack_3d():
    img_w = img_h = 30
    imarray = np.random.randint(0, high=2, size=(5, img_w, img_h))
    return imarray


def _generate_stack_4d():
    img_w = img_h = 30
    imarray = np.random.randint(0, high=2, size=(5, img_w, img_h, 2))
    return imarray


def _generate_df():
    df = pd.DataFrame(np.random.rand(8, 4))
    return df


def _sample1(w, h, imw, imh, merge):
    """Basic two cell merge/split"""
    x = np.random.randint(2, imw - w * 2)
    y = np.random.randint(2, imh - h * 2)

    im = np.zeros((imw, imh))
    im[0:2, 0:2] = 1
    im[x:x + w, y:y + h] = 2
    im[x + w:x + 2 * w, y:y + h] = 3

    # Randomly rotate to pick horizontal or vertical
    if np.random.random() > 0.5:
        im = np.rot90(im)

    if merge:
        # Return merge error
        pred = im.copy()
        pred[pred == 3] = 2
        return im.astype('int'), pred.astype('int')
    else:
        # Return split error
        true = im.copy()
        true[true == 3] = 2
        return true.astype('int'), im.astype('int')


def _sample1_3D(w, h, imw, imh, merge, z):
    """Two cell merge/split in 3D"""
    y_trues = []
    y_preds = []
    y_true, y_pred = _sample1(w, h, imw, imh, merge)

    for stack in range(z):
        y_trues.append(y_true)
        y_preds.append(y_pred)

    y_true = np.stack(y_trues, axis=0)
    y_pred = np.stack(y_preds, axis=0)
    return y_true, y_pred


def _sample2(w, h, imw, imh, similar_size=False):
    """Merge of three cells"""
    x = np.random.randint(2, imw - w)
    y = np.random.randint(2, imh - h)

    # Determine split points
    if similar_size:
        xs = np.random.randint(w * 0.4, w * 0.6)
        ys = np.random.randint(h * 0.4, h * 0.6)
    else:
        xs = np.random.randint(1, w * 0.9)
        ys = np.random.randint(1, h * 0.9)

    im = np.zeros((imw, imh))
    im[0:2, 0:2] = 1
    im[x:x + xs, y:y + ys] = 2
    im[x + xs:x + w, y:y + ys] = 3
    im[x:x + w, y + ys:y + h] = 4

    return im


def _sample2_2(w, h, imw, imh, merge=True, similar_size=False):

    im1 = _sample2(w, h, imw, imh, similar_size)

    a, b, c = sample(set([2, 3, 4]), 3)
    im2 = im1.copy()
    im2[im2 == b] = a

    # ensure that output is sequential so it doesn't get subsequently relabeled
    im2, _, _ = relabel_sequential(im2.astype('int'))

    # record which values of im1 were correctly assigned
    im1_wrong = {a, b}
    im1_correct = {1, c}

    # figure out which of newly relabeled values in im2 correspond to correct cells
    im2_wrong = {im2[im1 == b][0]}
    im2_correct = {1, im2[im1 == c][0]}

    if merge:
        return im1.astype('int'), im2.astype('int'), im1_wrong, \
            im1_correct, im2_wrong, im2_correct
    else:
        return im2.astype('int'), im1.astype('int'), im2_wrong, \
            im2_correct, im1_wrong, im1_correct


def _sample2_3(w, h, imw, imh, merge=True, similar_size=False):

    im1 = _sample2(w, h, imw, imh, similar_size)

    im2 = im1.copy()
    im2[im2 > 1] = 2

    if merge:
        return im1.astype('int'), im2.astype('int')
    else:
        return im2.astype('int'), im1.astype('int')


def _sample3(w, h, imw, imh):
    """Wrong boundaries for 3 call clump"""

    x = np.random.randint(0, imw - w)
    y = np.random.randint(0, imh - h)

    # Determine split points
    xs = np.random.randint(1, w * 0.9)
    ys = np.random.randint(1, h * 0.9)

    im = np.zeros((imw, imh))
    im[x:x + xs, y:y + ys] = 1
    im[x + xs:x + w, y:y + ys] = 2
    im[x:x + w, y + ys:y + h] = 3

    true = im

    # generate sequence of potential values for predicted split point
    x_splits = np.arange(1, w * 0.9)
    y_splits = np.arange(1, h * 0.9)

    # generate mask to keep values that are sufficiently different from those picked for true image
    x_keep = np.logical_or(x_splits < xs - w * 0.2, x_splits > xs + w * 0.3)
    y_keep = np.logical_or(y_splits < ys - h * 0.2, y_splits > ys + h * 0.3)

    # pick one of appropriate values as new cutoff point
    xs = int(np.random.choice(x_splits[x_keep], 1)[0])
    ys = int(np.random.choice(y_splits[y_keep], 1)[0])

    im = np.zeros((imw, imh))
    im[x:x + xs, y:y + ys] = 1
    im[x + xs:x + w, y:y + ys] = 2
    im[x:x + w, y + ys:y + h] = 3

    pred = im

    return true.astype('int'), pred.astype('int')


def _sample4_loner(w, h, imw, imh, gain):

    x = np.random.randint(2, imw - w * 2)
    y = np.random.randint(2, imh - h * 2)

    im = np.zeros((imw, imh))
    im[0:2, 0:2] = 1
    im[x:x + w, y:y + h] = 2

    if gain:
        # Return loner in pred
        true = im.copy()
        true[true == 2] = 0
        return true.astype('int'), im.astype('int')
    else:
        # Return loner in true
        pred = im.copy()
        pred[pred == 2] = 0
        return im.astype('int'), pred.astype('int')


def _dense_sample():
    true = np.zeros((10, 10))
    true[:5, :5] = 1
    true[:5, 5:] = 2
    true[5:, :5] = 3
    true[5:, 5:] = 4

    pred = np.zeros((10, 10))
    pred[2:, :] = true[2:, :]

    return true.astype('int'), pred.astype('int')


class TestDetection():

    def test_init(self):
        # test correct detection
        true_idx, pred_idx = 1, 1
        detection = metrics.Detection(true_idx, pred_idx)
        assert detection.is_correct

        # test missed detection
        true_idx, pred_idx = 1, None
        detection = metrics.Detection(true_idx, pred_idx)
        assert detection.is_missed

        # test gained detection
        true_idx, pred_idx = None, 1
        detection = metrics.Detection(true_idx, pred_idx)
        assert detection.is_gained

        # test split detection
        true_idx, pred_idx = 1, [1, 2]
        detection = metrics.Detection(true_idx, pred_idx)
        assert detection.is_split

        # test merge detection
        true_idx, pred_idx = [1, 2], 1
        detection = metrics.Detection(true_idx, pred_idx)
        assert detection.is_merge

        # test catastrophe
        true_idx, pred_idx = [1, 2], [2, 3]
        detection = metrics.Detection(true_idx, pred_idx)
        assert detection.is_catastrophe

    def test_hash(self):
        # test that Detections get hashed appropriately
        detection_set = set()
        d1 = metrics.Detection(1, 1)
        detection_set.add(d1)

        d2 = metrics.Detection(1, 1)
        assert d2 is not d1
        # should be in the set since d2 == d1
        assert d2 in detection_set

    def test_eq(self):
        # test Detection equality comparisons
        detection_set = set()
        d1 = metrics.Detection(1, 1)
        detection_set.add(d1)

        d2 = metrics.Detection(1, 1)
        assert d2 is not d1
        assert d2 == d1

        assert d1 != 1

        print(d1)  # test that __repr__ is called


class TestPixelMetrics():

    def test_init(self):
        y_true, _ = _sample1(10, 10, 30, 30, True)

        # Test basic initialization
        o = metrics.PixelMetrics(y_true, y_true)
        assert np.array_equal(o.y_true, o.y_pred)

        # Test mismatched input size
        with pytest.raises(ValueError, match="Input shapes must match"):
            metrics.PixelMetrics(y_true, y_true[0])

    def test_y_true_equals_y_pred(self):
        y_true, _ = _sample1(10, 10, 30, 30, True)
        y_pred = y_true.copy()

        # Test basic initialization
        o = metrics.PixelMetrics(y_true, y_pred)

        # metrics should be perfect since y_true == y_pred
        assert o.recall == 1
        assert o.precision == 1
        assert o.f1 == 1
        assert o.jaccard == 1
        assert o.dice == 1

        # test both empty
        o = metrics.PixelMetrics(np.zeros_like(y_true), np.zeros_like(y_pred))
        assert o.dice == 1

    def test_y_pred_empty(self):
        y_true, _ = _sample1(10, 10, 30, 30, True)

        # Test basic initialization with empty array y_pred
        o = metrics.PixelMetrics(y_true, np.zeros_like(y_true))

        assert o.recall == 0
        assert o.precision == 0
        assert o.f1 == 0
        assert o.jaccard == 0

    def test_y_true_empty(self):
        y_pred, _ = _sample1(10, 10, 30, 30, True)

        # Test basic initialization with empty array y_pred
        o = metrics.PixelMetrics(np.zeros_like(y_pred), y_pred)

        assert np.isnan(o.recall)
        assert o.precision == 0
        assert np.isnan(o.f1)
        assert o.jaccard == 0


class TestObjectMetrics():

    def test_init(self):
        y_true, _ = _sample1(10, 10, 30, 30, True)

        # Test basic initialization
        o = metrics.ObjectMetrics(y_true, y_true)

        # Test __repr__
        print(o)

        # Test using float dtype warns but still works
        with pytest.warns(UserWarning, match="Casting.*from float64 to int"):
            o = metrics.ObjectMetrics(y_true.astype('float'), y_true.astype('float'))

        # test errors thrown for improper ndim inputs
        y_true = np.zeros(shape=(10))  # too few dimensions
        with pytest.raises(ValueError):
            metrics.ObjectMetrics(y_true, y_true)

        y_true = np.zeros(shape=(10, 5, 5, 5))  # too many dimensions
        with pytest.raises(ValueError):
            metrics.ObjectMetrics(y_true, y_true)

        # test errors thrown for improper ndim inputs with 3d data
        y_true = np.zeros(shape=(10, 15))  # too few dimensions
        with pytest.raises(ValueError):
            metrics.ObjectMetrics(y_true, y_true, is_3d=True)

        y_true = np.zeros(shape=(10, 15, 15, 10))  # too many dimensions
        with pytest.raises(ValueError):
            metrics.ObjectMetrics(y_true, y_true, is_3d=True)

        # Test mismatched input size
        with pytest.raises(ValueError):
            metrics.ObjectMetrics(y_true, y_true[0])

    def test__get_props(self):
        y_true, y_pred = _sample1(10, 10, 30, 30, True)
        o = metrics.ObjectMetrics(y_true, y_true)

        props = o._get_props('correct')
        assert props != []

        # Test _get_props with invalid detection type
        with pytest.raises(ValueError):
            o._get_props('invalid_type')

    def test_y_true_equals_y_pred(self):
        y_true, _ = _sample1(10, 10, 30, 30, True)
        y_pred = y_true.copy()

        # Test basic initialization
        o = metrics.ObjectMetrics(y_true, y_pred, force_event_links=True)

        # Check that object numbers are integers
        assert isinstance(o.n_true, int)
        assert isinstance(o.n_pred, int)
        assert o.n_true == o.n_pred

        # metrics should be perfect since y_true == y_pred
        assert o.recall == 1
        assert o.precision == 1
        assert o.f1 == 1
        assert o.jaccard == 1
        assert o.dice == 1

        # test other properties
        assert o.correct_detections == o.n_pred
        assert o.missed_detections == 0
        assert o.gained_detections == 0
        assert o.splits == 0
        assert o.merges == 0
        assert o.catastrophes == 0
        assert o.gained_det_from_split == 0
        assert o.missed_det_from_merge == 0
        assert o.true_det_in_catastrophe == 0
        assert o.pred_det_in_catastrophe == 0

        assert o.missed_props == []
        assert o.merge_props == []
        assert o.split_props == []
        assert o.gained_props == []

    def test_y_pred_empty(self):
        y_true, _ = _sample1(10, 10, 30, 30, True)

        # Test basic initialization with empty array y_pred
        o = metrics.ObjectMetrics(y_true, np.zeros_like(y_true),
                                  force_event_links=True)

        assert o.n_pred == 0
        assert o.correct_detections == 0
        assert o.missed_detections == o.n_true
        assert o.recall == 0
        assert o.precision == 0
        assert o.f1 == 0
        assert o.jaccard == 0
        assert o.gained_det_from_split == 0
        assert o.missed_det_from_merge == 0
        assert o.true_det_in_catastrophe == 0
        assert o.pred_det_in_catastrophe == 0

    def test_y_true_empty(self):
        y_pred, _ = _sample1(10, 10, 30, 30, True)

        # Test basic initialization with empty array y_pred
        o = metrics.ObjectMetrics(np.zeros_like(y_pred), y_pred,
                                  force_event_links=True)

        assert o.n_true == 0
        assert o.correct_detections == 0
        assert o.gained_detections == o.n_pred
        assert o.recall == 0
        assert o.precision == 0
        assert o.f1 == 0
        assert o.jaccard == 0
        assert o.gained_det_from_split == 0
        assert o.missed_det_from_merge == 0
        assert o.true_det_in_catastrophe == 0
        assert o.pred_det_in_catastrophe == 0

    def test_merge_error(self):
        y_true, y_pred = _sample1(10, 10, 30, 30, True)
        o = metrics.ObjectMetrics(y_true, y_pred)
        assert o.merges == 1
        assert o.missed_det_from_merge == 1
        assert o.merge_props != []

    def test_split_error(self):
        y_true, y_pred = _sample1(10, 10, 30, 30, False)
        o = metrics.ObjectMetrics(y_true, y_pred)
        assert o.splits == 1
        assert o.gained_det_from_split == 1
        assert o.split_props != []

    def test_multi_merge_error(self):
        # 3 cells merged together
        # forced event links to ensure accurate assignment
        y_true, y_pred = _sample2_3(10, 10, 30, 30,
                                    merge=True, similar_size=False)
        o = metrics.ObjectMetrics(y_true, y_pred, force_event_links=True)
        assert o.merges == 1
        assert o.missed_det_from_merge == 2
        assert o.merge_props != []

    def test_multi_split_error(self):
        # 1 cell split into 3
        # forced event links to ensure accurate assignment
        y_true, y_pred = _sample2_3(10, 10, 30, 30,
                                    merge=False, similar_size=False)
        o = metrics.ObjectMetrics(y_true, y_pred, force_event_links=True)
        assert o.splits == 1
        assert o.gained_det_from_split == 2
        assert o.split_props != []

    def test_calc_iou(self):
        # TODO: test correctness
        y_true, y_pred = _sample1(10, 10, 30, 30, True)
        o = metrics.ObjectMetrics(y_true, y_pred)

        # Check that it is not equal to initial value
        assert np.count_nonzero(o.iou) != 0
        assert np.count_nonzero(o.seg_thresh) != 0

        # check that image without any background passes
        y_true, y_pred = _dense_sample()
        o = metrics.ObjectMetrics(y_true=y_true, y_pred=y_pred)

        # Check that it is not equal to initial value
        assert np.count_nonzero(o.iou) != 0
        assert np.count_nonzero(o.seg_thresh) != 0

    def test_calc_iou_3D(self):
        # TODO: test correctness
        y_true, y_pred = _sample1_3D(10, 10, 30, 30, True, 8)
        o = metrics.ObjectMetrics(y_true, y_pred, is_3d=True)

        # Check that it is not equal to initial value
        assert np.count_nonzero(o.iou) != 0
        assert np.count_nonzero(o.seg_thresh) != 0


class TestMetrics():

    def test_df_to_dict(self):
        m = metrics.Metrics('test')
        df = _generate_df()

        L = m.df_to_dict(df)

        # Check output types
        assert len(L) != 0
        assert isinstance(L, list)
        assert isinstance(L[0], dict)

    def test_calc_pixel_stats(self):
        m = metrics.Metrics('test')

        y_true = _generate_stack_4d()
        y_pred = _generate_stack_4d()

        pixel_stats = m.calc_pixel_stats(y_true, y_pred)

        for stat in pixel_stats:
            assert 'name' in stat
            assert 'stat_type' in stat

    def test_confusion_matrix(self):
        y_true = _generate_stack_4d()
        y_pred = _generate_stack_4d()

        m = metrics.Metrics('test')

        cm = m.calc_pixel_confusion_matrix(y_true, y_pred)
        testing.assert_equal(cm.shape[0], y_true.shape[-1])

    def test_calc_object_stats(self):
        y_true = label(_generate_stack_3d())
        y_pred = label(_generate_stack_3d())

        m = metrics.Metrics('test')

        # test that metrics are generated
        object_metrics = m.calc_object_stats(y_true, y_pred)
        # each row of metrics corresponds to a batch
        assert len(object_metrics) == len(y_true)

        with pytest.warns(UserWarning, match=".*prediction and truth arrays are empty"):
            object_metrics = m.calc_object_stats(
                np.zeros_like(y_true), np.zeros_like(y_pred))

        # test accuracy of metrics with blank predictions
        assert object_metrics['precision'].sum() == 0
        assert object_metrics['recall'].sum() == 0

        # Raise input size error
        with testing.assert_raises(ValueError):
            m.calc_object_stats(np.random.rand(10, 10), np.random.rand(10, 10))

        # Raise error if y_pred.shape != y_true.shape
        with testing.assert_raises(ValueError):
            m.calc_object_stats(np.random.rand(10, 10), np.random.rand(10,))

        # data that needs to be relabeled raises a warning
        with pytest.warns(UserWarning):
            y_pred[0, 0, 0] = 40
            m.calc_object_stats(y_true, y_pred)

        # seg is deprecated (TODO: this will be removed)
        with pytest.warns(DeprecationWarning):
            _ = metrics.Metrics('test', seg=True)

    def test_calc_object_stats_3d(self):
        y_true = _generate_stack_4d()
        y_pred = _generate_stack_4d()

        m = metrics.Metrics('test', is_3d=True)

        # test that metrics are generated
        object_metrics = m.calc_object_stats(y_true, y_pred)
        # each row of metrics corresponds to a batch
        assert len(object_metrics) == len(y_true)

        # test accuracy of metrics with blank predictions
        with pytest.warns(UserWarning, match=".*prediction and truth arrays are empty"):
            object_metrics = m.calc_object_stats(
                np.zeros_like(y_true), np.zeros_like(y_pred))

        assert object_metrics['precision'].sum() == 0
        assert object_metrics['recall'].sum() == 0

        # Raise error if is_3d and ndim !=4
        with testing.assert_raises(ValueError):
            m3d = metrics.Metrics('test', is_3d=True)
            m3d.calc_object_stats(np.random.random((32, 32, 1)),
                                  np.random.random((32, 32, 1)))

    def test_run_all(self, tmpdir):
        tmpdir = str(tmpdir)
        y_true = label(_generate_stack_3d())
        y_pred = label(_generate_stack_3d())

        name = 'test'
        m = metrics.Metrics(name, outdir=tmpdir)

        m.run_all(y_true, y_pred)

    def test_save_to_json(self, tmpdir):
        name = 'test'
        tmpdir = str(tmpdir)
        m = metrics.Metrics(name, outdir=tmpdir)

        # Create test list to save
        L = []
        for i in range(10):
            L.append(dict(
                name=i,
                value=i,
                feature='test',
                stat_type='output'
            ))

        m.save_to_json(L)
        todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
        outfilename = os.path.join(tmpdir, name + '_' + todays_date + '.json')

        # Check that file exists
        testing.assert_equal(os.path.isfile(outfilename), True)

        # Check that it can be opened
        with open(outfilename) as json_file:
            data = json.load(json_file)

        # Check data types from loaded data
        assert isinstance(data, dict)
        assert np.array_equal(sorted(list(data.keys())), ['metadata', 'metrics'])
        assert isinstance(data['metrics'], list)
        assert isinstance(data['metadata'], dict)


def test__cast_to_tuple():
    assert metrics._cast_to_tuple(None) == ()
    assert metrics._cast_to_tuple(1) == (1,)
    assert metrics._cast_to_tuple((1,)) == (1,)


def test_split_stack():
    # Test batch True condition
    arr = np.ones((10, 100, 100, 1))
    out = metrics.split_stack(arr, True, 10, 1, 10, 2)
    outshape = (10 * 10 * 10, 100 / 10, 100 / 10, 1)
    testing.assert_equal(outshape, out.shape)

    # Test batch False condition
    arr = np.ones((100, 100, 1))
    out = metrics.split_stack(arr, False, 10, 0, 10, 1)
    outshape = (10 * 10, 100 / 10, 100 / 10, 1)
    testing.assert_equal(outshape, out.shape)

    # Test splitting in only one axis
    out = metrics.split_stack(arr, False, 10, 0, 1, 1)
    outshape = (10 * 1, 100 / 10, 100 / 1, 1)
    testing.assert_equal(outshape, out.shape)

    out = metrics.split_stack(arr, False, 1, 0, 10, 1)
    outshape = (10 * 1, 100 / 1, 100 / 10, 1)
    testing.assert_equal(outshape, out.shape)

    # Raise errors for uneven division
    with pytest.raises(ValueError):
        metrics.split_stack(arr, False, 11, 0, 10, 1)
    with pytest.raises(ValueError):
        metrics.split_stack(arr, False, 10, 0, 11, 1)
