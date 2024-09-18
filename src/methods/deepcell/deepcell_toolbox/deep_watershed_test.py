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
"""Tests for post-processing functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import pytest

from deepcell_toolbox import deep_watershed


def test_deep_watershed():
    shape = (5, 21, 21, 1)
    maxima = np.random.random(shape) * 100
    interior = np.random.random(shape) * 100
    other = np.random.randint(0, 1, size=shape)
    inputs = [maxima, interior]

    # basic tests for both h_maxima and peak_local_max
    for algo in ('h_maxima', 'peak_local_max'):
        label_img = deep_watershed.deep_watershed(inputs, maxima_algorithm=algo)
        np.testing.assert_equal(label_img.shape, shape[:-1] + (1,))

        # flip the order and give correct indices, same answer
        label_img_2 = deep_watershed.deep_watershed([other, maxima, interior],
                                                    maxima_index=1,
                                                    interior_index=2,
                                                    maxima_algorithm=algo)
        np.testing.assert_array_equal(label_img, label_img_2)

        # all the bells and whistles
        label_img_3 = deep_watershed.deep_watershed(inputs, maxima_algorithm=algo,
                                                    small_objects_threshold=1,
                                                    label_erosion=1,
                                                    pixel_expansion=1,
                                                    fill_holes_threshold=1)

        np.testing.assert_equal(label_img_3.shape, shape[:-1] + (1,))

    # test bad inputs, pairs of maxima and interior shapes
    bad_shapes = [
        ((1, 32, 32, 1), (1, 32, 16, 1)),  # unequal dimensions
        ((1, 32, 32, 1), (1, 16, 32, 1)),  # unequal dimensions
        ((32, 32, 1), (32, 32, 1)),  # no batch dimension
        ((1, 32, 32), (1, 32, 32)),  # no channel dimension
        ((1, 5, 10, 32, 32, 1), (1, 5, 10, 32, 32, 1)),  # too many dims
    ]
    for bad_maxima_shape, bad_interior_shape in bad_shapes:
        bad_inputs = [np.random.random(bad_maxima_shape),
                      np.random.random(bad_interior_shape)]
        with pytest.raises(ValueError):
            deep_watershed.deep_watershed(bad_inputs)

    # test bad values of maxima_algorithm.
    with pytest.raises(ValueError):
        deep_watershed.deep_watershed(inputs, maxima_algorithm='invalid')

    # pass weird data types
    bad_inputs = [
        {'interior-distance': maxima, 'outer-distance': interior},
        None,
    ]
    for bad_input in bad_inputs:
        with pytest.raises(ValueError):
            _ = deep_watershed.deep_watershed(bad_input)

    # test deprecated values still work
    # each pair is the deprecated name, then the new name.
    old_new_pairs = [
        ('min_distance', 'radius', np.random.randint(10)),
        ('distance_threshold', 'interior_threshold', np.random.randint(1, 100) / 100),
        ('detection_threshold', 'maxima_threshold', np.random.randint(1, 100) / 100),
    ]
    for deprecated_arg, new_arg, value in old_new_pairs:
        dep_kwargs = {deprecated_arg: value}
        new_kwargs = {new_arg: value}

        with pytest.deprecated_call():
            dep_img = deep_watershed.deep_watershed(inputs, **dep_kwargs)
        new_img = deep_watershed.deep_watershed(inputs, **new_kwargs)
        np.testing.assert_array_equal(dep_img, new_img)


def test_deep_watershed_mibi():
    shape = (5, 21, 21, 1)
    inner_distance = np.random.random(shape) * 100
    outer_distance = np.random.random(shape) * 100
    fgbg = np.random.randint(0, 1, size=shape)
    pixelwise = np.random.random(shape[:-1] + (3, ))
    model_output = {'inner-distance': inner_distance,
                    'outer-distance': outer_distance,
                    'fgbg-fg': fgbg,
                    'pixelwise-interior': pixelwise}

    # basic tests
    with pytest.deprecated_call():
        watershed_img = deep_watershed.deep_watershed_mibi(model_output=model_output)
    np.testing.assert_equal(watershed_img.shape, shape)

    # turn some knobs
    with pytest.deprecated_call():
        watershed_img = deep_watershed.deep_watershed_mibi(model_output=model_output,
                                                           small_objects_threshold=1,
                                                           pixel_expansion=5)

    # turn turn
    with pytest.deprecated_call():
        watershed_img = deep_watershed.deep_watershed_mibi(model_output=model_output,
                                                           small_objects_threshold=1,
                                                           maxima_model='fgbg-fg',
                                                           interior_model='outer-distance',
                                                           maxima_model_smooth=0,
                                                           fill_holes_threshold=4)

    np.testing.assert_equal(watershed_img.shape, shape)

    for model_under_test in ['interior_model', 'maxima_model']:
        with pytest.raises(ValueError):
            bad_model = {model_under_test: 'bad_model_name'}
            with pytest.deprecated_call():
                watershed_img = deep_watershed.deep_watershed_mibi(model_output=model_output,
                                                                   **bad_model)

    bad_array = pixelwise[..., 0]
    for bad_transform in ['inner-distance', 'pixelwise-interior']:
        bad_model_output = model_output.copy()
        bad_model_output[bad_transform] = bad_array
        with pytest.raises(ValueError):
            with pytest.deprecated_call():
                watershed_img = deep_watershed.deep_watershed_mibi(
                    model_output=bad_model_output
                )


def test_deep_watershed_3D():
    shape = (5, 10, 21, 21, 1)
    maxima = np.random.random(shape) * 100
    interior = np.random.random(shape) * 100
    other = np.random.randint(0, 1, size=shape)
    inputs = [maxima, interior]

    # basic tests for both h_maxima and peak_local_max
    for algo in ('h_maxima', 'peak_local_max'):
        label_img = deep_watershed.deep_watershed(inputs, maxima_algorithm=algo)
        np.testing.assert_equal(label_img.shape, shape[:-1] + (1,))

        # flip the order and give correct indices, same answer
        label_img_2 = deep_watershed.deep_watershed([other, maxima, interior],
                                                    maxima_index=1,
                                                    interior_index=2,
                                                    maxima_algorithm=algo)
        np.testing.assert_array_equal(label_img, label_img_2)

        # all the bells and whistles
        with pytest.warns(UserWarning, match="`fill_holes` is not supported for 3D data"):
            label_img_3 = deep_watershed.deep_watershed(inputs, maxima_algorithm=algo,
                                                        small_objects_threshold=1,
                                                        label_erosion=1,
                                                        pixel_expansion=1,
                                                        fill_holes_threshold=1)

        np.testing.assert_equal(label_img_3.shape, shape[:-1] + (1,))

        # test deprecated `deep_watershed_3D` function
        with pytest.deprecated_call():
            label_img_3d = deep_watershed.deep_watershed_3D(
                inputs, maxima_algorithm=algo)
        np.testing.assert_array_equal(label_img, label_img_3d)
