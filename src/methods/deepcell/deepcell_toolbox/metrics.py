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
"""Custom metrics for pixel-based and object-based classification accuracy.

The schema for this analysis was adopted from the description of object-based
statistics in Caicedo et al. (2018) Evaluation of Deep Learning Strategies for
Nucleus Segmentation in Fluorescence Images. BioRxiv 335216.

The SEG metric was adapted from Maska et al. (2014). A benchmark for comparison
of cell tracking algorithms. Bioinformatics 30, 1609-1617.

The linear classification schema used to match objects in truth and prediction
frames was adapted from Jaqaman et al. (2008). Robust single-particle tracking
in live-cell time-lapse sequences. Nature Methods 5, 695-702.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
import json
import logging
import operator
import os
import warnings

import numpy as np
import pandas as pd
import networkx as nx

from scipy.optimize import linear_sum_assignment
from scipy.stats import hmean
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from deepcell_toolbox import erode_edges
#from deepcell_toolbox.compute_overlap import compute_overlap  # pylint: disable=E0401
#from deepcell_toolbox.compute_overlap import compute_overlap_3D
from keras_retinanet.utils import compute_overlap


def _cast_to_tuple(x):
    try:
        tup_x = tuple(x)
    except TypeError:
        tup_x = () if x is None else (x,)
    return tup_x


class Detection(object):  # pylint: disable=useless-object-inheritance
    """Object to hold relevant information about a given detection."""

    def __init__(self, true_index=None, pred_index=None):
        # cast the indices as tuples if possible to make them immutable
        try:
            self.true_index = tuple(true_index)
        except TypeError:
            self.true_index = true_index
        try:
            self.pred_index = tuple(pred_index)
        except TypeError:
            self.pred_index = pred_index

    def __eq__(self, other):
        """Custom comparator. Detections with the same indices are the same."""
        try:
            is_true_same = self.true_index == other.true_index
            is_pred_same = self.pred_index == other.pred_index
            return is_true_same and is_pred_same
        except AttributeError:
            return False

    def __hash__(self):
        """Custom hasher, allow Detections to be hashable."""
        return tuple((self.true_index, self.pred_index)).__hash__()

    def __repr__(self):
        return 'Detection({}, {})'.format(self.true_index, self.pred_index)

    @property
    def is_correct(self):
        is_linked = self.true_index is not None and self.pred_index is not None
        return is_linked and not self.is_split and not self.is_merge

    @property
    def is_gained(self):
        return self.true_index is None and self.pred_index is not None

    @property
    def is_missed(self):
        return self.true_index is not None and self.pred_index is None

    @property
    def is_split(self):
        if self.is_gained or self.is_missed:
            return False

        try:
            is_many_pred = len(self.pred_index) > 1
        except TypeError:
            is_many_pred = False

        try:
            is_single_true = len(tuple(self.true_index)) == 1
        except TypeError:
            is_single_true = isinstance(self.true_index, int)

        return is_single_true and is_many_pred

    @property
    def is_merge(self):
        if self.is_gained or self.is_missed:
            return False

        try:
            is_many_true = len(self.true_index) > 1
        except TypeError:
            is_many_true = False

        try:
            is_single_pred = len(tuple(self.pred_index)) == 1
        except TypeError:
            is_single_pred = isinstance(self.pred_index, int)

        return is_single_pred and is_many_true

    @property
    def is_catastrophe(self):
        if self.is_gained or self.is_missed:
            return False

        try:
            is_many_true = len(self.true_index) > 1
        except TypeError:
            is_many_true = False

        try:
            is_many_pred = len(self.pred_index) > 1
        except TypeError:
            is_many_pred = False

        return is_many_true and is_many_pred


class BaseMetrics(object):  # pylint: disable=useless-object-inheritance

    """Base class for Metrics classes."""

    def __init__(self, y_true, y_pred):
        if y_pred.shape != y_true.shape:
            raise ValueError('Input shapes must match. Shape of prediction '
                             'is: {}.  Shape of y_true is: {}'.format(
                                 y_pred.shape, y_true.shape))

        if not np.issubdtype(y_true.dtype, np.integer):
            warnings.warn('Casting y_true from {} to int'.format(y_true.dtype))
            y_true = y_true.astype('int32')

        if not np.issubdtype(y_pred.dtype, np.integer):
            warnings.warn('Casting y_pred from {} to int'.format(y_pred.dtype))
            y_pred = y_pred.astype('int32')

        self.y_true = y_true
        self.y_pred = y_pred


class PixelMetrics(BaseMetrics):
    """Calculates pixel-based statistics.
    (Dice, Jaccard, Precision, Recall, F-measure)

    Takes in raw prediction and truth data in order to calculate accuracy
    metrics for pixel based classfication. Statistics were chosen according
    to the guidelines presented in Caicedo et al. (2018) Evaluation of Deep
    Learning Strategies for Nucleus Segmentation in Fluorescence Images.
    BioRxiv 335216.

    Args:
        y_true (numpy.array): Binary ground truth annotations for a single
            feature, (batch,x,y)
        y_pred (numpy.array): Binary predictions for a single feature,
            (batch,x,y)

    Raises:
        ValueError: Shapes of y_true and y_pred do not match.

    Warning:
        Comparing labeled to unlabeled data will produce low accuracy scores.
        Make sure to input the same type of data for y_true and y_pred
    """

    def __init__(self, y_true, y_pred):
        super(PixelMetrics, self).__init__(
            y_true=(y_true != 0).astype('int'),
            y_pred=(y_pred != 0).astype('int'))

        self._y_true_sum = np.count_nonzero(self.y_true)
        self._y_pred_sum = np.count_nonzero(self.y_pred)

        # Calculations for IOU
        self._intersection = np.count_nonzero(np.logical_and(self.y_true, self.y_pred))
        self._union = np.count_nonzero(np.logical_or(self.y_true, self.y_pred))

    @classmethod
    def get_confusion_matrix(cls, y_true, y_pred, axis=-1):
        """Calculate confusion matrix for pixel classification data.

        Args:
            y_true (numpy.array): Ground truth annotations after any
                necessary transformations
            y_pred (numpy.array): Prediction array
            axis (int): The channel axis of the input arrays.

        Returns:
            numpy.array: nxn confusion matrix determined by number of features.
        """
        # Argmax collapses on feature dimension to assign class to each pixel
        # Flatten is required for confusion matrix
        y_true = y_true.argmax(axis=axis).flatten()
        y_pred = y_pred.argmax(axis=axis).flatten()
        return confusion_matrix(y_true, y_pred)

    @property
    def recall(self):
        try:
            _recall = self._intersection / self._y_true_sum
        except ZeroDivisionError:
            _recall = np.nan
        return _recall

    @property
    def precision(self):
        try:
            _precision = self._intersection / self._y_pred_sum
        except ZeroDivisionError:
            _precision = 0
        return _precision

    @property
    def f1(self):
        _recall = self.recall
        _precision = self.precision

        # f1 is nan if recall is nan and no false negatives
        if np.isnan(_recall) and _precision == 0:
            return np.nan

        f_measure = hmean([_recall, _precision])
        # f_measure = (2 * _precision * _recall) / (_precision + _recall)
        return f_measure

    @property
    def dice(self):
        y_sum = self._y_true_sum + self._y_pred_sum
        if y_sum == 0:
            warnings.warn('DICE score is technically 1.0, '
                          'but prediction and truth arrays are empty.')
            return 1.0

        return 2.0 * self._intersection / y_sum

    @property
    def jaccard(self):
        try:
            _jaccard = self._intersection / self._union
        except ZeroDivisionError:
            _jaccard = np.nan
        return _jaccard

    def to_dict(self):
        return {
            'jaccard': self.jaccard,
            'recall': self.recall,
            'precision': self.precision,
            'f1': self.f1,
            'dice': self.dice,
        }


def get_box_labels(arr):
    """Get the bounding box and label for all objects in the image.

    Args:
        arr (np.array): integer label array of objects.

    Returns:
        tuple(list(np.array), list(int)): A tuple of bounding boxes and
            the corresponding integer labels.
    """
    props = regionprops(np.squeeze(arr.astype('int')), cache=False)
    boxes, labels = [], []
    for prop in props:
        boxes.append(np.array(prop.bbox))
        labels.append(int(prop.label))
    boxes = np.array(boxes).astype('double')
    return boxes, labels


class ObjectMetrics(BaseMetrics):
    """Classifies object prediction errors as TP, FP, FN, merge or split

    The schema for this analysis was adopted from the description of
    object-based statistics in Caicedo et al. (2018) Evaluation of Deep
    Learning Strategies for Nucleus Segmentation in Fluorescence Images.
    BioRxiv 335216.
    The SEG metric was adapted from Maska et al. (2014). A benchmark for
    comparison of cell tracking algorithms.
    Bioinformatics 30, 1609-1617.
    The linear classification schema used to match objects in truth and
    prediction frames was adapted from Jaqaman et al. (2008).
    Robust single-particle tracking in live-cell time-lapse sequences.
    Nature Methods 5, 695-702.

    Args:
        y_true (numpy.array): Labeled ground truth annotation
        y_pred (numpy.array): Labled object prediction, same size as y_true
        cutoff1 (:obj:`float`, optional): Threshold for overlap in cost matrix,
            smaller values are more conservative, default 0.4
        cutoff2 (:obj:`float`, optional): Threshold for overlap in unassigned
            cells, smaller values are better, default 0.1
        seg (:obj:`bool`, optional): Calculates SEG score for cell tracking
            competition
        force_event_links(:obj:'bool, optional): Flag that determines whether to modify IOU
            calculation so that merge or split events with cells of very different sizes are
            never misclassified as misses/gains.
        is_3d(:obj:'bool', optional): Flag that determines whether or not the input data
            should be treated as 3-dimensional.

    Raises:
        ValueError: If y_true and y_pred are not the same shape
        ValueError: If data_type is 2D, if input shape does not have ndim 2 or 3
        ValueError: If data_type is 3D, if input shape does not have ndim 3
    """
    def __init__(self,
                 y_true,
                 y_pred,
                 cutoff1=0.4,
                 cutoff2=0.1,
                 force_event_links=False,
                 is_3d=False):

        # If 2D, dimensions can be 3 or 4 (with or without channel dimension)
        if not is_3d and y_true.ndim not in {2, 3}:
            raise ValueError('Expected dimensions for y_true (2D data) are 2 '
                             '(x, y) and 3 (x, y, chan). '
                             'Got ndim: {}'.format(y_true.ndim))

        elif is_3d and y_true.ndim != 3:
            raise ValueError('Expected dimensions for y_true (3D data) is 3.'
                             'Requires format is: (z, x, y)'
                             'Got ndim: {}'.format(y_true.ndim))

        super(ObjectMetrics, self).__init__(y_true=y_true, y_pred=y_pred)

        self.cutoff1 = cutoff1
        self.cutoff2 = cutoff2
        self.is_3d = is_3d

        self.compute_overlap = compute_overlap_3D if is_3d else compute_overlap

        self.n_true = len(np.unique(self.y_true[np.nonzero(self.y_true)]))
        self.n_pred = len(np.unique(self.y_pred[np.nonzero(self.y_pred)]))

        # keep track of every pair of objects through the detections dict
        # using tuple(true_index, pred_index): Detection as a key/vaue pair
        self._detections = set()

        # store the keys of relevant Detections in a set for easy fetching
        # types of detections
        self._splits = set()
        self._gained = set()
        self._missed = set()
        # types of errors
        self._merges = set()
        self._catastrophes = set()
        self._correct = set()

        # IoU: used to determine relative overlap of y_pred and y_true
        self.iou = np.zeros((self.n_true, self.n_pred))

        # used to determine seg score
        self.seg_thresh = np.zeros((self.n_true, self.n_pred))

        # Check if either frame is empty before proceeding
        if self.n_true == 0:
            logging.info('Ground truth frame is empty')

        if self.n_pred == 0:
            logging.info('Prediction frame is empty')

        self._calc_iou()  # set self.iou and update self.seg_thresh

        self.iou_modified = self._get_modified_iou(force_event_links)

        matrix = self._linear_assignment()

        # Identify direct matches as true positives
        correct_index = np.nonzero(matrix[:self.n_true, :self.n_pred])

        for i, j in zip(correct_index[0], correct_index[1]):
            self._add_detection(true_index=int(i), pred_index=int(j))

        # Calc seg score for true positives if requested
        iou_mask = np.where(self.seg_thresh == 0, self.iou, np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            # correct_index may be empty, suppress mean of empty slice warning
            self.seg_score = np.nanmean(iou_mask[correct_index])

        # Classify other errors using a graph
        G = self._array_to_graph(matrix)
        self._classify_graph(G)

        # Calculate pixel-level stats
        self.pixel_stats = PixelMetrics(y_true, y_pred)

    def _add_detection(self, true_index=None, pred_index=None):
        detection = Detection(true_index=true_index, pred_index=pred_index)

        self._detections.add(detection)

        # keep track of all error types
        # TODO: better way to do this?
        if detection.is_correct:
            self._correct.add(detection)
        if detection.is_gained:
            self._gained.add(detection)
        if detection.is_missed:
            self._missed.add(detection)
        if detection.is_split:
            self._splits.add(detection)
        if detection.is_merge:
            self._merges.add(detection)
        if detection.is_catastrophe:
            self._catastrophes.add(detection)

    def _calc_iou(self):
        """Calculates IoU matrix for each pairwise comparison between true and
        predicted. Additionally, if seg is True, records a 1 for each pair of
        objects where $|Tbigcap P| > 0.5 * |T|$
        """
        # Use bounding boxes to find masks that are likely to overlap
        y_true_boxes, y_true_labels = get_box_labels(self.y_true)
        y_pred_boxes, y_pred_labels = get_box_labels(self.y_pred)

        if not y_true_boxes.shape[0] or not y_pred_boxes.shape[0]:
            return  # cannot compute overlaps of nothing

        # has the form [gt_bbox, res_bbox]
        overlaps = self.compute_overlap(y_true_boxes, y_pred_boxes)

        # Find the bboxes that have any overlap
        # (ind_ corresponds to box number - starting at 0)
        ind_true, ind_pred = np.nonzero(overlaps)

        # TODO: this accounts for ~50+% of the time spent on calc_iou
        for index in range(ind_true.shape[0]):
            iou_y_true_idx = y_true_labels[ind_true[index]]
            iou_y_pred_idx = y_pred_labels[ind_pred[index]]

            is_true = self.y_true == iou_y_true_idx
            is_pred = self.y_pred == iou_y_pred_idx

            intersection = np.count_nonzero(np.logical_and(is_true, is_pred))
            union = np.count_nonzero(np.logical_or(is_true, is_pred))

            iou = intersection / union

            # Subtract 1 from index to account for skipping 0
            self.iou[iou_y_true_idx - 1, iou_y_pred_idx - 1] = iou

            if intersection > 0.5 * np.count_nonzero(self.y_true == index):
                self.seg_thresh[iou_y_true_idx - 1, iou_y_pred_idx - 1] = 1

    def _get_modified_iou(self, force_event_links):
        """Modifies the IoU matrix to boost the value for small cells.

        Args:
            force_event_links (:obj:`bool'): Whether to modify IOU values of
                large objects if they have been split or merged by
                a small object.

        Returns:
            np.array: The modified IoU matrix.
        """
        # identify cells that have matches in IOU but may be too small
        true_labels, pred_labels = np.nonzero(
            np.logical_and(self.iou > 0, self.iou < 1 - self.cutoff1)
        )

        iou_modified = self.iou.copy()

        for idx in range(len(true_labels)):
            # add 1 to get back to original label id
            true_idx, pred_idx = true_labels[idx], pred_labels[idx]
            true_label, pred_label = true_idx + 1, pred_idx + 1
            true_mask = self.y_true == true_label
            pred_mask = self.y_pred == pred_label

            # fraction of true cell that is contained within pred cell, vice versa
            true_in_pred = np.count_nonzero(
                self.y_true[pred_mask] == true_label) / np.sum(true_mask)
            pred_in_true = np.count_nonzero(
                self.y_pred[true_mask] == pred_label) / np.sum(pred_mask)

            iou_val = self.iou[true_idx, pred_idx]
            max_val = np.max([true_in_pred, pred_in_true])

            # if this cell has a small IOU due to its small size,
            # but is at least half contained within the big cell,
            # we bump its IOU value up so it doesn't get dropped from the graph
            if iou_val <= self.cutoff1 and max_val > 0.5:
                iou_modified[true_idx, pred_idx] = self.cutoff2

                # optionally, we can also decrease the IOU value of the cell
                # that swallowed up the small cell so that it doesn't directly
                # match a different cell
                if force_event_links and true_in_pred > 0.5:
                    fix_idx = np.nonzero(self.iou[:, pred_idx] >= 1 - self.cutoff1)
                    iou_modified[fix_idx, pred_idx] = 1 - self.cutoff1 - 0.01

                if force_event_links and pred_in_true > 0.5:
                    fix_idx = np.nonzero(self.iou[true_idx, :] >= 1 - self.cutoff1)
                    iou_modified[true_idx, fix_idx] = 1 - self.cutoff1 - 0.01

        return iou_modified

    def _get_cost_matrix(self):
        """Assembles cost matrix using the iou matrix and cutoff1

        The previously calculated iou matrix is cast into the top left and
        transposed for the bottom right corner. The diagonals of the two
        remaining corners are populated according to cutoff1. The lower the
        value of cutoff1 the more likely it is for the linear sum assignment
        to pick unmatched assignments for objects.
        """
        n_obj = self.n_true + self.n_pred
        matrix = np.ones((n_obj, n_obj))

        # Assign 1 - iou to top left and bottom right
        cost = 1 - self.iou_modified
        matrix[:self.n_true, :self.n_pred] = cost
        matrix[n_obj - self.n_pred:, n_obj - self.n_true:] = cost.T

        # Calculate diagonal corners
        bl = (self.cutoff1 * np.eye(self.n_pred)
              + np.ones((self.n_pred, self.n_pred))
              - np.eye(self.n_pred))
        tr = (self.cutoff1 * np.eye(self.n_true)
              + np.ones((self.n_true, self.n_true))
              - np.eye(self.n_true))

        # Assign diagonals to cm
        matrix[n_obj - self.n_pred:, :self.n_pred] = bl
        matrix[:self.n_true, n_obj - self.n_true:] = tr
        return matrix

    def _linear_assignment(self):
        """Runs linear sun assignment on cost matrix, identifies true
        positives and unassigned true and predicted cells.

        True positives correspond to assignments in the top left or bottom
        right corner. There are two possible unassigned positions: true cell
        unassigned in bottom left or predicted cell unassigned in top right.
        """
        cost_matrix = self._get_cost_matrix()

        results = linear_sum_assignment(cost_matrix)

        # Map results onto cost matrix
        assignment_matrix = np.zeros_like(cost_matrix)
        assignment_matrix[results] = 1
        return assignment_matrix

    def _array_to_graph(self, matrix):
        """Transform matrix for unassigned cells into a graph object

        In order to cast the iou matrix into a graph form, we treat each
        unassigned cell as a node. The iou values for each pair of cells is
        treated as an edge between nodes/cells. Any iou values equal to 0 are
        dropped because they indicate no overlap between cells.

        Args:
            matrix (np.array): Assignment matrix.
        """
        # Collect unassigned objects
        x, y = matrix.shape
        gained, _ = np.nonzero(matrix[x - self.n_pred:, :self.n_pred])
        missed, _ = np.nonzero(matrix[:self.n_true, y - self.n_true:])

        # Use meshgrid to get true and predicted object index for each val
        tt, pp = np.meshgrid(missed, gained, indexing='ij')

        true_nodes = tt.flatten()
        pred_nodes = pp.flatten()

        # construct list of edges for networkx
        G = nx.Graph()

        for t, p in zip(true_nodes, pred_nodes):
            # edges between overlapping objects only
            if self.iou_modified[t, p] >= self.cutoff2:
                G.add_edge('true_{}'.format(t), 'pred_{}'.format(p))

        # Add nodes to ensure all cells are included
        G.add_nodes_from(('true_{}'.format(n) for n in missed))
        G.add_nodes_from(('pred_{}'.format(n) for n in gained))

        return G

    def _classify_graph(self, G):
        """Assign each node in graph to an error type

        Nodes with a degree (connectivity) of 0 correspond to either false
        positives or false negatives depending on the origin of the node from
        either the predicted objects (false positive) or true objects
        (false negative). Any nodes with a connectivity of 1 are considered to
        be true positives that were missed during linear assignment.
        Finally any nodes with degree >= 2 are indicative of a merge or split
        error. If the top level node is a predicted cell, this indicates a merge
        event. If the top level node is a true cell, this indicates a split event.
        """
        # Find subgraphs, e.g. merge/split
        for g in (G.subgraph(c) for c in nx.connected_components(G)):
            # Get the highest degree node
            _, max_d = max(dict(g.degree).items(), key=operator.itemgetter(1))

            true_indices, pred_indices = [], []

            for node in g.nodes:
                node_type, index = node.split('_')
                index = int(index) + 1

                if node_type == 'true':
                    if max_d > 1:
                        true_indices.append(index)
                    else:
                        self._add_detection(true_index=index)

                if node_type == 'pred':
                    if max_d > 1:
                        pred_indices.append(index)
                    else:
                        self._add_detection(pred_index=index)

            self._add_detection(
                true_index=tuple(true_indices) if true_indices else None,
                pred_index=tuple(pred_indices) if pred_indices else None,
            )

    def _get_props(self, detection_type):
        prediction_types = {
            'gained',
        }
        is_pred_type = detection_type in prediction_types
        arr = self.y_pred if is_pred_type else self.y_true
        label_image = np.zeros_like(arr)
        attrname = '_{}'.format(detection_type)

        try:
            detections = getattr(self, attrname)
        except AttributeError:
            raise ValueError('Invalid detection_type: {}'.format(
                detection_type))

        for det in detections:
            idx = det.pred_index if is_pred_type else det.true_index
            idx = idx if isinstance(idx, tuple) else (idx,)
            for i in idx:
                label_image[arr == i] = i

        return regionprops(label_image)

    def __repr__(self):
        """Format the calculated statistics as a ``pd.DataFrame``."""
        return json.dumps(self.to_dict())

    def to_dict(self):
        """Return a dictionary representation of the calclulated metrics."""
        return {
            'n_pred': self.n_pred,
            'n_true': self.n_true,
            'correct_detections': self.correct_detections,
            'missed_detections': self.missed_detections,
            'gained_detections': self.gained_detections,
            'missed_det_from_merge': self.missed_det_from_merge,
            'gained_det_from_split': self.gained_det_from_split,
            'true_det_in_catastrophe': self.true_det_in_catastrophe,
            'pred_det_in_catastrophe': self.pred_det_in_catastrophe,
            'merge': self.merges,
            'split': self.splits,
            'catastrophe': self.catastrophes,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'seg': self.seg_score,
            'jaccard': self.jaccard,
            'dice': self.dice,
        }

    @property
    def correct_detections(self):
        return len(self._correct)

    @property
    def missed_detections(self):
        return len(self._missed)

    @property
    def gained_detections(self):
        return len(self._gained)

    @property
    def splits(self):
        return len(self._splits)

    @property
    def merges(self):
        return len(self._merges)

    @property
    def catastrophes(self):
        return len(self._catastrophes)

    @property
    def gained_det_from_split(self):
        gained_dets = 0
        for det in self._splits:
            true_idx = _cast_to_tuple(det.true_index)
            pred_idx = _cast_to_tuple(det.pred_index)
            gained_dets += len(true_idx) + len(pred_idx) - 2
        return gained_dets

    @property
    def missed_det_from_merge(self):
        missed_dets = 0
        for det in self._merges:
            true_idx = _cast_to_tuple(det.true_index)
            pred_idx = _cast_to_tuple(det.pred_index)
            missed_dets += len(true_idx) + len(pred_idx) - 2
        return missed_dets

    @property
    def true_det_in_catastrophe(self):
        return sum([len(d.true_index) for d in self._catastrophes])

    @property
    def pred_det_in_catastrophe(self):
        return sum([len(d.pred_index) for d in self._catastrophes])

    @property
    def split_props(self):
        return self._get_props('splits')

    @property
    def merge_props(self):
        return self._get_props('merges')

    @property
    def missed_props(self):
        return self._get_props('missed')

    @property
    def gained_props(self):
        return self._get_props('gained')

    @property
    def recall(self):
        try:
            recall = self.correct_detections / self.n_true
        except ZeroDivisionError:
            recall = 0
        return recall

    @property
    def precision(self):
        try:
            precision = self.correct_detections / self.n_pred
        except ZeroDivisionError:
            precision = 0
        return precision

    @property
    def f1(self):
        return hmean([self.recall, self.precision])

    @property
    def jaccard(self):
        return self.pixel_stats.jaccard

    @property
    def dice(self):
        return self.pixel_stats.dice

    def plot_errors(self):
        """Plots the errors identified from linear assignment code.

        This must be run with sequentially relabeled data.

        TODO: this is not working!
        """

        import matplotlib as mpl
        import matplotlib.pyplot as plt

        # erode edges for easier visualization of adjacent cells
        y_true = erode_edges(self.y_true.copy(), 1)
        y_pred = erode_edges(self.y_pred.copy(), 1)

        # semantic labels for each error
        categories = ['Background', 'missed', 'splits', 'merges',
                      'gained', 'catastrophes', 'correct']

        # Background is set to zero
        plotting_tif = np.zeros_like(y_true)

        # missed detections are tracked with true labels
        misses = [d.true_index for d in self._missed]
        plotting_tif[np.isin(y_true, misses)] = 1

        # skip background and misses, already done
        for i, category in enumerate(categories[2:]):
            # the rest are all on y_pred
            labels = list(getattr(self, '_{}'.format(category)))
            plotting_tif[np.isin(y_pred, labels)] = i + 2

        plotting_colors = ['Black', 'Pink', 'Blue', 'Green',
                           'tan', 'Red', 'Grey']

        cmap = mpl.colors.ListedColormap(plotting_colors)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        mat = ax.imshow(plotting_tif, cmap=cmap,
                        vmin=np.min(plotting_tif) - .5,
                        vmax=np.max(plotting_tif) + .5)

        # tell the colorbar to tick at integers
        ticks = np.arange(len(categories))
        cbar = fig.colorbar(mat, ticks=ticks)
        cbar.ax.set_yticklabels(categories)
        fig.tight_layout()


class Metrics(object):
    """Class to calculate and save various segmentation metrics.

    Args:
        model_name (str): Name of the model which determines output file names
        outdir (:obj:`str`, optional): Directory to save json file, default ''
        cutoff1 (:obj:`float`, optional): Threshold for overlap in cost matrix,
            smaller values are more conservative, default 0.4
        cutoff2 (:obj:`float`, optional): Threshold for overlap in unassigned
            cells, smaller values are better, default 0.1
        pixel_threshold (:obj:`float`, optional): Threshold for converting
            predictions to binary
        ndigits (:obj:`int`, optional): Sets number of digits for rounding,
            default 4
        feature_key (:obj:`list`, optional): List of strings, feature names
        json_notes (:obj:`str`, optional): Str providing any additional
            information about the model
        force_event_links(:obj:`bool`, optional): Flag that determines whether to modify IOU
            calculation so that merge or split events with cells of very different sizes are
            never misclassified as misses/gains.
        is_3d(:obj:`bool`, optional): Flag that determines whether or not the input data
            should be treated as 3-dimensional.

    Examples:
        >>> from deepcell import metrics
        >>> m = metrics.Metrics('model_name')
        >>> all_metrics = m.run_all(y_true, y_pred)
        >>> m.save_to_json(all_metrics)
    """
    def __init__(self, model_name,
                 outdir='',
                 cutoff1=0.4,
                 cutoff2=0.1,
                 pixel_threshold=0.5,
                 ndigits=4,
                 crop_size=None,
                 feature_key=[],
                 json_notes='',
                 force_event_links=False,
                 is_3d=False,
                 **kwargs):
        self.model_name = model_name
        self.outdir = outdir
        self.cutoff1 = cutoff1
        self.cutoff2 = cutoff2
        self.pixel_threshold = pixel_threshold
        self.ndigits = ndigits
        self.crop_size = crop_size
        self.feature_key = feature_key
        self.json_notes = json_notes
        self.force_event_links = force_event_links
        self.is_3d = is_3d

        if 'seg' in kwargs:
            warnings.warn('seg is deprecated and will be removed '
                          'in a future release', DeprecationWarning)

        # Initialize output list to collect stats
        self.object_metrics = []
        self.pixel_metrics = []

    def df_to_dict(self, df, stat_type='pixel'):
        """Output pandas df as a list of dictionary objects

        Args:
            df (pandas.DataFrame): Dataframe of statistics for each channel
            stat_type (str): Category of statistic.

        Returns:
            list: List of dictionaries
        """

        # Initialize output dictionary
        L = []

        # Write out average statistics
        for k, v in df.mean().items():
            L.append(dict(
                name=k,
                value=v,
                feature='average',
                stat_type=stat_type,
            ))

        # Save individual stats to list
        for i, row in df.iterrows():
            for k, v in row.items():
                L.append(dict(
                    name=k,
                    value=v,
                    feature=i,
                    stat_type=stat_type,
                ))

        return L

    def calc_pixel_stats(self, y_true, y_pred, axis=-1):
        """Calculate pixel statistics for each feature.

        ``y_true`` should have the appropriate transform applied to match
        ``y_pred``. Each channel is converted to binary using the threshold
        ``pixel_threshold`` prior to calculation of accuracy metrics.

        Args:
            y_true (numpy.array): Ground truth annotations after transform
            y_pred (numpy.array): Model predictions without labeling

        Returns:
            list: list of dictionaries with each stat being a key.

        Raises:
            ValueError: If y_true and y_pred are not the same shape
        """
        n_features = y_pred.shape[axis]

        pixel_metrics = []

        slc = [slice(None)] * y_pred.ndim
        for i in range(n_features):
            slc[axis] = slice(i, i + 1)
            yt = y_true[tuple(slc)] > self.pixel_threshold
            yp = y_pred[tuple(slc)] > self.pixel_threshold
            pm = PixelMetrics(yt, yp)
            pixel_metrics.append(pm.to_dict())

        pixel_df = pd.DataFrame.from_records(pixel_metrics)

        # Calculate confusion matrix
        cm = PixelMetrics.get_confusion_matrix(y_true, y_pred, axis=axis)

        print('\n____________Pixel-based statistics____________\n')
        print(pixel_df)
        print('\nConfusion Matrix')
        print(cm)

        output = self.df_to_dict(pixel_df)

        output.append(dict(
            name='confusion_matrix',
            value=cm.tolist(),
            feature='all',
            stat_type='pixel'
        ))
        return output

    def calc_pixel_confusion_matrix(self, y_true, y_pred, axis=-1):
        """DEPRECATED: Use ``PixelMetrics.get_confusion_matrix``.

        Calculate confusion matrix for pixel classification data.

        Args:
            y_true (numpy.array): Ground truth annotations after any
                necessary transformations
            y_pred (numpy.array): Prediction array
            axis (int): The channel axis of the input arrays.

        Returns:
            numpy.array: nxn confusion matrix determined by number of features.
        """
        return PixelMetrics.get_confusion_matrix(y_true, y_pred, axis=axis)

    def calc_object_stats(self, y_true, y_pred, progbar=True):
        """Calculate object statistics and save to output

        Loops over each frame in the zeroth dimension, which should pass in
        a series of 2D arrays for analysis. 'metrics.split_stack' can be
        used to appropriately reshape the input array if necessary

        Args:
            y_true (numpy.array): Labeled ground truth annotations
            y_pred (numpy.array): Labeled prediction mask
            progbar (bool): Whether to show the progress tqdm progress bar

        Returns:
            list: list of dictionaries with each stat being a key.

        Raises:
            ValueError: If y_true and y_pred are not the same shape
            ValueError: If data_type is 2D, if input shape does not have ndim 3 or 4
            ValueError: If data_type is 3D, if input shape does not have ndim 4
        """
        if y_pred.shape != y_true.shape:
            raise ValueError('Input shapes need to match. Shape of prediction '
                             'is: {}.  Shape of y_true is: {}'.format(
                                 y_pred.shape, y_true.shape))

        # If 2D, dimensions can be 3 or 4 (with or without channel dimension)
        if not self.is_3d:
            if y_true.ndim not in {3, 4}:
                raise ValueError('Expected dimensions for y_true (2D data) are 3 or 4.'
                                 'Accepts: (batch, x, y), or (batch, x, y, chan)'
                                 'Got ndim: {}'.format(y_true.ndim))

        # If 3D, inputs must have 4 dimensions (batch, z, x, y) - cannot have channel dimension or
        # _classify_graph breaks, as it expects input to be 2D or 3D
        # TODO - add compatibility for multi-channel 3D-data
        else:
            if y_true.ndim != 4:
                raise ValueError('Expected dimensions for y_true (3D data) is 4. '
                                 'Required format is: (batch, z, x, y) '
                                 'Got ndim: {}'.format(y_true.ndim))

        all_object_metrics = []  # store all calculated metrics
        is_batch_relabeled = False  # used to warn if batches were relabeled

        for i in tqdm(range(y_true.shape[0]), disable=not progbar):
            # check if labels aren't sequential, raise warning on first occurence if so
            true_batch, pred_batch = y_true[i], y_pred[i]
            true_batch_relabel, _, _ = relabel_sequential(true_batch)
            pred_batch_relabel, _, _ = relabel_sequential(pred_batch)

            # check if segmentations were relabeled
            if not is_batch_relabeled:  # only one True is required
                is_batch_relabeled = not (
                    np.array_equal(true_batch, true_batch_relabel)
                    and np.array_equal(pred_batch, pred_batch_relabel)
                )

            o = ObjectMetrics(
                true_batch_relabel,
                pred_batch_relabel,
                cutoff1=self.cutoff1,
                cutoff2=self.cutoff2,
                force_event_links=self.force_event_links,
                is_3d=self.is_3d)

            all_object_metrics.append(o)

        if is_batch_relabeled:
            warnings.warn(
                'Provided data is being relabeled. Cell ids from metrics will not match '
                'cell ids in original data. Relabel your data prior to running the '
                'metrics package if you wish to maintain cell ids. ')

        # print the object report
        object_metrics = pd.DataFrame.from_records([
            o.to_dict() for o in all_object_metrics
        ])
        self.print_object_report(object_metrics)
        return object_metrics

    def summarize_object_metrics_df(self, df):
        correct_detections = int(df['correct_detections'].sum())
        n_true = int(df['n_true'].sum())
        n_pred = int(df['n_pred'].sum())

        _round = lambda x: round(x, self.ndigits)

        seg = df['seg'].mean()
        jaccard = df['jaccard'].mean()

        try:
            recall = correct_detections / n_true
        except ZeroDivisionError:
            recall = np.nan
        try:
            precision = correct_detections / n_pred
        except ZeroDivisionError:
            precision = 0

        errors = [
            'gained_detections',
            'missed_detections',
            'split',
            'merge',
            'catastrophe',
        ]

        bad_detections = [
            'gained_det_from_split',
            'missed_det_from_merge',
            'true_det_in_catastrophe',
            'pred_det_in_catastrophe',
        ]

        summary = {
            'correct_detections': correct_detections,
            'n_true': n_true,
            'n_pred': n_pred,
            'recall': _round(recall),
            'precision': _round(precision),
            'seg': _round(seg),
            'jaccard': _round(jaccard),
            'total_errors': 0,
        }
        # update bad detections
        for k in bad_detections:
            summary[k] = int(df[k].sum())
        # update error counts
        for k in errors:
            count = int(df[k].sum())
            summary[k] = count
            summary['total_errors'] += count
        return summary

    def print_object_report(self, object_metrics):
        """Print neat report of object based statistics

        Args:
            object_metrics (pd.DataFrame): DataFrame of all calculated metrics
        """
        summary = self.summarize_object_metrics_df(object_metrics)
        errors = [
            'gained_detections',
            'missed_detections',
            'split',
            'merge',
            'catastrophe'
        ]

        bad_detections = [
            'gained_det_from_split',
            'missed_det_from_merge',
            'true_det_in_catastrophe',
            'pred_det_in_catastrophe',
        ]

        print('\n____________Object-based statistics____________\n')
        print('Number of true cells:\t\t', summary['n_true'])
        print('Number of predicted cells:\t', summary['n_pred'])

        print('\nCorrect detections:  {}\tRecall: {}%'.format(
            summary['correct_detections'], summary['recall']))

        print('Incorrect detections: {}\tPrecision: {}%'.format(
            summary['n_pred'] - summary['correct_detections'],
            summary['precision']))

        print('\n')
        for k in errors:
            v = summary[k]
            name = k.replace('_', ' ').capitalize()
            if not name.endswith('s'):
                name += 's'

            try:
                err_fraction = v / summary['total_errors']
            except ZeroDivisionError:
                err_fraction = 0

            print('{name}: {val}{tab}Perc Error {percent}%'.format(
                name=name, val=v,
                percent=round(100 * err_fraction, self.ndigits),
                tab='\t' * (1 if ' ' in name else 2)))

        for k in bad_detections:
            name = k.replace('_', ' ').capitalize().replace(' det ', ' detections')
            print('{name}: {val}'.format(name=name, val=summary[k]))

        print('SEG:', round(summary['seg'], self.ndigits), '\n')

        print('Average Pixel IOU (Jaccard Index):',
              round(summary['jaccard'], self.ndigits), '\n')

    def run_all(self, y_true, y_pred, axis=-1):
        object_metrics = self.calc_object_stats(y_true, y_pred)
        pixel_metrics = self.calc_pixel_stats(y_true, y_pred, axis=axis)

        object_list = self.df_to_dict(object_metrics, stat_type='object')
        all_output = object_list + pixel_metrics
        self.save_to_json(all_output)

    def save_to_json(self, L):
        """Save list of dictionaries to json file with file metadata

        Args:
            L (list): List of metric dictionaries
        """
        todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
        outname = os.path.join(
            self.outdir, '{}_{}.json'.format(self.model_name, todays_date))

        # Configure final output
        D = {}

        # Record metadata
        D['metadata'] = dict(
            model_name=self.model_name,
            date=todays_date,
            notes=self.json_notes
        )

        # Record metrics
        D['metrics'] = L

        with open(outname, 'w') as outfile:
            json.dump(D, outfile)

        logging.info('Saved to {}'.format(outname))


def split_stack(arr, batch, n_split1, axis1, n_split2, axis2):
    """Crops an array in the width and height dimensions to produce
    a stack of smaller arrays

    Args:
        arr (numpy.array): Array to be split with at least 2 dimensions
        batch (bool): True if the zeroth dimension of arr is a batch or
            frame dimension
        n_split1 (int): Number of sections to produce from the first split axis
            Must be able to divide arr.shape[axis1] evenly by n_split1
        axis1 (int): Axis on which to perform first split
        n_split2 (int): Number of sections to produce from the second split axis
            Must be able to divide arr.shape[axis2] evenly by n_split2
        axis2 (int): Axis on which to perform first split

    Returns:
        numpy.array: Array after dual splitting with frames in the zeroth dimension

    Raises:
        ValueError: arr.shape[axis] must be evenly divisible by n_split
            for both the first and second split

    Examples:
        >>> from deepcell import metrics
        >>> from numpy import np
        >>> arr = np.ones((10, 100, 100, 1))
        >>> out = metrics.split_stack(arr, True, 10, 1, 10, 2)
        >>> out.shape
        (1000, 10, 10, 1)
        >>> arr = np.ones((100, 100, 1))
        >>> out = metrics.split_stack(arr, False, 10, 1, 10, 2)
        >>> out.shape
        (100, 10, 10, 1)
    """
    # Check that n_split will divide equally
    if ((arr.shape[axis1] % n_split1) != 0) | ((arr.shape[axis2] % n_split2) != 0):
        raise ValueError(
            'arr.shape[axis] must be evenly divisible by n_split'
            'for both the first and second split')

    split1 = np.split(arr, n_split1, axis=axis1)

    # If batch dimension doesn't exist, create and adjust axis2
    if batch is False:
        split1con = np.stack(split1)
        axis2 += 1
    else:
        split1con = np.concatenate(split1, axis=0)

    split2 = np.split(split1con, n_split2, axis=axis2)
    split2con = np.concatenate(split2, axis=0)

    return split2con


def match_nodes(y_true, y_pred):
    """Loads all data that matches each pattern and compares the graphs.

    Args:
        y_true (numpy.array): ground truth array with all cells labeled uniquely.
        y_pred (numpy.array): data array to match to unique.

    Returns:
        numpy.array: IoU of ground truth cells and predicted cells.
    """
    num_frames = y_true.shape[0]
    # TODO: does max make the shape bigger than necessary?
    iou = np.zeros((num_frames, np.max(y_true) + 1, np.max(y_pred) + 1))

    # Compute IOUs only when neccesary
    # If bboxs for true and pred do not overlap with each other, the assignment
    # is immediate. Otherwise use pixelwise IOU to determine which cell is which

    # Regionprops expects one frame at a time
    for frame in range(num_frames):
        gt_frame = y_true[frame]
        res_frame = y_pred[frame]

        gt_props = regionprops(np.squeeze(gt_frame.astype('int')))
        gt_boxes = [np.array(gt_prop.bbox) for gt_prop in gt_props]
        gt_boxes = np.array(gt_boxes).astype('double')
        gt_box_labels = [int(gt_prop.label) for gt_prop in gt_props]

        res_props = regionprops(np.squeeze(res_frame.astype('int')))
        res_boxes = [np.array(res_prop.bbox) for res_prop in res_props]
        res_boxes = np.array(res_boxes).astype('double')
        res_box_labels = [int(res_prop.label) for res_prop in res_props]

        # has the form [gt_bbox, res_bbox]
        overlaps = compute_overlap(gt_boxes, res_boxes)

        # Find the bboxes that have overlap at all
        # (ind_ corresponds to box number - starting at 0)
        ind_gt, ind_res = np.nonzero(overlaps)

        # frame_ious = np.zeros(overlaps.shape)
        for index in range(ind_gt.shape[0]):
            iou_gt_idx = gt_box_labels[ind_gt[index]]
            iou_res_idx = res_box_labels[ind_res[index]]
            intersection = np.logical_and(
                gt_frame == iou_gt_idx, res_frame == iou_res_idx)
            union = np.logical_or(
                gt_frame == iou_gt_idx, res_frame == iou_res_idx)
            iou[frame, iou_gt_idx, iou_res_idx] = intersection.sum() / union.sum()

    return iou
