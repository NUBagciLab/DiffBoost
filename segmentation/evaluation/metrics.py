'''
Stole from nnUNet
'''
#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
from medpy import metric
from medpy.metric.binary import __surface_distances

def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)


class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.smooth = 1e-3
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        if test is not None:
            if not np.any(test):
                center = tuple([int(s//2) for s in test.shape])
                test[(center)] = True
        self.test = test
        self.reset()

    def set_reference(self, reference):
         ### To address the none-prediction issue
        # Just assert one pixel to maintain the surface distance calculation
        if reference is not None:
            if not np.any(reference):
                center = tuple([int(s//2) for s in reference.shape])
                reference[(center)] = True
        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = float(((self.test != 0) * (self.reference != 0)).sum()) + self.smooth
        self.fp = float(((self.test != 0) * (self.reference == 0)).sum()) + self.smooth
        self.tn = float(((self.test == 0) * (self.reference == 0)).sum()) + self.smooth
        self.fn = float(((self.test == 0) * (self.reference != 0)).sum()) + self.smooth
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def dice(confusion_matrix=None, **kwargs):
    """2TP / (2TP + FP + FN)"""


    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float(2. * tp / (2 * tp + fp + fn))


def jaccard(confusion_matrix=None, **kwargs):
    """TP / (TP + FP + FN)"""
    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float(tp / (tp + fp + fn))


def precision(confusion_matrix=None, **kwargs):
    """TP / (TP + FP)"""
    tp, fp, tn, fn = confusion_matrix.get_matrix()
    return float(tp / (tp + fp))


def sensitivity(confusion_matrix=None, **kwargs):
    """TP / (TP + FN)"""

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float(tp / (tp + fn))


def recall(confusion_matrix=None, **kwargs):
    """TP / (TP + FN)"""

    return sensitivity(confusion_matrix)


def specificity(confusion_matrix=None, **kwargs):
    """TN / (TN + FP)"""

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    return float(tn / (tn + fp))


def accuracy(confusion_matrix=None, **kwargs):
    """(TP + TN) / (TP + FP + FN + TN)"""

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float((tp + tn) / (tp + fp + tn + fn))


def fscore(confusion_matrix=None, beta=1., **kwargs):
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""

    precision_ = precision(confusion_matrix)
    recall_ = recall(confusion_matrix)

    return (1 + beta*beta) * precision_ * recall_ /\
        ((beta*beta * precision_) + recall_)


def false_positive_rate(confusion_matrix=None, **kwargs):
    """FP / (FP + TN)"""

    return 1 - specificity(confusion_matrix)


def false_omission_rate(confusion_matrix=None, **kwargs):
    """FN / (TN + FN)"""

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float(fn / (fn + tn))


def false_negative_rate(confusion_matrix=None, **kwargs):
    """FN / (TP + FN)"""

    return 1 - sensitivity(confusion_matrix)


def true_negative_rate(confusion_matrix=None, **kwargs):
    """TN / (TN + FP)"""

    return specificity(confusion_matrix)


def false_discovery_rate(confusion_matrix=None, **kwargs):
    """FP / (TP + FP)"""

    return 1 - precision(confusion_matrix)


def negative_predictive_value(confusion_matrix=None, **kwargs):
    """TN / (TN + FN)"""

    return 1 - false_omission_rate(confusion_matrix)


def total_positives_test(confusion_matrix=None, **kwargs):
    """TP + FP"""
    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fp


def total_negatives_test(confusion_matrix=None, **kwargs):
    """TN + FN"""
    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fn


def total_positives_reference(confusion_matrix=None, **kwargs):
    """TP + FN"""
    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fn


def total_negatives_reference(confusion_matrix=None, **kwargs):
    """TN + FP"""
    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fp


def hausdorff_distance(confusion_matrix=None, voxel_spacing=None, connectivity=1, **kwargs):
    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd(test, reference, voxel_spacing, connectivity)


def hausdorff_distance_95(confusion_matrix=None, voxel_spacing=None, connectivity=1, **kwargs):
    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd95(test, reference, voxel_spacing, connectivity)


def avg_surface_distance(confusion_matrix=None, voxel_spacing=None, connectivity=1, **kwargs):
    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.asd(test, reference, voxel_spacing, connectivity)


def avg_surface_distance_symmetric(confusion_matrix=None,voxel_spacing=None, connectivity=1, **kwargs):
    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.assd(test, reference, voxel_spacing, connectivity)


def normalized_surface_dice(a: np.ndarray, b: np.ndarray, threshold: float, spacing: tuple = None, connectivity=1):

    assert all([i == j for i, j in zip(a.shape, b.shape)]), "a and b must have the same shape. a.shape= %s, " \
                                                            "b.shape= %s" % (str(a.shape), str(b.shape))
    if spacing is None:
        spacing = tuple([1 for _ in range(len(a.shape))])
    a_to_b = __surface_distances(a, b, spacing, connectivity)
    b_to_a = __surface_distances(b, a, spacing, connectivity)

    numel_a = len(a_to_b)
    numel_b = len(b_to_a)

    tp_a = np.sum(a_to_b <= threshold) / numel_a
    tp_b = np.sum(b_to_a <= threshold) / numel_b

    fp = np.sum(a_to_b > threshold) / numel_a
    fn = np.sum(b_to_a > threshold) / numel_b

    dc = (tp_a + tp_b + 1e-8) / (tp_a + tp_b + fp + fn + 1e-8)  # 1e-8 just so that we don't get div by 0
    return dc

ALL_METRICS = {
    "False Positive Rate": false_positive_rate,
    "Dice": dice,
    "Jaccard": jaccard,
    "Hausdorff Distance": hausdorff_distance,
    "Hausdorff Distance 95": hausdorff_distance_95,
    "Precision": precision,
    "Recall": recall,
    "Avg. Symmetric Surface Distance": avg_surface_distance_symmetric,
    "Avg. Surface Distance": avg_surface_distance,
    "Accuracy": accuracy,
    "False Omission Rate": false_omission_rate,
    "Negative Predictive Value": negative_predictive_value,
    "False Negative Rate": false_negative_rate,
    "True Negative Rate": true_negative_rate,
    "False Discovery Rate": false_discovery_rate,
    "Total Positives Test": total_positives_test,
    "Total Negatives Test": total_negatives_test,
    "Total Positives Reference": total_positives_reference,
    "total Negatives Reference": total_negatives_reference,
    "Normalized Surface Dice":normalized_surface_dice
}
