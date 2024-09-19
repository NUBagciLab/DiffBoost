"""
For full segmentation evaluation on volume level (3D) or image level (2D)
Regardless of training method

- Including the Evaluation Metric

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

"""

'''
Here to define the inference process for the given folder
-> Preprocessing:
Including->
    1. cropping, spacing unifying, intensity normalization
    2. prediction using the loaded network
    3. transfer back the cropping and spacing unifying
'''
import numpy as np
import SimpleITK as sitk
import functools

from typing import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *

from evaluation.metrics import ConfusionMatrix, ALL_METRICS


default_metrics = [
        "Dice",
        "Jaccard",
        "Precision",
        "Recall",
        "Accuracy",
        "Hausdorff Distance 95",
        "Avg. Symmetric Surface Distance"
    ]

def evaluate_single_case(predict_np, label_np, case_name, 
                         voxel_spacing, labels, metric_list=default_metrics):

    final_out_results = OrderedDict()
    try:
        confusion_matrix = ConfusionMatrix()
        for label, name in labels.items():
            out_results = OrderedDict()
            confusion_matrix.set_test((predict_np==int(label)).astype(np.int64))
            confusion_matrix.set_reference((label_np==int(label)).astype(np.int64))
            confusion_matrix.compute()

            for metric_name in metric_list:
                out_results[metric_name] = ALL_METRICS[metric_name](confusion_matrix,
                                                                    voxel_spacing=voxel_spacing)
            final_out_results[name] = out_results
    except Exception as e:
        print("Error evaluation: ", case_name, "due to ", e)

    final_out_results['case_name'] = case_name
    return final_out_results
