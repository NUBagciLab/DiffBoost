import json
import os.path as osp
from collections import OrderedDict, defaultdict

import torch
import pandas as pd
import numpy as np
from skimage import transform
from sklearn.metrics import confusion_matrix
from evaluation.case_evaluate import evaluate_single_case, default_metrics
from utils import write_2d_image, write_3d_image, mkdir_if_missing

eps = 1e-5
def compute_dice(output, target):
    """Computes the Dice over the output and predict value

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes, hwd).
        target (torch.LongTensor): ground truth labels with shape (batch_size, num_classes, hwd).

    Returns:
        dice coefficient.
    """

    output_onehot = torch.zeros_like(output)
    output_onehot.scatter_(dim=1, index=torch.argmax(output, dim=1, keepdim=True), value=1)
    output_onehot = torch.flatten(output_onehot, start_dim=2)
    
    label_onehot = torch.zeros_like(output)
    label_onehot.scatter_(dim=1, index=target.to(torch.long), value=1)
    label_onehot = torch.flatten(label_onehot, start_dim=2)

    dice_value = (2*torch.sum(output_onehot*label_onehot, dim=[0, 2])+eps) / (torch.sum((label_onehot + output_onehot), dim=[0, 2]) + eps)
    return dice_value


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class Segmentation(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self,  lab2cname, **kwargs):
        super().__init__()
        self._lab2cname = lab2cname
        self.num_classes  = len(self._lab2cname)

        self.average_dice = 0
        self.best_dice = 0
        self._dice_class = 0
        self._dice_list = []

    def reset(self):
        self._dice = 0
        self._dice_class = 0

    def process(self, mo, gt):
        dice_value = compute_dice(mo, gt)
        self._dice_list.append(dice_value.data.cpu().numpy())

    def evaluate(self):
        results = OrderedDict()
        dice = 100. * np.mean(np.stack(self._dice_list), axis=0)
        self._dice_class = dice
        self.average_dice = float(dice[1]) if self.num_classes==2 else float(np.mean(dice[1:]))
        self.best_dice = max(self.best_dice, self.average_dice)
        err = 100. - self.average_dice
        results['dice'] = self.average_dice
        results['error_rate'] = err
        results['best_dice'] = self.best_dice

        print(
            '=> result\n'
            '* current dice: {:.2f}\n'
            '* best dice: {:.2f}\n'
            '* error: {:.2f}'.format(self.average_dice, self.best_dice, err)
        )

        return results


class FinalSegmentation(EvaluatorBase):
    """Evaluator for classification.
    Need to add domain support here
    """

    def __init__(self, lab2cname, output_dir, data_shape="3D", metrics=default_metrics, **kwargs):
        super().__init__()
        self._lab2cname = lab2cname
        self.metrics = metrics
        self._data_shape = data_shape
        self.output_dir = output_dir
        self.output_dir_summary = osp.join(self.output_dir, "summary")
        self.num_classes  = len(self._lab2cname)

        # This evaluation should be based on domain
        self._mean_evaludation_dict = {}
        self._evaluation_dict = {}
        self.exp_distance = "Distance"

        self.writter = self.get_writer()

    def reset(self):
        self._mean_evaludation_dict = {}
        self._evaluation_dict = {}

    def process(self, mo, gt, case_meta):
        case_name = case_meta['case_name']
        case_spacing = case_meta['spacing']
        case_orgsize = case_meta['org_size']
        predict_np, label_np = mo.data.cpu().numpy(), gt.data.cpu().numpy()
        predict_np, label_np = np.argmax(predict_np, axis=1), label_np[:, 0]

        # For squeezing the batch size direction if needed
        if predict_np.shape[0] == 1:
            predict_np, label_np = predict_np[0], label_np[0]
        # print('Before', predict_np.shape)
        predict_np, label_np = predict_np.astype(np.int64), label_np.astype(np.int64)
        # predict_np, label_np = (predict_np>0.5).astype(np.int64), (label_np>0.5).astype(np.int64)
        predict_np, label_np = transform.resize(predict_np, case_orgsize, order=0), transform.resize(label_np, case_orgsize, order=0)
        # print(np.sum(predict_np==1), np.sum(label_np==1), label_np.shape)
        evaluation_value = evaluate_single_case(predict_np, label_np,
                                                case_name=case_name, voxel_spacing=case_spacing,
                                                labels=self._lab2cname, metric_list=self.metrics)
        temp_result_dict = {case_name: evaluation_value}
        # print(temp_result_dict)
        self._evaluation_dict.update(temp_result_dict)

        ## write image here
        predict_folder = osp.join(self.output_dir, "out_image", "prediction")
        label_folder = osp.join(self.output_dir, "out_image", "label")
        mkdir_if_missing(predict_folder)
        mkdir_if_missing(label_folder)
        self.writter(predict_np, out_dir=predict_folder, 
                     case_name=case_name, meta_info=case_meta)
        self.writter(label_np, out_dir=label_folder, 
                     case_name=case_name, meta_info=case_meta)
    
    def get_writer(self):
        ### use data modality to distinguish
        if self._data_shape == "3D":
            return write_3d_image
        else:
            return write_2d_image

    def evaluate(self, extra_name:str=''):
        mkdir_if_missing(self.output_dir_summary)

        results = {}
        for label in list(self._lab2cname.values()):
            self._mean_evaludation_dict[label] = {}
            results[label] = {}
            for metric in self.metrics:
                temp_metric_list = [item[label][metric] for item in (self._evaluation_dict.values())]
                temp_mean = np.mean(temp_metric_list)
                temp_std = np.std(temp_metric_list)
                self._mean_evaludation_dict[label][metric] = {}
                self._mean_evaludation_dict[label][metric]["mean"] = temp_mean
                self._mean_evaludation_dict[label][metric]["std"] = temp_std
                if self.exp_distance in metric:
                    results[label][metric] = f"{np.round(temp_mean, 2)} " + "\u00B1" + f" {np.round(temp_std, 2)}"
                else:
                    results[label][metric] = f"{np.round(temp_mean, 4)} " + "\u00B1" + f" {np.round(temp_std, 4)}"

        pf = pd.DataFrame.from_dict(results, orient='index')
        print('=> result\n', pf)

        ### save the summary result
        print(self.output_dir_summary)
        pf.to_csv(osp.join(self.output_dir_summary, f'summary_result{extra_name}.csv'))
        with open(osp.join(self.output_dir_summary, f'detail_result{extra_name}.json'), 'w') as f:
            json.dump({"case_level": self._evaluation_dict,
                        "mean_level": self._mean_evaludation_dict}, f, indent=4)
