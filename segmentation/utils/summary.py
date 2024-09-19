"""
Summary the evaluation result after the training
Generate the csv file and the latex table (For simple evaluation)
"""

import os
import json
import argparse
import copy
import pandas as pd
import numpy as np

out_metrics = ['Dice', 'Precision', 'Recall', 'Hausdorff Distance 95']


def generate_summary_interdomain(folder, method, metrics=out_metrics):
    folder_dir = os.path.join(folder, method)
    fold_list = [name for name in os.listdir(folder_dir) \
                    if not os.path.isfile(os.path.join(folder_dir, name))]
    # Plus one for avaerage evaludation
    results_dict = {}

    for fold in fold_list:
        temp_summary_dir = os.path.join(folder_dir, fold, "summary")
        with open(os.path.join(temp_summary_dir, "detail_result.json")) as f:
            temp_summary = json.load(f)['mean_level']

        results_dict[fold] = temp_summary

    label_names = list(temp_summary.keys())
    if "Background" in label_names:
        label_names.remove('Background')
    if "background" in label_names:
        label_names.remove('background')

    out_dict = {}

    for label_name in label_names:
        for metric in metrics:
            if "Distance" in metric:
                indent = 2
            else:
                indent = 4

            cross_fold_metric_mean = []
            cross_fold_metric_std = []
            for fold in fold_list:
                cross_fold_metric_mean.append(results_dict[fold][label_name][metric]["mean"])
                cross_fold_metric_std.append(results_dict[fold][label_name][metric]["std"])
            
            temp_mean, temp_std = np.mean(cross_fold_metric_mean), np.sqrt(np.mean(np.array(cross_fold_metric_std)**2))
            out_dict[label_name+metric] = [label_name, metric, 
                                    f"{np.round(temp_mean, indent)} " +"\u00B1" + f" {np.round(temp_std, indent)}"]

    index_list = ['name', 'metric', method]
    df = pd.DataFrame(out_dict, index=index_list)
    return df


def generate_summary(folder, method_list=None, metrics=out_metrics):
    if method_list is None:
        method_list = [name for name in os.listdir(folder) \
                        if not os.path.isfile(os.path.join(folder, name))]
    
    df_list = [generate_summary_interdomain(folder, method, metrics=metrics) for method in method_list]
    # print(df_list[0])
    final_df = pd.concat(df_list).drop_duplicates()
    # Get latex if needed
    latex_code = final_df.style.to_latex()
    # latex_code = latex_code.replace("\\\n", "\\ \hline\n")
    print(latex_code)
    return final_df

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='', help='path to result folder')
    parser.add_argument('--methods', '--names-list', nargs='+',default=None, help='method for the final result')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parse()
    folder = args.folder
    methods = args.methods
    print(generate_summary(folder, methods))