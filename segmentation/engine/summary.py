####
# This code is used to summarize the training results.
# Previous results are stored in the folder "results/{datasets}".
# Calculate the average and standard deviation of the results across the folder.
####

import os
import yaml
import argparse
import numpy as np
import pandas as pd


def summary_single_methods(dataset, method, num_fold:int=5):
    file_list = [os.path.join('./results', dataset, method+f"_fold{i+1}.yaml") for i in range(num_fold)]

    results = {}
    for file in file_list:
        with open(file, 'r') as f:
            result = yaml.safe_load(f)
        
        for key in result.keys():
            if key not in results.keys():
                results[key] = []
            results[key].append(result[key])
    
    df = {}
    for key in results.keys():
        results[key] = 100*np.array(results[key])
        df[key] = f'{results[key].mean():.2f} ± {results[key].std():.2f}'
        # print(f'{key}: {results[key].mean():.3f} ± {results[key].std():.3f}')
    df = pd.DataFrame(df, index=[method])
    return df


def get_summary(dataset, methods=None):
    
    if methods is None:
        methods = [file.rsplit("_")[0] for file in os.listdir(f'./results/{dataset}') if file.endswith('.yaml')]
        methods = list(set(methods))

    df = []
    for method in methods:
        df.append(summary_single_methods(dataset, method))
    df = pd.concat(df, axis=0)
    # print the result in latex format in dataframes
    print(df)
    print(df.to_latex(index=True))
    df.to_csv(f'./results/{dataset}/summary.csv', index=True)

    return df


def get_parse():
    parser = argparse.ArgumentParser(description='Summarize the results')
    parser.add_argument('--dataset', type=str, default='acl', help='the dataset to be summarized')
    parser.add_argument('--methods', type=list, default=None, help='the dataset to be summarized')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parse()
    get_summary(args.dataset, args.methods)
