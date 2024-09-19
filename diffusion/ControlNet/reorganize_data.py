
import os
import fnmatch
import skimage
import cv2
import argparse
import numpy as np
import json
import pandas as pd
from multiprocessing.pool import Pool

from annotator.hed import HEDdetector
from annotator.util import resize_image, HWC3


def get_file_path(img_dir):
    match_pattern = ('.png','.jpeg','.jpg')
    match_dict = {
        'abd': 'abdomen',
        'af': 'ankle',
        'cbd': 'common bile duct',
        'gb': 'gallbladder',
        'ivc': 'inferior vena cava'
    }
    file_dir_summary = []
    file_prompt_summary = []

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(match_pattern):
                file_prompt = root.replace(img_dir+'/', '').replace("/", ",")
                for key, value in match_dict.items():
                    file_prompt = file_prompt.replace(','+key+',',
                                                    ','+value+',')
                file_prompt = file_prompt.replace("_", " ")
                file_dir_summary.append(os.path.join(root, file))
                file_prompt_summary.append(file_prompt)
    return file_dir_summary, file_prompt_summary

def get_file_path_from_dataset(path_dir, label_dir, base_prompts, match_dict=None):
    match_pattern = ('.png','.jpeg','.jpg')
    cases_id = pd.read_csv(os.path.join(path_dir, label_dir))
    name_list, label_list = list(cases_id['img_dir']), list(cases_id['label'])
    name_dict = {i.split('/')[-1] : v  for i, v in zip(name_list, label_list)}
    
    file_dir_summary = []
    file_prompt_summary = []
    img_dir  = os.path.join(path_dir, "images/batch_0")

    for file in name_dict.keys():
        if file.endswith(match_pattern):
            if match_dict is not None:
                    file_prompt = base_prompts + "," + match_dict[name_dict[file]]
            else:
                file_prompt = base_prompts + "," + name_dict[file]
            
            file_prompt = file_prompt.replace("_", " ")
            file_dir_summary.append(os.path.join(img_dir, file))
            file_prompt_summary.append(file_prompt)
    return file_dir_summary, file_prompt_summary

detector = HEDdetector()
def get_boundary(img_path, out_path):
    input_image = skimage.io.imread(img_path)
    detected_map = detector(input_image)
    detected_map = HWC3(detected_map)
    skimage.io.imsave(out_path, detected_map)


def get_edge_boundary(file_dirs, file_prompts, fold, out_mark='edge'):
    # file_dirs, file_prompts = get_file_path(img_dir=img_dir)
    img_dir = file_dirs[0].rsplit('/', 1)[0]
    out_path = img_dir.rsplit('/', 1)[0]+'/'+out_mark
    replace_mark = img_dir.rsplit('/', 1)[1]

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    out_files = []
    for file in file_dirs:
        out_temp_file = file.replace(replace_mark, out_mark)
        out_temp_dir = out_temp_file.rsplit('/', 1)[0]
        if not os.path.isdir(out_temp_dir):
            os.makedirs(out_temp_dir)
        
        out_files.append(out_temp_file)
    
    """with Pool(processes=8) as pool:
        pool.starmap(get_boundary, zip(file_dirs, out_files))"""
    for file_dir, out_file in zip(file_dirs, out_files):
        if not os.path.isfile(out_file):
            get_boundary(file_dir, out_file)

    final_out = []
    for prompt, target, source in zip(file_prompts,
                                      file_dirs, out_files):
        final_out.append({"prompt": prompt,
                          "target": target,
                          "source": source})
    
    with open(img_dir.rsplit('/', 1)[0]+f'/prompt/prompts_{fold+1}.json', 'w+') as f:
        os.mkdir(img_dir.rsplit('/', 1)[0]+f'/prompt', exist_ok=True)
        for dicts in final_out:
            json.dump(dicts, f)
            f.write("\n")

    print("Finished")

def get_config():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Input the config directory")

    # Add arguments
    parser.add_argument('--dataset', type=str,
                        help='dataset name')
    parser.add_argument('--base_prompt', type=str,
                        help='base_prompt')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    
    """
    match_dict = {'yes': "positive",
                  "no": "negative"}
    file_dirs, file_prompts = get_file_path_from_dataset(path_dir='/data/datasets/DiffusionMedAug/acl',
                                                         label_dir='splits/train_fold1.csv',
                                                         base_prompts='Anterior cruciate ligament, MRI',
                                                         match_dict=match_dict)"""
    
    args = get_config()
    dataset, base_prompt = args.dataset, args.base_prompt
    for fold in range(5):
        file_dirs, file_prompts = get_file_path_from_dataset(path_dir=f'/data/datasets/DiffusionMedAug/{dataset}',
                                                             label_dir=f'splits/train_fold{fold+1}.csv',
                                                             base_prompts=base_prompt)
        # print(file_dirs)
        get_edge_boundary(file_dirs=file_dirs,
                          file_prompts=file_prompts, fold=fold)
    """
    img_dir = '/data/datasets/RadImageNet/radiology_ai'
    file_dirs, file_prompts = get_file_path(img_dir=img_dir)
    get_edge_boundary(img_dir=img_dir,
                      file_dirs=file_dirs,
                      file_prompts=file_prompts)"""
