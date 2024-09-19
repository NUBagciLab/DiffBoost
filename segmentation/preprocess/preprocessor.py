"""
Multidomain preprocessor for 2D, 3D data
"""

import os
import json
import pickle
import yaml
import functools as func
import numpy as np
import argparse

from multiprocessing.pool import Pool
from sklearn.model_selection import KFold

from utils import image_preprocessor_2d, image_preprocessor_3d_slice


TWO_DIM_DTYPE = ["png", "jpg"]
THREE_DIM_DTYPE = ["dicom", "nii", "nii.gz"]


class Preprocessor(object):
    """Dataset preprocessor using the dataset configuration file
    
    """
    NUM_THREAD = 8
    def __init__(self, dataset_config_file:str):
        self.data_config_file = dataset_config_file
        
        with open(dataset_config_file, 'r') as f:
            self.data_preprocess_config = yaml.safe_load(f)

        self.dataset_dir = self.data_preprocess_config["data_dir"]
        self.out_tag = self.data_preprocess_config["out_tag"]
        self.raw_dir = os.path.join(self.dataset_dir, "raw_data")
        self.prompt = self.data_preprocess_config["prompt"]
        
        self.processed_dir = os.path.join(self.dataset_dir, "processed", self.out_tag)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.file_type = self.data_preprocess_config["file_type"]
        if self.file_type in TWO_DIM_DTYPE:
            self.is_threeD_data = False
        elif self.file_type in THREE_DIM_DTYPE:
            self.is_threeD_data = True
        else:
            raise NotImplementedError(f"file type {self.file_type} is not supported")

        self.is_threeD_training = False
        if "threeD_training" in self.data_preprocess_config.keys():
            self.is_threeD_training = True
            assert "target_space" in self.data_preprocess_config.keys(), ValueError("When use threeD training, you should specify the space size")
            self.target_space = tuple(self.data_preprocess_config["target_space"])
        else:
            self.target_size = tuple(self.data_preprocess_config["target_size"])
        
        self.extract_region = False
        if "extract_region" in self.data_preprocess_config.keys():
            self.extract_region = self.data_preprocess_config["extract_region"]
            assert not (self.extract_region and self.is_threeD_training), "Not support 3D Training"

            self.seg_kwargs = {}
            if "num_seg" in self.data_preprocess_config.keys():
                self.seg_kwargs['n_seg_region'] = self.data_preprocess_config['num_seg']
            
            if "expand_pixels" in self.data_preprocess_config.keys():
                self.seg_kwargs['expand_pixels'] = self.data_preprocess_config['expand_pixels']
            
            if "split_ratio" in self.data_preprocess_config.keys():
                self.seg_kwargs['split_ratio'] = self.data_preprocess_config['split_ratio']

        self.fold_nums:int = 3
        if "fold_nums" in self.data_preprocess_config.keys():
            self.fold_nums = int(self.data_preprocess_config["fold_nums"])

    def generate_mapfiles(self):
        dataset_meta_dict = {}

        with open(os.path.join(self.raw_dir, "dataset.json"), 'r') as f:
            temp_dataset_json = json.load(f)
        
        temp_file = temp_dataset_json.pop("training")
        num_classes = len(temp_dataset_json['labels'])
        if "label" in temp_file[0].keys():
            temp_file_list = [{"image": os.path.abspath(os.path.join(self.raw_dir, item["image"])),
                                "label": os.path.abspath(os.path.join(self.raw_dir, item["label"])),
                                "num_classes": num_classes} for item in temp_file]
        else:
            temp_file_list = [{"image": os.path.abspath(os.path.join(self.raw_dir, item["image"])),
                                "num_classes": num_classes} for item in temp_file]
        
        os.makedirs(os.path.join(self.processed_dir, "batch_0"), exist_ok=True)
        temp_outdir_list = [os.path.abspath(os.path.join(self.processed_dir, "batch_0", item["image"].split('/')[-1].split('.')[0])) for item in temp_file]

        _ = temp_dataset_json.pop("test")
        dataset_meta_dict = temp_dataset_json
        return temp_file_list, temp_outdir_list, dataset_meta_dict
    
    def kfold_split(self, data_dict):
        """Split the whole dataset files according to the case name
        Ensure the files from one case not participate the train and test at the same time
        data_dict: the whole data dict case_name:files
        """
        splits = {}
        case_list = list(data_dict.keys())
        kf = KFold(n_splits=self.fold_nums)
        for i, (train_id, test_id) in enumerate(kf.split(case_list)):
            splits[i] = {}
            train_keys = list(np.array(case_list)[train_id])
            test_keys = list(np.array(case_list)[test_id])
            train_files = []
            for key in train_keys:
                train_files.extend(data_dict[key])
            test_files = []
            for key in test_keys:
                test_files.extend(data_dict[key])
            splits[i]['train'] = train_files
            splits[i]['test'] = test_files

        return splits

    def __call__(self):
        file_list, outdir_list, dataset_meta_dict = self.generate_mapfiles()

        if self.is_threeD_data:
            assert "num_slice" in self.data_preprocess_config.keys(), "You should define the slice number"
            num_slice = self.data_preprocess_config["num_slice"]
            map_func = func.partial(image_preprocessor_3d_slice, target_size=self.target_size, prompt=self.prompt,
                                                            clip_percent=(0.5, 99.5), num_slice=num_slice)
        else:
            map_func = func.partial(image_preprocessor_2d, target_size=self.target_size, prompt=self.prompt,
                                                               clip_percent=(0.5, 99.5), is_gray=True)
        
        """with Pool(processes=self.NUM_THREAD) as pool:
            meta_list = pool.starmap(map_func, zip(file_list, outdir_list))"""
        meta_list = [map_func(file, out_dir) for file, out_dir in zip(file_list, outdir_list)]
        
        meta_dict, data_dict= {}, {}
        for item in meta_list:
            data_dict[item['case_name']] = item["data"]
            meta_dict.update({item['case_name']:item})
        
        out_dict = {'case_info':meta_dict,
                    'dataset_info': dataset_meta_dict}
        out_dict['kfold_split'] = self.kfold_split(data_dict=data_dict)
        with open(os.path.join(self.processed_dir, "meta.pickle"), 'wb') as f:
            pickle.dump(out_dict, f)

        print("Finished preprocessing")

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default='', help='path to dataset config')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parse()
    propressor = Preprocessor(args.config_dir)
    propressor()
