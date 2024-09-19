import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler
from torchvision.transforms import transforms

"""
Include the dataset supporting
2D Training -> 2D data image and slice of 3D data volume
3D Training -> 3D data volume
2D Evaluation-> 2D data image
3D Evaluation -> 3D data volume
"""
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from dataset.augmentation.data_augment import get_augmentation

class TrainDataset(Dataset):
    def __init__(self, data_dir:str, data_files:list, transform_name:str='Baseline', patch_size:tuple=(384, 384),
                 is_aug:bool=False, is_train:bool=True, aug_ratio:int=10, fold:int=1, keys=("data", "seg")):
        super().__init__()
        self.data_dir = data_dir
        self.data_files = data_files
        self.transform = get_augmentation(patch_size=patch_size, transform_name=transform_name)
        self.is_aug = is_aug
        self.is_train = is_train
        self.aug_ratio = aug_ratio
        self.fold = fold
        self.keys = keys
        self.dtype_dict ={key: torch.long if key=='seg' else torch.float for key in self.keys}

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        slice_name = self.data_files[index]
        out_dict = {}
        
        if self.is_aug and self.is_train:
            randchoice = np.random.randint(1, self.aug_ratio+1)
            aug_dir = os.path.join(self.data_dir, f"fold{self.fold}/batch_{randchoice}", slice_name)
            data_dir = os.path.join(self.data_dir, f"batch_0", slice_name)
            raw_temp_dict = np.load(data_dir)
            aug_temp_dict = np.load(aug_dir)

            raw_data = raw_temp_dict['data']
            aug_data = aug_temp_dict['data']
            combine_dict, aug_dict = {}, {}
            combine_dict['data'] = np.stack([raw_data, aug_data], axis=1)
            combine_dict['seg'] = np.expand_dims(raw_temp_dict['seg'], axis=0)
            combine_dict = self.transform(**combine_dict)
            out_dict['data'] = torch.from_numpy(combine_dict['data'][:, 0]).to(self.dtype_dict['data'])
            out_dict['seg'] = torch.from_numpy(combine_dict['seg'][0]).to(self.dtype_dict['seg'])
            aug_dict['data'] = torch.from_numpy(combine_dict['data'][:, 1]).to(self.dtype_dict['data'])
            aug_dict['seg'] = torch.from_numpy(combine_dict['seg'][0]).to(self.dtype_dict['seg'])
            
            for key in self.keys:
                out_dict['aug_'+key] = aug_dict[key]
        else:
            temp_dir = os.path.join(self.data_dir, f"batch_0", slice_name)
            out_dict = self.get_data(temp_dir)
        
        return out_dict

    def get_data(self, temp_dir):
        temp_data = np.load(temp_dir)
        out_dict = {}
        for key in self.keys:
            out_dict[key] = np.expand_dims(temp_data[key], axis=0)
        out_dict = self.transform(**out_dict)
        for key in self.keys:
            out_dict[key] = torch.from_numpy(out_dict[key][0]).to(self.dtype_dict[key])
        return out_dict


class EvalDataset(Dataset):
    def __init__(self, data_dir:str, data_files:list, keys=("data", "seg")):
        super().__init__()
        self.data_dir = data_dir
        self.data_files = data_files
        self.case_names = []
        with open(os.path.join(self.data_dir, 'meta.pickle'), 'rb') as f:
            meta_data = pickle.load(f)
        self.meta_data_maps = meta_data

        self.keys = keys
        self.dtype_dict ={key: torch.long if key=='seg' else torch.float for key in self.keys}

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_name = self.data_files[index]
        out_dict = {}
        temp_data = np.load(os.path.join(self.data_dir, f"batch_0", file_name))

        for key in self.keys:
            out_dict[key] = torch.from_numpy(temp_data[key]).to(self.dtype_dict[key])

        out_dict["meta"] = self.meta_data_maps["case_info"][file_name.rsplit('.', 1)[0]]
        return out_dict


class Eval3DDatasetWarpperFrom2D(Dataset):
    """Well this is not the style I like, looks dirty
    """
    def __init__(self, data_dir:str, data_files:list, keys=("data", "seg")):
        super().__init__()
        self.data_dir = data_dir
        self.data_files = data_files

        with open(os.path.join(self.data_dir, 'meta.pickle'), 'rb') as f:
            meta_data = pickle.load(f)
        self.meta_data_maps = meta_data

        self.case_names = list(set([file.rsplit('_', 1)[0] for file in data_files]))
        # print(self.case_names )
        self.keys = keys
        self.dtype_dict ={key: torch.long if key=='seg' else torch.float for key in self.keys}

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, index):
        case_name = self.case_names[index]
        out_dict = {}
        for key in self.keys:
            out_dict[key] = []
        for i in range(self.meta_data_maps["case_info"][case_name]["depth"]):
            temp_data = np.load(os.path.join(self.data_dir, "batch_0", case_name+"_slice{:03d}.npz".format(i)))
            for key in self.keys:
                out_dict[key].append(temp_data[key])

        for key in self.keys:
            out_dict[key] = torch.from_numpy(np.stack(out_dict[key])).to(self.dtype_dict[key])

        out_dict["meta"] = self.meta_data_maps["case_info"][case_name]
        return out_dict

def collect_3d_eval_fn(data, data_key = ("data", "seg")):
    out_dict = {}
    for key in data[0].keys():
        if key in data_key:
            out_dict[key] = torch.cat([item[key] for item in data], dim=0)
        else:
            out_dict[key] = [item[key] for item in data]
    return out_dict

def collect_2d_eval_fn(data, data_key = ("data", "seg")):
    out_dict = {}
    for key in data[0].keys():
        if key in data_key:
            out_dict[key] = torch.stack([item[key] for item in data], dim=0)
        else:
            out_dict[key] = [item[key] for item in data]
    return out_dict


def get_dataloader(path_dir, fold, batch_size, data_type:str="3D",
                   is_aug:bool=False, augment_ratio:int=10, transform:str="Baseline",
                   *args, **kwargs):
    with open(os.path.join(path_dir, 'meta.pickle'), 'rb') as f:
        meta_data = pickle.load(f)
    train_files, test_files = meta_data['kfold_split'][int(fold)]["train"], meta_data['kfold_split'][int(fold)]["test"]

    # print("test_files", test_files)
    train_dataset = TrainDataset(data_dir=path_dir, data_files=train_files, is_aug=is_aug, is_train=True,
                                 aug_ratio=augment_ratio, fold=fold, transform_name=transform)
    eval_dataset = TrainDataset(data_dir=path_dir, data_files=test_files, is_aug=is_aug, is_train=False,
                                 aug_ratio=augment_ratio, fold=fold, transform_name="Baseline")
    if data_type == "2D":
        test_dataset = EvalDataset(data_dir=path_dir, data_files=test_files)
        test_dl = DataLoader(dataset=test_dataset, batch_size=1, collate_fn=collect_2d_eval_fn)
    else:
        test_dataset = Eval3DDatasetWarpperFrom2D(data_dir=path_dir, data_files=test_files)
        test_dl = DataLoader(dataset=test_dataset, batch_size=1, collate_fn=collect_3d_eval_fn)

    train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    eval_dl = DataLoader(dataset=eval_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    return train_dl, eval_dl, test_dl

if __name__ == '__main__':
    train_dl, eval_dl, test_dl = get_dataloader(path_dir='/data/datasets/DiffusionMedAug/prostate/processed/2DSlice',
                                                fold=1,
                                                batch_size=32, augment_ratio=2)
    batch0 = next(iter(train_dl))
    print(batch0["data"].shape)
