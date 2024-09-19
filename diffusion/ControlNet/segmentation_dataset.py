import os
import json
import pickle
import numpy as np

from skimage import transform
from torch.utils.data import Dataset


def random_rotate_and_scale(image, edge, rotate_angle_range, crop_factor_range):
    # Generate random angle and scale
    angle = np.random.uniform(*rotate_angle_range)
    height, width = image.shape[:2]

    # Randomly select the crop factor
    crop_factor = np.random.uniform(*crop_factor_range)
    # Calculate the new dimensions after cropping
    if crop_factor < 1:
        new_width = int(width * crop_factor)
        new_height = int(height * crop_factor)

        # Calculate padding if the crop factor is less than one
        padding_x, padding_y = 0, 0
        # Crop the image, edge
        x_offset = (width - new_width) // 2
        y_offset = (height - new_height) // 2
        image_cropped = image[y_offset-padding_y:y_offset + new_height+padding_y, x_offset-padding_x:x_offset + new_width+padding_x]
        edge_cropped = edge[y_offset-padding_y:y_offset + new_height+padding_y, x_offset-padding_x:x_offset + new_width+padding_x]

        # Perform scaling with output shape same as input shape\
    else:
        new_width = int(width * (2-crop_factor))
        new_height = int(height * (2-crop_factor))
        padding_x, padding_y = (width - new_width) // 2, (height - new_height) // 2
        image_cropped = transform.resize(image, (new_height, new_width), mode='constant', cval=-1)
        edge_cropped = transform.resize(edge, (new_height, new_width), mode='constant', cval=0)
        image_cropped = np.pad(image_cropped, ((padding_y, padding_y), (padding_x, padding_x)), mode='symmetric')
        edge_cropped = np.pad(edge_cropped, ((padding_y, padding_y), (padding_x, padding_x)), mode='constant', constant_values=0)

    image_cropped = transform.resize(image_cropped, (height, width), mode='constant', cval=-1)
    edge_cropped = transform.resize(edge_cropped, (height, width), mode='constant', cval=0)
    # Perform rotation without resizing
    image_rotated = transform.rotate(image_cropped, angle=angle, resize=False, mode='symmetric')
    edge_rotated = transform.rotate(edge_cropped, angle=angle, resize=False, mode='constant', cval=0)
    
    return image_rotated, edge_rotated


class MyDataset(Dataset):
    def __init__(self, dataset_dir, fold:int=0):
        self.dataset_dir = dataset_dir
        self.img_size = 384
        self.rotate_angle_range = (-20, 20)
        self.crop_factor_range = (0.75, 1.05)
        with open(os.path.join(self.dataset_dir, "meta.pickle"), "rb") as f:
            self.data_meta = pickle.load(f)
        self.data_files = self.data_meta['kfold_split'][int(fold)]["train"]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        slice_name = self.data_files[idx]
        temp_dir = os.path.join(self.dataset_dir, f"batch_0", slice_name)
        temp_data = np.load(temp_dir)

        source = temp_data['edge'][0]
        target = temp_data['data'][0]
        prompt = str(temp_data['prompt'])

        # Randomly rotate the source and target images.
        target, source = random_rotate_and_scale(target, source, self.rotate_angle_range, self.crop_factor_range)

        # Normalize target images to [-1, 1].
        source = np.stack([source]*3, axis=-1).astype(np.float32)
        target = np.stack([target]*3, axis=-1).astype(np.float32)
        target = np.clip(target, -1.0, 1.0)
        # print(source[:,:,0])
        # print(type(source), type(target), type(prompt))
        return dict(jpg=target, txt=prompt, hint=source)


if __name__ == "__main__":
    test_dataset = MyDataset("/data/datasets/DiffusionMedAug/prostate/processed/2DSliceEdge")
    _ = test_dataset[0]
