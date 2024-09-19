import json
import cv2
import numpy as np

from torch.utils.data import Dataset


def random_transform(source, target, rotation_angle_range, crop_factor_range):
    # Randomly select the rotation angle
    rotation_angle = np.random.uniform(rotation_angle_range[0], rotation_angle_range[1])
    
    # Rotate the source, target
    height, width = source.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
    source_rotated = cv2.warpAffine(source, rotation_matrix, (width, height))
    target_rotated = cv2.warpAffine(target, rotation_matrix, (width, height))

    # Randomly select the crop factor
    crop_factor = np.random.uniform(crop_factor_range[0], crop_factor_range[1])

    # Calculate the new dimensions after cropping
    new_width = int(width * crop_factor)
    new_height = int(height * crop_factor)

    # Calculate padding if the crop factor is larger than one
    if crop_factor > 1:
        padding_x = int((new_width - width) // 2)
        padding_y = int((new_height - height) // 2)
    else:
        padding_x, padding_y = 0, 0

    # Crop the source, target
    x_offset = (width - new_width) // 2
    y_offset = (height - new_height) // 2
    source_cropped = source_rotated[y_offset-padding_y:y_offset + new_height+padding_y, x_offset-padding_x:x_offset + new_width+padding_x]
    target_cropped = target_rotated[y_offset-padding_y:y_offset + new_height+padding_y, x_offset-padding_x:x_offset + new_width+padding_x]

    # Resize the source, target back to the original dimensions
    source_resized = cv2.resize(source_cropped, (width, height))
    target_resized = cv2.resize(target_cropped, (width, height))

    return source_resized, target_resized


class MyDataset(Dataset):
    def __init__(self, dataset_file):
        self.data = []
        self.img_size = 256
        self.rotate_angle_range = (-45, 45)
        self.crop_factor_range = (0.75, 1.25)
        with open(dataset_file, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        source = cv2.resize(source, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # Randomly rotate the source and target images.
        # source, target = random_transform(source, target, self.rotate_angle_range, self.crop_factor_range)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        print(source.shape, target.shape, prompt)
        return dict(jpg=target, txt=prompt, hint=source)


if __name__ == "__main__":
    test_dataset = MyDataset(dataset_file='/data/datasets/DiffusionMedAug/acl/images/prompt/prompts_1.json')
    _ = test_dataset[0]

