"""
Here to define the basic augmentation method for medical image segmentation

The augmentation will be based on 2D and using the nnunet augmentation format
"""
import warnings
import numpy as np

from batchgenerators.transforms.abstract_transforms import Compose, AbstractTransform
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform,SpatialTransform_2

from .nn_augment import nnunet_augmentation, spatial_aug


class MinMaxNormalization(AbstractTransform):
    """ Min-Max Normalization Method
    """
    def __init__(self, norm_range=(-1, 1), data_key="data"):
        self.data_key = data_key
        self.norm_range = norm_range

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
    
        data_dict[self.data_key] = (data - np.min(data)) / (np.max(data) - np.min(data))*\
                                    (self.norm_range[1]-self.norm_range[0]) + self.norm_range[0]

        return data_dict


def get_single_augmentation(patch_size, transform_name):
    transform = []

    if transform_name == "Baseline":
        pass
    elif transform_name == "RandomContrast":
        transform.append(ContrastAugmentationTransform((0.75, 1.25), preserve_range=True, per_channel=True))
    elif transform_name == "RandomGamma":
        transform.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True))
    elif transform_name == "RandomBrightness":
        transform.append(BrightnessTransform(mu=0, sigma=0.5, per_channel=True, p_per_sample=1))
    elif transform_name == "RandomNoise":
        transform.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=1))
    elif transform_name == "RandomResolution":
        transform.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True, p_per_channel=0.5))
    elif transform_name == "RandomMirror":
        transform.append(MirrorTransform(axes=(0, 1, 2), p_per_sample=0.75))
    elif transform_name == "RandomRotate":
        transform.append(SpatialTransform(patch_size, patch_center_dist_from_border=None, do_elastic_deform=False,
                                          do_rotation=True, angle_x=(-5 / 360. * 2. * np.pi, 5 / 360. * 2. * np.pi),
                                          angle_y=(-5 / 360. * 2. * np.pi, 5 / 360. * 2. * np.pi), do_scale=False, random_crop=False, p_el_per_sample=0.5))
    elif transform_name == "RandomScale":
        transform.append(SpatialTransform(patch_size, patch_center_dist_from_border=None, do_elastic_deform=False,
                                          do_rotation=False, do_scale=True, scale=(0.95, 1.05), random_crop=False))
    elif transform_name == "MedDiffAug":
        pass
    else:
        warnings.warn(f"Augmentation method {transform_name} is not defined")
    # transform.append(MinMaxNormalization(norm_range=(-1, 1)))
    return transform

def get_augmentation(patch_size, transform_name:list):
    """
    :param patch_size:
    :param transform_name:
    :return:
    """
    transforms = []
    if transform_name not in ["Baseline", "MedDiffAug"]:
        transforms.extend(get_single_augmentation(patch_size, transform_name))
    elif transform_name == "DeepStack":
        transforms = nnunet_augmentation(patch_size, augment_key='2D')
    elif transform_name == "MedDiffAug_Plus":
        transforms = spatial_aug(patch_size, augment_key='2D')
    else:
        # transforms = nnunet_augmentation(patch_size, augment_key='2D')
        pass
    transforms.append(MinMaxNormalization(norm_range=(-1, 1)))
    return Compose(transforms)


if __name__ == "__main__":
    pass
