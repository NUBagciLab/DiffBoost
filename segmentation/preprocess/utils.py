"""
Basic Utils function for the proprocessing
Including: 

file reader for 3d medical image and 2d medical image
resize for 3d or 2d image to target size
normalization for 3d or 2d image to [-1, 1] with clipping

"""

import os
import cv2
import numpy as np
from skimage import io as skio
from skimage import transform
import SimpleITK as sitk

from scipy.signal import convolve2d
from preprocess.annotator.hed import HEDdetector
from preprocess.annotator.util import resize_image, HWC3

detector = HEDdetector()
def get_boundary_slice(image_np):
    image_np = (image_np+1)/2*255
    image_np = np.stack([image_np]*3, axis=-1)
    detected_map = detector(image_np)
    return detected_map

def get_boundary_batch(image_np, batch_size=32):
    image_np = (image_np+1)/2*255
    image_np = np.stack([image_np]*3, axis=-1)
    n_samples = image_np.shape[0]
    for i in range(int(np.ceil(n_samples/batch_size))):
        detected_map = detector.batch_edge(image_np[i*batch_size:min((i+1)*batch_size, n_samples)])
        detected_map = np.transpose(detected_map, axes=(0, 2, 3, 1))
        detected_map =  np.squeeze(detected_map, axis=-1)
        if i == 0:
            detected_map_all = detected_map
        else:
            detected_map_all = np.concatenate([detected_map_all, detected_map], axis=0)
    # print("image edge shape", detected_map_all.shape)
    
    return detected_map_all

def get_2dmask_boundary(mask_np):
    convolution_kernel = np.array([[ -3-3j, 0-10j,  +3 -3j],
                                   [-10+0j, 0+ 0j, +10 +0j],
                                   [ -3+3j, 0+10j,  +3 +3j]])
    grad = convolve2d(mask_np, convolution_kernel, boundary='symm', mode='same')
    boundary = np.clip(np.absolute(grad), a_min=0, a_max=1)*255
    # print("label edge shape", boundary.shape)
    return boundary


def image_preprocessor_2d(file_dict, out_dir, target_size, prompt, clip_percent=(0.5, 99.5), is_gray=True):
    ### This one is to support the png, jpg ... 2d data tyle
    # file dict should be like {"image":image_dir, "label":label_dir}
    out_dict = {}
    if is_gray:
        image = np.expand_dims(skio.imread(file_dict["image"], as_gray=True).astype(np.float32), 0)
    else:
        image = np.transpose(skio.imread(file_dict["image"]), axes=(2, 0, 1)).astype(np.float32)
    target_size_image = (image.shape[0], *target_size)

    case_name, file_type = file_dict["image"].rsplit('/', 1)[-1].split('.', 1)
    meta_dict = {}
    meta_dict["case_name"] = case_name
    meta_dict["org_size"] = image.shape[1:]
    meta_dict["file_type"] = file_type
    meta_dict["target_size"] = target_size
    meta_dict["spacing"] = (1, 1)
    meta_dict["meta"] = {}
    meta_dict["data"] = [case_name+".npz"]

    image_resize = transform.resize(image, target_size_image, order=1)
    image_min, image_max = np.percentile(image_resize, q=clip_percent[0]), np.percentile(image_resize, q=clip_percent[1])
    image_resize = np.clip(image_resize, a_min=image_min, a_max=image_max)
    image_resize = 2*(image_resize - image_min)/(image_max - image_min) - 1
    # print(image_resize.shape)
    out_dict["data"] = image_resize.astype(np.float32)

    num_classes = file_dict['num_classes']
    meta_dict["meta"]['num_classes'] = num_classes
    if "label" in file_dict.keys():
        # For PNG you need to transfer them to corresponding label
        seg = skio.imread(file_dict["label"], as_gray=True).astype(np.float32) / 255
        seg = np.round(seg * (num_classes - 1)).astype(np.int64)
        seg_resize = transform.resize(seg, target_size, order=0, preserve_range=True)
        out_dict["seg"] = np.expand_dims(seg_resize, 0)
        out_dict.update({"edge": np.expand_dims(np.clip(get_boundary_slice(image_resize[0]) + \
                                         get_2dmask_boundary(seg_resize), 0, 255)/255, 0)})
        out_dict.update({"prompt": prompt})

    np.savez(out_dir+".npz", **out_dict)
    return meta_dict

def image_preprocessor_3d_slice(file_dict, out_dir, target_size, prompt, clip_percent=(0.5, 99.5), num_slice:int=1):
    ### This one is to support the nii, dicom ... 3d data tyle
    # file dict should be like {"image":image_dir, "label":label_dir}
    # note that the resize transform will be limited with xy plane

    # This one is designed for 2d network training with 3d data
    out_dict = {}
    image_org= sitk.ReadImage(file_dict["image"])
    image = sitk.GetArrayFromImage(image_org)
    target_size_image = (image.shape[0], *target_size)
    case_name, file_type = file_dict["image"].rsplit('/', 1)[-1].split('.', 1)
    meta_dict = {}
    meta_dict["case_name"] = case_name
    meta_dict["org_size"] = image.shape
    meta_dict["file_type"] = file_type
    meta_dict["target_size"] = target_size_image
    meta_dict["spacing"] = image_org.GetSpacing()[::-1]
    meta_dict["meta"] = {}
    meta_dict["data"] = []
    num_classes = file_dict['num_classes']

    for key in image_org.GetMetaDataKeys():
        meta_dict["meta"][key] = image_org.GetMetaData(key)
    # For the z dimension, the axis size should not be changed
    # print('before', image.max(), image.min(), image.shape, target_size_image)
    image_resize = transform.resize(image, target_size_image, order=1)
    image_min, image_max = np.percentile(image_resize, q=clip_percent[0], axis=(1, 2), keepdims=True), \
                            np.percentile(image_resize, q=clip_percent[1], axis=(1, 2), keepdims=True)
    
    # This clip is designed for CT image, which has a lot of -1024 value
    image_min = np.clip(image_min, a_min=-256, a_max=None)
    # print('after', image_resize.shape)
    image_resize = np.clip(image_resize, a_min=image_min, a_max=image_max)
    image_resize = 2*(image_resize - image_min)/(image_max - image_min) - 1
    out_dict["data"] = image_resize.astype(np.float32)
    if "label" in file_dict.keys():
        seg = sitk.GetArrayFromImage(sitk.ReadImage(file_dict["label"]))
        # This is just for some disgusting image issue
        seg[seg<0] = 0
        seg_resize = transform.resize(seg, target_size_image, order=0)
        out_dict["seg"] = seg_resize.astype(np.int64)
        positive_slice = list(np.nonzero(np.sum(seg_resize, axis=(1, 2)) > 20.)[0])
        len_pos = len(positive_slice)

    image_edge = get_boundary_batch(image_resize)
    # image_edge = np.clip(get_boundary_batch(image_resize) + \
    #                      get_boundary_batch(seg_resize), 0, 255)/255
    depth = image.shape[0]
    for slice_id, i in enumerate(positive_slice):
        slice_list = [max(min(i-num_slice//2+idx, depth-1), 0) for idx in range(num_slice)]
        out_dict_slice = {"data": image_resize[slice_list]}
        out_dict_slice.update({"seg": np.expand_dims(seg_resize[i], 0)})
        out_dict_slice.update({"edge": np.expand_dims(np.clip(image_edge[i] + \
                                                              get_2dmask_boundary(seg_resize[i]), 0, 255)/255, 0)})
        # out_dict_slice.update({"edge": np.clip(get_boundary(image_resize[i]) + \
        #                                        get_boundary(seg_resize[i]/(num_classes-1)), 0, 255)/255})
        out_dict_slice.update({"prompt": prompt})

        np.savez(out_dir+"_slice{:03d}.npz".format(slice_id), **out_dict_slice)
        meta_dict["data"].append(case_name+"_slice{:03d}.npz".format(slice_id))

    meta_dict["depth"] = len_pos
    return meta_dict


if __name__ =='__main__':
    file_dict = {'image':'/data/datasets/DiffusionMedAug/prostate/raw_data/imagesTr/prostate_00.nii.gz',
                 'label':'/data/datasets/DiffusionMedAug/prostate/raw_data/labelsTr/prostate_00.nii.gz',
                 'num_classes': 3}
    out_dir = '/data/datasets/DiffusionMedAug/prostate/processed/prostate_00.npz'
    _ = image_preprocessor_3d_slice(file_dict, out_dir, target_size=(384, 384), prompt="prostate, MRI", clip_percent=(0.5, 99.5))
