import os
import pickle

from share import *
import config
import cv2
import json
import einops
import numpy as np
import torch
import random
import argparse

import skimage
from skimage import transform
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

torch.cuda.empty_cache()
device = torch.device('cuda')

apply_hed = HEDdetector()

def get_model(model_weight_dir):
    config_dir='./models/cldm_v15.yaml'
    # model_weight_dir='./lightning_logs/version_0/checkpoints/epoch=9-step=159.ckpt'
    model = create_model(config_dir).cpu()
    model.load_state_dict(load_state_dict(model_weight_dir, location='cuda'))
    model = model.to(device)
    sampler = DDIMSampler(model)
    return model, sampler

from skimage import transform

def random_rotate_and_scale(image, edge, rotate_angle_range=(-20, 20), crop_factor_range=(0.75, 1.05)):
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
        image_cropped = transform.resize(image, (new_height, new_width), mode='constant', cval=0)
        edge_cropped = transform.resize(edge, (new_height, new_width), mode='constant', cval=0)
        image_cropped = np.pad(image_cropped, ((padding_y, padding_y), (padding_x, padding_x)), mode='constant', constant_values=0)
        edge_cropped = np.pad(edge_cropped, ((padding_y, padding_y), (padding_x, padding_x)), mode='constant', constant_values=0)

    image_cropped = transform.resize(image_cropped, (height, width), mode='constant', cval=0)
    edge_cropped = transform.resize(edge_cropped, (height, width), mode='constant', cval=0)
    image_rotated = transform.rotate(image_cropped, angle=angle, resize=False, mode='edge', cval=0)
    edge_rotated = transform.rotate(edge_cropped, angle=angle, resize=False, mode='edge', cval=0)
    
    return image_rotated, edge_rotated

with open('./aug_list.txt', 'r') as f:
    aug_list = f.readlines()

def augment_prompts_and_edges(mask, edges, num_samples):
    masks_list, edges_list = [], []
    for _ in range(num_samples):
        # temp_mask, temp_edge = random_rotate_and_scale(mask, edges)
        temp_mask, temp_edge = mask, edges
        masks_list.append(temp_mask.astype(np.int64))
        edges_list.append(temp_edge.astype(np.float32))
    prompt_list = [str(aug_list[i%len(aug_list)]).replace('\n', '') for i in range(num_samples)]
    return prompt_list, edges_list, masks_list

def get_augment_data(model, sampler, img_dir, num_samples, fold, image_resolution=384):

    ddim_steps = 20
    strength = 1.
    scale = 9
    eta = 0.0
    seed = 1
    guess_mode = False
    C = 3

    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    with torch.no_grad():
        # input_image = HWC3(input_image)
        # detected_map = apply_hed(resize_image(input_image, detect_resolution))
        # detected_map = HWC3(detected_map)
        # img = resize_image(input_image, image_resolution)
        temp_data = np.load(img_dir)

        source = temp_data['edge'][0]
        mask = temp_data['seg'][0]
        H, W = source.shape
        prompt = str(temp_data['prompt'])
        out_dict = {"seg": temp_data['seg'],
                    "prompt": prompt}
        rand_prompt_list, sources_list, masks_list = augment_prompts_and_edges(mask, source, num_samples)

        control = torch.stack([torch.from_numpy(source).to(device) for source in sources_list], dim=0)
        control = torch.stack([control]*C, dim=1).clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        
        cond = {"c_concat": [control], "c_crossattn": [0.045*model.get_learned_conditioning([prompt+","+a_prompt] * num_samples) + \
                                                       0.005*model.get_learned_conditioning(rand_prompt_list) + \
                                                       0.95*model.get_learned_conditioning(['gray, sketch'] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = sampler.sample(ddim_steps, num_samples,
                                                shape, cond, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)

    ### Finish the output file sample here

    x_out = x_samples.cpu().numpy().clip(-1, 1)
    x_out = np.transpose(x_out, (0, 2, 3, 1))

    for i in range(num_samples):
        out_dir, _, img_name= img_dir.rsplit('/', 2)
        out_dir = os.path.join(out_dir, f'fold{fold}', f"batch_{i+1}")
        out_dict["data"] = np.expand_dims(x_out[i,:,:,0], 0)
        out_dict["seg"] = np.expand_dims(masks_list[i], 0)
        out_dict["edge"] = np.expand_dims(sources_list[i], 0)
        
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        np.savez_compressed(os.path.join(out_dir, img_name),
                            **out_dict)


def sample_folder_withlabel(model, sampler, folder_dir, num_samples, fold, image_resolution):
    img_list = [file for file in os.listdir(folder_dir)]
    
    with open(os.path.join(folder_dir, "meta.pickle"), "rb") as f:
        data_meta = pickle.load(f)
    data_files = data_meta['kfold_split'][int(fold)]["train"]

    if os.path.isdir(folder_dir+f"/fold{fold}/batch_1"):
        exclude = [file for file in os.listdir(folder_dir+f"/fold{fold}/batch_1")]
        final_list = [file for file in data_files if file not in exclude]
    else:
        final_list = data_files

    for img_name in final_list:
        get_augment_data(model, sampler, os.path.join(folder_dir, "batch_0", img_name),
                         num_samples, fold, image_resolution)

def get_config():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Input the config directory")

    # Add arguments
    parser.add_argument('--folder_dir', type=str,
                        help='Directory where is the folder')

    parser.add_argument('--model_weight_dir', type=str,
                        help='Directory where is the prompts')
    
    parser.add_argument('--fold', type=int, 
                        help='fold for training')
    
    parser.add_argument('--num_samples', type=int, default=10,
                        help='number of sampler')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_config()
    folder_dir = args.folder_dir
    model_weight_dir = args.model_weight_dir
    fold = args.fold
    num_samples = args.num_samples
    model, sampler = get_model(model_weight_dir=model_weight_dir)
    sample_folder_withlabel(model, sampler, folder_dir, num_samples, fold, image_resolution=256)
    # sample_folder(folder_dir, num_samples, prompt, fold, image_resolution=256)
