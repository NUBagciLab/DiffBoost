import os

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
from torchvision.transforms import transforms
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

edge_transforms= transforms.Compose([transforms.RandomRotation(degrees=10),])
                                     # transforms.RandomAffine(degrees=10),
                                     # transforms.RandomResizedCrop(size=256)])
with open('./aug_list.txt', 'r') as f:
    aug_list = f.readlines()

def augment_prompts_and_edges(edges, num_samples):
    edges = edge_transforms(edges)
    prompt_list = [aug_list[i%len(aug_list)] for i in range(num_samples)]
    return prompt_list, edges

def get_augment_data(model, sampler, img_dir, num_samples, prompt, fold, image_resolution=256):
    input_image = skimage.io.imread(img_dir)
    detect_resolution = image_resolution

    ddim_steps = 20
    strength = 1.
    scale = 9
    eta = 0.0
    seed = 1
    guess_mode = False

    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = apply_hed(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).to(device)
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        
        rand_prompt_list, control = augment_prompts_and_edges(control, num_samples)
        control = control.float() / 255.0
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

    x_out = 255*(0.5*x_samples + 0.5).cpu().numpy().clip(0, 255)
    x_out = np.transpose(x_out, (0, 2, 3, 1)).astype(np.uint8)

    for i in range(num_samples):
        out_dir, _, img_name= img_dir.rsplit('/', 2)
        out_dir = os.path.join(out_dir, f'fold_{fold}', f"batch_{i+1}")
        
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        skimage.io.imsave(fname=os.path.join(out_dir, img_name),
                          arr=x_out[i])


def sample_folder(model, sampler, folder_dir, num_samples, prompt, fold, image_resolution):
    img_list = [file for file in os.listdir(folder_dir)]
    exclude = [file for file in os.listdir(folder_dir.rsplit('/', 1)[0]+"/batch_1")]
    process_file = [file for file in img_list if file not in exclude]

    for img_name in process_file:
        get_augment_data(model, sampler, os.path.join(folder_dir, img_name),
                         num_samples, prompt, fold, image_resolution)

def sample_folder_withlabel(model, sampler, folder_dir, num_samples, fold, image_resolution):
    img_list = [file for file in os.listdir(folder_dir)]
    if os.path.isdir(folder_dir.rsplit('/', 1)[0]+f"fold_{fold}/batch_1"):
        exclude = [file for file in os.listdir(folder_dir.rsplit('/', 1)[0]+f"fold_{fold}/batch_1")]
        process_file = [file for file in img_list if file not in exclude]
    else:
        process_file = img_list

    prompts_dict = {}
    with open(os.path.join(folder_dir.rsplit('/', 1)[0], f'prompt/prompts_{fold}.json'), 'rt') as f:
        for line in f:
            item = json.loads(line)
            prompts_dict.update({(item['target'].rsplit('/', 1)[-1]): item['prompt']})

    final_list = [file for file in process_file if file in prompts_dict.keys()]
    for img_name in final_list:
        get_augment_data(model, sampler, os.path.join(folder_dir, img_name),
                         num_samples, prompts_dict[img_name], fold, image_resolution)

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
