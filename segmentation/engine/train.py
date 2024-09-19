import os
import yaml
import torch
import argparse

import numpy as np
import random

from pytorch_lightning import Trainer 
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from engine.network import Segmentor
from dataset.dataset import get_dataloader
from engine.test import run_final_report

from torchmetrics.functional import accuracy, precision, recall, auroc

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")

def train_process(config):
    is_aug = False
    if "is_aug" in config["network_structure"].keys():
        is_aug = config["network_structure"]["is_aug"]
    
    os.makedirs(config["training_setting"]["result_dir"], exist_ok=True)
    os.makedirs(config["training_setting"]["model_dir"].rsplit("/", 1)[0], exist_ok=True)

    print(f"Use Diffusion Data augmentation: {is_aug}")
    model = Segmentor(model_name=config["network_structure"]["model_name"],
                      model_config=config["network_structure"]["model_config"],
                      is_aug=is_aug, max_epochs=config["training_setting"]["epoch"])
    trainer = Trainer(devices=1, accelerator="gpu",
                      max_epochs=config["training_setting"]["epoch"], callbacks=[PrintCallback(),
                                                             ModelCheckpoint(dirpath=config["training_setting"]["model_dir"].rsplit('/', 1)[0],
                                                                             save_top_k=1,
                                                                             monitor='quick_dice',
                                                                             mode='max',
                                                                             filename=config["training_setting"]["model_dir"].rsplit('/', 1)[1].split('.')[0]+"_best")])
    train_dl, eval_dl, test_dl = get_dataloader(**config["dataset"], is_aug=is_aug)
    trainer.fit(model=model,
                train_dataloaders=train_dl,
                val_dataloaders=eval_dl)
    trainer.save_checkpoint(config["training_setting"]["model_dir"])
    
    # Load the best
    print("load config from ", config["training_setting"]["model_dir"].replace(".ckpt", "_best.ckpt"))
    model = Segmentor.load_from_checkpoint(config["training_setting"]["model_dir"].replace(".ckpt", "_best.ckpt"))
    model.eval()
    # trainer.test(model, test_dl)
    run_final_report(model=model, test_dl=test_dl, config=config)

def get_config():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Input the config directory")

    # Add arguments
    parser.add_argument('--config', type=str, default='./config',
                        help='Directory where the config is stored (default: ./config)')

    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise NotADirectoryError(f"Error: config directory '{args.config}' not found!")
    else:
        print(f"Dataset directory: '{args.config}'")

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    return config

def set_seeds(worker_seed):
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    set_seeds(0)
    config = get_config()
    print(config)
    train_process(config)
