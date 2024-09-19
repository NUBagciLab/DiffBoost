import os
import yaml
import torch
import argparse

from pytorch_lightning import Trainer 
from pytorch_lightning.callbacks import Callback
from monai.inferers import SlidingWindowInferer
from engine.network import Segmentor
from evaluation.evaluator import FinalSegmentation
from dataset.dataset import get_dataloader


def run_final_report(model, test_dl, config):
    final_evaluator = FinalSegmentation(lab2cname=config['evaluation']["_lab2cname"],
                                        data_shape=config['dataset']['data_type'],
                                        output_dir=config['training_setting']['result_dir'])
    infer = SlidingWindowInferer(roi_size=config['dataset']["patch_size"], sw_batch_size=config['dataset']["batch_size"], overlap=0.5)
    final_evaluator.reset()

    for batch_idx, batch in enumerate(test_dl):
        input, label, meta = batch["data"].to(model.device), batch["seg"].to(model.device), batch["meta"][0]
        # print(f"input shape: {input.shape}, label shape: {label.shape}")
        output = infer(input, model)
        final_evaluator.process(output, label, meta)

    final_evaluator.evaluate(extra_name=config['evaluation']['extra_name'])


def eval_process(config, saved_weight_dir):
    model = Segmentor.load_from_checkpoint(saved_weight_dir).to(torch.device("cuda"))
    _, _, test_dl = get_dataloader(**config["dataset"])
    model.eval()
    run_final_report(model=model,
                     test_dl=test_dl,
                     config=config)

def get_config():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Input the pretrained weight directory")

    # Add arguments
    parser.add_argument('--config', type=str, default='./config',
                        help='Directory where the config is stored (default: ./config)')
    parser.add_argument('--fold', type=int, help='training fold')
    parser.add_argument('--weight_dir', type=str, 
                        help='Directory where the model weight is stored (default: ./config)')
    parser.add_argument('--result_dir', type=str, 
                        help='Directory where the result is stored (default: ./config)')

    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise NotADirectoryError(f"Error: config directory '{args.config}' not found!")
    else:
        print(f"Dataset directory: '{args.config}'")

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    config['training_setting']['result_dir'] = args.result_dir
    config['dataset']['fold'] = args.fold
    return config, args.weight_dir


if __name__ == '__main__':
    config, weight_dir = get_config()
    print(config, weight_dir)
    eval_process(config, weight_dir)
