from share import *
import os
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from segmentation_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


def train(pretrained_path, dataset_dir, fold, out_dir, epochs=10):
    # resume_path = './lightning_logs/version_1/checkpoints/epoch=5-step=42341.ckpt'
    batch_size = 32
    logger_freq = 300
    learning_rate = 1e-6
    sd_locked = True
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.

    model = create_model('./models/cldm_v15.yaml').cpu()
    if os.path.exists(out_dir):
        print(f"Loading checkpoint from {out_dir}")
        model.load_state_dict(load_state_dict(out_dir, location='cpu'))
    else:
        model.load_state_dict(load_state_dict(pretrained_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    dataset = MyDataset(dataset_dir, fold=fold)
    dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(accelerator="gpu", devices=1,
                         precision=32, callbacks=[logger],
                         max_epochs=epochs, default_root_dir=out_dir.rsplit("/", 1)[0])

    # Train!
    trainer.fit(model, dataloader)
    trainer.save_checkpoint(filepath=out_dir)

def get_config():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Input the config directory")

    # Add arguments
    parser.add_argument('--pretrained_dir', type=str, default='./lightning_logs/radimagenet/checkpoints/model.ckpt',
                        help='Directory where is the pretrained weight')

    parser.add_argument('--dataset_dir', type=str, default='/data/datasets/DiffusionMedAug/meniscus/images',
                        help='Directory where is the prompts')
    
    parser.add_argument('--fold', type=int, default=0,
                        help='which fold to train')
    
    parser.add_argument('--epochs', type=int, default=10,
                        help='epochs for training')
    
    parser.add_argument('--out_dir', type=str, default='/data/datasets/DiffusionMedAug/saved_model/model.ckpt',
                        help='Directory for saved checkpoint')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_config()
    train(args.pretrained_dir, args.dataset_dir, args.fold, args.out_dir, args.epochs)
