import os
import yaml
import torch
import monai
import pytorch_lightning  as pl

from monai import losses
from monai.networks.utils import normal_init
from evaluation.evaluator import compute_dice

from .resunet import ResUNet

class Segmentor(pl.LightningModule):
    def __init__(self, model_name:str='2DUNet', model_config:dict={'loss': "DiceCELoss"}, is_aug:bool=False, max_epochs:int=100):
        super().__init__()

        # normal_init(self.backbone, std=0.005)
        self.loss_fn = getattr(losses, model_config['loss'])(include_background=False, to_onehot_y=True, softmax=True)
        if "resolution" in model_config.keys():
            self.resolution = model_config.pop('resolution')
        if "alpha" in model_config.keys():
            self.alpha = model_config.pop('alpha')

        self.backbone = self.get_model(model_name=model_name, model_config=model_config)
        self.is_aug = is_aug
        self.max_epochs = max_epochs
        self.save_hyperparameters()
    
    def get_model(self, model_name:str='2DUNet', model_config:dict={}):

        with open('./configs/trainer/default_network_configs.yaml', 'r') as f:
            network_dict = yaml.safe_load(f)
        network_name = network_dict[model_name]['model']

        network_config  = network_dict[model_name]['network']
        for key in model_config.keys():
            if key != 'loss':
                network_config[key] = model_config[key]
        
        self.in_channels = network_config['in_channels']
        self.out_channels = network_config['out_channels']
        if model_name == "ResNet50UNet":
            basemodel:torch.nn.Module = ResUNet(in_channel=self.in_channels,
                                                num_classes=self.out_channels)
            return basemodel

        basemodel:torch.nn.Module = getattr(monai.networks.nets, network_name)(**network_config)
        
        return basemodel
    
    def forward(self, x):
        # print(f"max: {torch.max(x)}, min {torch.min(x)}")
        # Warp all pretrained model to fixed outcome number
        # This is not what we like, but for fixed outcome, it is what it is
        return self.backbone(x)
    
    def base_training_step(self, batch, *args, **kwargs):
        x, y = batch["data"].to(self.device), batch["seg"].to(self.device)
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss()
    
    def aug_training_step(self, batch, *args, **kwargs):
        x, y, x_aug, y_aug = batch["data"].to(self.device), batch["seg"].to(self.device), batch["aug_data"].to(self.device), batch["aug_seg"].to(self.device)
        y_pred = self.forward(x)
        y_aug_pred = self.forward(x_aug)
        # random_select = (torch.rand(size=x.shape, device=self.device) < 0.75).to(torch.float32)
        N, C, H, W = x.shape
        random_select = (torch.rand(size=(N, C, H//self.resolution, W//self.resolution), device=self.device) < self.alpha).to(torch.float32)
        random_select = torch.nn.functional.interpolate(random_select, size=(H, W), mode='nearest')
        # random_select = (torch.rand(size=(N, 1, 1, 1), device=self.device) < 0.75).to(torch.float32)
        # loss = 0.75*self.loss_fn(y_pred, y) + 0.25*self.loss_fn(y_aug_pred, y_aug)
        loss = self.loss_fn(y_pred*random_select+y_aug_pred*(1-random_select), 
                            (y*random_select+y_aug*(1-random_select)).to(torch.long))
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def training_step(self, batch, *args, **kwargs):
        if self.is_aug:
            return self.aug_training_step(batch, *args, **kwargs)
        else:
            return self.base_training_step(batch, *args, **kwargs)
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1e-3)
        # optim = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
        # scheduler = CosineAnnealingWarmupRestarts(optim, first_cycle_steps=self.max_epochs,
        #                                          max_lr=1e-3, min_lr=1e-6, warmup_steps=10)
        return optim

    def validation_step(self, batch, *args, **kwargs):
        x, y = batch["data"], batch["seg"]
        y_pred = self.forward(x)
        dice = compute_dice(y_pred, y)
        self.log("quick_dice", dice[1:].mean(), prog_bar=True)
    
    def test_step(self, batch, *args, **kwargs):
        x, y = batch["data"], batch["seg"]
        y_pred = self.forward(x)
        dice = compute_dice(y_pred, y)
        loss = self.loss_fn(y_pred, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("quick_dice", dice[1:].mean(), prog_bar=True)
    
    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
        return logits


if __name__ == '__main__':
    # unit test
    model_config = {'in_channels': 3, 'out_channels': 4}
    segnet = Segmentor(model_name='2DUNet', model_config=model_config)
    img = torch.randn((3, 3, 384, 384))
    print(segnet(img).shape)
