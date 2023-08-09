#%%
import os
os.chdir("..")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from models.segmentation_trainer import Model
# from models.baseline import Model
from IPython.display import display
from datasets.transformations import get_transforms
import torchvision.transforms.functional as tf
from datasets.segmentation_dataset import SegmentationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from utils.metrics import mIoU
import torch.nn.functional as F
from hydra.experimental import initialize_config_dir, compose
from PIL import Image
from utils.vis_flow import flow_to_color 

pl.seed_everything(42)
avg_pool = nn.AdaptiveAvgPool2d((1,1))

#%%

initialize_config_dir(config_dir="/data/aCardace/iterativive_da/conf")
cfg = compose(config_name="config")

#%%

# setup model
target_scorer = mIoU(cfg.dataset.num_classes)
device = "cuda:0"
model = Model.load_from_checkpoint(checkpoint_path=cfg.resume_checkpoint)
model.to(device)
model.eval()
_, val_transform = get_transforms(cfg)

val_ds_target = SegmentationDataset(
    cfg.dataset.data_dir,
    cfg.dataset.target_train_list,
    mean=cfg.dataset.mean,
    std=cfg.dataset.std,
    transform=val_transform,
)

val_dl_target = DataLoader(
    val_ds_target,
    batch_size=cfg.val_batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=False,
)

val_ds_source = SegmentationDataset(
    cfg.dataset.data_dir,
    cfg.dataset.source_train_list,
    mean=cfg.dataset.mean,
    std=cfg.dataset.std,
    transform=val_transform,
)

val_dl_source = DataLoader(
    val_ds_source,
    batch_size=cfg.val_batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=False,
)
#%%

def MMD(x, y, kernel, device="cpu"):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
      

    return torch.mean(XX + YY - 2. * XY)
c = 512*4
#%%
target_fetures = torch.ones((len(val_ds_target),c))
for num_image, (image, label, image_path, label_path) in enumerate(tqdm(val_dl_target)):
    image = image.to(device)
    with torch.no_grad():
        logits, _, m2 = model.net(image)
        # logits = tf.resize(logits, tuple(cfg.dataset.out_val_size))

    # prediction = logits.cpu().numpy().squeeze()
    # prediction = prediction.transpose(1, 2, 0)
    # prediction = np.asarray(np.argmax(prediction, axis=2), dtype=np.uint8)
    # target_scorer.update(prediction.flatten(), label.numpy().flatten())
    m2 = avg_pool(m2).squeeze(3).squeeze(2)
    target_fetures[num_image, ...] = m2.cpu()
# target_scorer.print_score()
#%%
source_features = torch.ones((len(val_ds_target),c))
for num_image, (image, label, image_path, label_path) in enumerate(tqdm(val_dl_source)):
    if num_image >= len(val_ds_target):
        break
    image = image.to(device)
    with torch.no_grad():
        logits, _, m2 = model.net(image)
    m2 = avg_pool(m2).squeeze(3).squeeze(2)
    source_features[num_image, ...] = m2.cpu()

#%%
result = MMD(source_features[:2975, ...], target_fetures, kernel="rbf")
print(result)
# %%
