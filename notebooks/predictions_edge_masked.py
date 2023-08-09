#%%
import sys
sys.path.append(".")
import cv2
import pytorch_lightning as pl
from torch import random
# from dataloader.segmentation_data_module import DataModule
# from models.segmentation_trainer import Model
from models.baseline import Model as Model
from omegaconf import DictConfig
from PIL import Image               # to load images
from IPython.display import display # to display images
import logging
from datasets.transformations import get_transforms
import torchvision.transforms.functional as tf
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets.segmentation_dataset import SegmentationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os
from utils.metrics import mIoU
from hydra.experimental import initialize_config_dir, compose
logging.getLogger("lightning").setLevel(logging.WARNING)
pl.seed_everything(42)

#%%

initialize_config_dir(config_dir="/data/aCardace/iterativive_da/conf", job_name="bau")
cfg = compose(config_name="config")

#%%

# evaluate the model on a test set
device = "cuda:1"
model = Model.load_from_checkpoint(checkpoint_path=cfg.resume_checkpoint)
model.to(device)
model.eval()
_, val_transform = get_transforms(cfg)

val_ds_source = SegmentationDataset(
    cfg.dataset.data_dir,
    cfg.dataset.target_val_list,
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


# kernel = np.ones((11,11))

for num_image, (image, gt, image_path, label_path) in enumerate(tqdm(val_dl_source)):
    image = image.to(device)

    with torch.no_grad():
        logits = model(image, size=tuple(cfg.dataset.out_val_size))
    
    # edges = F.interpolate(edges, size=tuple(cfg.dataset.out_val_size), mode="bilinear", align_corners=True)
    
    prediction = F.softmax(logits, dim=1)
    confidence, predicted_label = torch.max(prediction, dim=1)
    
    predicted_label = predicted_label.squeeze()
    
    fname = image_path[0].split('/')[-1]
    
    dest_path_label = os.path.join("/data/aCardace/iterativive_da/qualitatives/UDAS/predictions_ST_distillation", fname)
    cv2.imwrite(dest_path_label, predicted_label.cpu().numpy().squeeze())    
    # break
# %%
