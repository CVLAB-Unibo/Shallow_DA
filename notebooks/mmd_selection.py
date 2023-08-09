#%%
import os

from numpy.lib.function_base import insert
os.chdir("..")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
# from models.segmentation_trainer import Model
from models.baseline import Model
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

c = 512*4
#%%
target_fetures = torch.ones((len(val_ds_target),c))
for num_image, (image, label, image_path, label_path) in enumerate(tqdm(val_dl_target)):
    image = image.to(device)
    with torch.no_grad():
        logits, _, m2 = model.net(image)
    m2 = avg_pool(m2).squeeze(3).squeeze(2)
    target_fetures[num_image, ...] = m2.cpu()

    # logits = tf.resize(logits, tuple(cfg.dataset.out_val_size))
    # prediction = logits.cpu().numpy().squeeze()
    # prediction = prediction.transpose(1, 2, 0)
    # prediction = np.asarray(np.argmax(prediction, axis=2), dtype=np.uint8)
    # target_scorer.update(prediction.flatten(), label.numpy().flatten())
    # break
# target_scorer.print_score()
#%%
source_image_path = {}
source_paths = {}
source_features = torch.ones((len(val_dl_source),c))
for num_image, (image, label, image_path, label_path) in enumerate(tqdm(val_dl_source)):
    source_paths[num_image] = label_path
    source_image_path[num_image] = image_path

    image = image.to(device)
    with torch.no_grad():
        logits, _, m2 = model.net(image)
    m2 = avg_pool(m2).squeeze(3).squeeze(2)
    source_features[num_image, ...] = m2.cpu()
    # break

#%%
copy_source_features = source_features.clone()
fout = open("/data/aCardace/iterativive_da/splits/gta/selected_train.txt", "w")

#%%
added = []
for num_image, (image, label, image_path, label_path) in enumerate(tqdm(val_dl_target)):
    diff_vectors = source_features - target_fetures[num_image]
    diff_vectors = diff_vectors**2
    norms = torch.sum(diff_vectors, dim=1)
    image_indeces = torch.argsort(norms)
    source_features[image_indeces[0].item(), ...] = 100000
    source_features[image_indeces[1].item(), ...] = 100000
    source_features[image_indeces[2].item(), ...] = 100000

    if image_indeces[0].item() in added:
        print("already inserted")
    else:
        added.append(image_indeces[0].item())
        fout.write(f"{source_image_path[image_indeces[0].item()][0]};{source_paths[image_indeces[0].item()][0]};{source_paths[image_indeces[0].item()][0]}\n")

    if image_indeces[1].item() in added:
        print("already inserted")
    else:
        added.append(image_indeces[1].item())
        fout.write(f"{source_image_path[image_indeces[1].item()][0]};{source_paths[image_indeces[1].item()][0]};{source_paths[image_indeces[1].item()][0]}\n")

    if image_indeces[2].item() in added:
        print("already inserted")
    else:
        added.append(image_indeces[2].item())
        fout.write(f"{source_image_path[image_indeces[2].item()][0]};{source_paths[image_indeces[2].item()][0]};{source_paths[image_indeces[2].item()][0]}\n")
    
#%%
fout.close()

# %%

# %%1

