#%%
# import os
# os.chdir("..")
import sys
sys.path.append(".")
import cv2
import pytorch_lightning as pl
# from models.segmentation_trainer import Model
# from models.baseline import Model
from networks.baseline import Res_Deeplab
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
from utils.metrics import mIoU
import numpy as np
import PIL
import skimage.morphology
from hydra.experimental import initialize_config_dir, compose
import hydra
logging.getLogger("lightning").setLevel(logging.WARNING)
pl.seed_everything(6)
import os



#%%

initialize_config_dir(config_dir="/data/aCardace/iterativive_da/conf")
cfg = compose(config_name="config")

#%%

# setup model
target_scorer = mIoU(cfg.dataset.num_classes)
target_scorer.reset()

# evaluate the model on a test set

device = "cuda:1"
# model = Model.load_from_checkpoint(checkpoint_path=cfg.resume_checkpoint)
saved_state_dict = torch.load("/data/aCardace/iterativive_da/experiments/warping_gta/baseline_1/checkpoint/last.ckpt")["state_dict"]
model = Res_Deeplab(cfg)

new_params = model.state_dict().copy()
for i in saved_state_dict:
    i_parts = i.split('.')
    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
model.load_state_dict(new_params)

model.to(device)
model.eval()
_, val_transform = get_transforms(cfg)

val_ds_target = SegmentationDataset(
    cfg.dataset.data_dir,
    cfg.dataset.target_train_list,
    mean=cfg.dataset.mean,
    std=cfg.dataset.std,
    transform=val_transform,
    encode=cfg.dataset.encode,
    encoding=cfg.dataset.target_encoding 
)

val_dl_target = DataLoader(
    val_ds_target,
    batch_size=cfg.val_batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
)

val_ds_source = SegmentationDataset(
    cfg.dataset.data_dir,
    cfg.dataset.source_train_list,
    mean=cfg.dataset.mean,
    std=cfg.dataset.std,
    transform=val_transform,
    encode=cfg.dataset.encode,
    encoding=cfg.dataset.source_encoding 
)

val_dl_source = DataLoader(
    val_ds_source,
    batch_size=cfg.val_batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
)

#%%

def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred, classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N

def oneMix(mask, source, target):
    stackedMask0, _ = torch.broadcast_tensors(mask, source)
    return stackedMask0*source+(1-stackedMask0)*target

iterator_target = iter(val_dl_target)
iterator_source = iter(val_dl_source)

take_classes = list(range(19))

for num_image, (image, label, image_path, label_path) in enumerate(tqdm(iterator_target)):
    num_image += 2975
    label = tf.resize(label, tuple(cfg.dataset.input_size_val), interpolation=PIL.Image.NEAREST)
    image = image.to(device)
    label = label.to(device)

    scaled_image = tf.resize(image, (512, 1024))

    if "gtFine" in label_path[0]:

        with torch.no_grad():
            logits = model(scaled_image)
            logits = F.interpolate(logits, size=tuple(cfg.dataset.out_val_size), mode="bilinear", align_corners=True)

        prediction = logits.squeeze()
        prediction = F.softmax(prediction, dim=0)
        _, label = torch.max(prediction, dim=0)

    try:
        image_source, source_label, _, _ = next(iterator_source)
    except:
        val_dl_source_onlys = DataLoader(
            val_ds_target,
            batch_size=cfg.val_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        iterator_source = iter(val_dl_source_onlys)
        image_source, source_label, _, _= next(iterator_source)

    image_source = image_source.to(device)
    source_label = source_label.to(device)

    new_mask = torch.ones_like(source_label, device=device) * 19
    rand_classes = np.random.choice(take_classes, size=10, replace=False)
    rand_classes = torch.tensor(rand_classes, device=device)
    mask = generate_class_mask(source_label, rand_classes)
    augmented_source_image = oneMix(mask, image_source, image)
    augmented_source_label = oneMix(mask, source_label, label)

    # display(Image.fromarray(val_ds_target.re_normalize(augmented_source_image.cpu().numpy().squeeze())))
    # display(val_ds_target.colorize_mask(augmented_source_label.cpu().numpy().squeeze()))

    # if "gtFine" in label_path[0]:
    #     dest_path_img = image_path[0].replace("left", "w_images_step1", 1)
    #     dest_path_label = label_path[0].replace("gtFine", "w_semantic_encoded_step1", 1)
    # else:
    #     dest_path_img = image_path[0].replace("images_translated_bdl", "w_images_step1", 1)
    #     dest_path_label = label_path[0].replace("semantic_encoded", "w_semantic_encoded_step1", 1)

    dest_path_img = "/data/aCardace/datasets/GTA5/step1_dacs_da/" + str(num_image) + ".png"
    dest_path_label = "/data/aCardace/datasets/GTA5/step1_dacs_da_semantic_encoded/" + str(num_image) + ".png"

    # print(dest_path_img)
    # print(augmented_source_label.max())
    # os.makedirs(os.path.dirname(dest_path_img), exist_ok=True)
    # os.makedirs(os.path.dirname(dest_path_label), exist_ok=True)

    augmented_source_image = val_ds_target.re_normalize(augmented_source_image.cpu().numpy().squeeze())
    plt.imsave(dest_path_img, augmented_source_image)
    cv2.imwrite(dest_path_label, augmented_source_label.cpu().numpy().squeeze())
    # break
# %%

# %%
