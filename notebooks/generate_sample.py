#%%
# import os
# os.chdir("..")
import sys
sys.path.append("..")
import cv2
import pytorch_lightning as pl
from torch import random
from dataloader.segmentation_data_module import DataModule
from models.segmentation_trainer import Model
# from models.baseline import Model
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
from utils.metrics import mIoU
import numpy as np
import PIL
import random 
import skimage.morphology
from hydra.experimental import initialize, initialize_config_module, initialize_config_dir, compose
import hydra
logging.getLogger("lightning").setLevel(logging.WARNING)
pl.seed_everything(42)
import os

#%%

def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N

def oneMix(mask, source, target):
    stackedMask0, _ = torch.broadcast_tensors(mask, source)
    return stackedMask0*source+(1-stackedMask0)*target

#%%

initialize_config_dir(config_dir="/data/aCardace/iterativive_da/conf")
cfg = compose(config_name="config")

#%%

# setup model
target_scorer = mIoU(cfg.dataset.num_classes)
target_scorer.reset()

# evaluate the model on a test set

device = "cpu"
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

iterator_target = iter(val_dl_target)


#%%
iterator_target = iter(val_dl_target)
kernel = np.ones((5, 5))
take_classes = [11]

for num_image, (image, label, image_path, label_path) in enumerate(tqdm(val_dl_source)):
    label = tf.resize(label, tuple(cfg.dataset.input_size_val), interpolation=PIL.Image.NEAREST)
    image = image.to(device)
    label = label.to(device)

    if "gtFine" in label_path[0]:

        scaled_image = tf.resize(image, (1024, 2048))
        label = tf.resize(label, tuple(cfg.dataset.input_size_val), interpolation=PIL.Image.NEAREST)

        with torch.no_grad():
            logits, _ = model(image, size=tuple(cfg.dataset.input_size_val))
            flipped, _ = model(tf.hflip(image), size=tuple(cfg.dataset.input_size_val))
            flipped = tf.hflip(flipped)
            logits_scaled_up, _ = model(scaled_image, size=tuple(cfg.dataset.input_size_val))

        prediction_source = logits.squeeze()
        flipped_prediction_source = flipped.squeeze()
        prediction_scaled_up_source = logits_scaled_up.squeeze()

        prediction = 0.6*prediction_source + 0.2*flipped_prediction_source + 0.2*prediction_scaled_up_source
        prediction = F.softmax(prediction, dim=0)
        _, label = torch.max(prediction, dim=0)

    for ii in range(1):
        print(ii)
        try:
            image_target, _, _, _ = next(iterator_target)
        except:
            val_dl_target = DataLoader(
                val_ds_target,
                batch_size=cfg.val_batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
            iterator_target = iter(val_dl_target)
            image_target, label_target, _, _= next(iterator_target)
        image_target = image_target.to(device)
        scaled_image = tf.resize(image_target, (1024, 2048))

        with torch.no_grad():
            logits, _ = model(image_target, size=tuple(cfg.dataset.input_size_val))
            # flipped, _= model(tf.hflip(image_target), size=tuple(cfg.dataset.input_size_val))
            # flipped = tf.hflip(flipped)
            # logits_scaled_up, _= model(scaled_image, size=tuple(cfg.dataset.input_size_val))

        prediction = logits.squeeze()
        # flipped_prediction = flipped.squeeze()
        # prediction_scaled_up = logits_scaled_up.squeeze()

        # prediction = 0.6*prediction + 0.2*flipped_prediction + 0.2*prediction_scaled_up
        # prediction = F.softmax(prediction, dim=0)
        predicted_prob, predicted_label = torch.max(prediction, dim=0)

        new_mask = torch.ones_like(predicted_label) * 19
        # rand_classes = np.random.choice(take_classes, size=10, replace=False)
        rand_classes = torch.tensor(take_classes).to(device)
    
        for c in rand_classes:
            class_mask = np.equal(predicted_label.cpu().numpy(), c.cpu().numpy())
            erored_mask = skimage.morphology.erosion(class_mask, kernel)
            new_mask[erored_mask] = c
        mask = generate_class_mask(new_mask, rand_classes)


        #do not take pixels in wich network is uncertain        
        # for i in range(19):
        #     predicted_label[(predicted_prob<thres[i])*(predicted_label==i)] = 19
        # mask[predicted_label==19] = 0

        if ii==0:
            augmented_source_image = oneMix(mask, image_target, image)
            augmented_source_label = oneMix(mask, predicted_label, label)
        else:
            augmented_source_image = oneMix(mask, image_target, augmented_source_image)
            augmented_source_label = oneMix(mask, predicted_label, augmented_source_label)

    # display(Image.fromarray(val_ds_target.re_normalize(augmented_source_image.cpu().numpy().squeeze())))
    # display(val_ds_target.colorize_mask(augmented_source_label.cpu().numpy().squeeze()))

    # if "gtFine" in label_path[0]:
    #     dest_path_img = image_path[0].replace("left", "w_images_step1", 1)
    #     dest_path_label = label_path[0].replace("gtFine", "w_semantic_encoded_step1", 1)
    # else:
    #     dest_path_img = image_path[0].replace("images_translated_bdl", "w_images_step1", 1)
    #     dest_path_label = label_path[0].replace("semantic_encoded", "w_semantic_encoded_step1", 1)

    # dest_path_img = "/data/aCardace/datasets/Synscapes/img/w_images_step1/" + str(num_image) + ".png"
    # dest_path_label = "/data/aCardace/datasets/Synscapes/img/w_semantic_encoded_step1/" + str(num_image) + ".png"

    # print(dest_path_img)
    # print(dest_path_label)
    # os.makedirs(os.path.dirname(dest_path_img), exist_ok=True)
    # os.makedirs(os.path.dirname(dest_path_label), exist_ok=True)

    image = val_ds_target.re_normalize(image.cpu().numpy().squeeze())
    Image.fromarray(image).save("/data/aCardace/iterativive_da/qualitatives/source_image.png")
    val_dl_target.dataset.colorize_mask(label.squeeze().cpu().numpy()).save("/data/aCardace/iterativive_da/qualitatives/source_label.png")

    augmented_source_image = val_ds_target.re_normalize(augmented_source_image.cpu().numpy().squeeze())
    Image.fromarray(augmented_source_image).save("/data/aCardace/iterativive_da/qualitatives/augmented_image.png")
    val_dl_target.dataset.colorize_mask(augmented_source_label.squeeze().cpu().numpy()).save("/data/aCardace/iterativive_da/qualitatives/augmented_label.png")
    image_target = val_ds_target.re_normalize(image_target.cpu().numpy().squeeze())

    # mask_label = np.ma.masked_array(predicted_label, np.logical_not(mask))
    # mask_label[mask_label!=11] = 0
    # display(Image.fromarray(mask_label.astype(np.uint8)*255))
    mask_label = np.where(mask, predicted_label, 20)
    mask_image = np.where(mask.unsqueeze(2).repeat(1,1,3), image_target, (255, 255, 255)).astype(np.uint8)

    # print(mask_label.shape) 
    Image.fromarray(image_target).save("/data/aCardace/iterativive_da/qualitatives/image_target.png")
    val_dl_target.dataset.colorize_mask(predicted_label.squeeze().cpu().numpy()).save("/data/aCardace/iterativive_da/qualitatives/label_target.png")
    val_dl_target.dataset.colorize_mask(mask_label).save("/data/aCardace/iterativive_da/qualitatives/mask_label.png")
    Image.fromarray(mask_image).save("/data/aCardace/iterativive_da/qualitatives/mask_image.png")

    
    # print(mask.size())

    # plt.imsave(dest_path_img, augmented_source_image)
    # cv2.imwrite(dest_path_label, augmented_source_label.cpu().numpy().squeeze())
    break

# %%

