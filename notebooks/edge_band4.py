#%%
# import os
# os.chdir("..")
import sys

from pytorch_lightning.core.saving import PRIMITIVE_TYPES
from torch._C import dtype
sys.path.append("..")
import pytorch_lightning as pl
# from models.segmentation_trainer import Model
from models.baseline import Model as Model
from IPython.display import display
from datasets.transformations import get_transforms
import torchvision.transforms.functional as tf
from datasets.segmentation_dataset import SegmentationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
from utils.metrics import mIoU
import torch.nn.functional as F
from hydra.experimental import initialize_config_dir, compose
from PIL import Image
from utils.vis_flow import flow_to_color 
import skimage.morphology
import cv2
import glob
from scipy.ndimage.morphology import distance_transform_edt

#%%


def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)
    """
    
    if radius < 0:
        return mask
    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    
    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    # edgemap = np.expand_dims(edgemap, axis=0)    
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

#%%

for labeL_PATH in tqdm(glob.glob("../../datasets/CityScapes/gtFine/val/*/*labelIds_encoded.png")):
    label = np.array(Image.open(labeL_PATH))
    labeL_PATH = labeL_PATH.split("/")[-1]

    labelone_hot = torch.nn.functional.one_hot(torch.tensor(label, dtype=torch.int64), 20).permute((2, 0, 1))
    edges = onehot_to_binary_edges(labelone_hot, 1, 20)*255

    label_4 = label.copy()
    edges_4 = onehot_to_binary_edges(labelone_hot, 6, 20)*255
    map_4 = edges_4-edges
    label_4[map_4==0] = 19
    cv2.imwrite("/data/aCardace/iterativive_da/qualitatives/GT_4/"+labeL_PATH, label_4)

    # label_8 = label.copy()
    # edges_8 = onehot_to_binary_edges(labelone_hot, 10, 20)*255
    # map_8 = edges_8-edges
    # label_8[map_8==0] = 19
    # cv2.imwrite("/data/aCardace/iterativive_da/qualitatives/GT_8/"+labeL_PATH, label_8)

    # label_16 = label.copy()
    # edges_16 = onehot_to_binary_edges(labelone_hot, 18, 20)*255
    # map_16 = edges_16-edges
    # label_16[map_16==0] = 19
    # cv2.imwrite("/data/aCardace/iterativive_da/qualitatives/GT_16/"+labeL_PATH, label_16)

    # break

# %%
