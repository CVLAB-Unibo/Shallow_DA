#%%
# import os
# os.chdir("..")
import sys
sys.path.append(".")

import matplotlib.pyplot as plt
from pytorch_lightning.core.saving import PRIMITIVE_TYPES
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

#%%

target_scorer = mIoU(19)
initialize_config_dir(config_dir="/data/aCardace/iterativive_da/conf")
cfg = compose(config_name="config")
_, val_transform = get_transforms(cfg)

val_ds_target = SegmentationDataset(
    cfg.dataset.data_dir,
    cfg.dataset.target_val_list,
    mean=cfg.dataset.mean,
    std=cfg.dataset.std,
    transform=val_transform,
    encode=cfg.dataset.encode,
    encoding=cfg.dataset.target_encoding  
)

#%%
# udas = ["predictions_Stuff", "predictions_adaptsegnet", "predictions_IAST", "predictions_LTIR", "predictions_MRNET", "predictions_OURS"]

udas = ["predictions_baseline_IT", "predictions_warp_edge_no_ST", "predictions_ST_aug_only", "predictions_warp_edge", "predictions_OURS"]
bands = ["GT_4", "GT_8", "GT_16", "GT_20"]

x = [4, 8, 16, 20]
ys = []

for band in bands:
    scores = []
    for uda in udas:
        target_scorer.reset()
        correct=0
        tot=0
        for labeL_PATH in tqdm(glob.glob(f"/data/aCardace/iterativive_da/qualitatives/trimaps/{band}/*.png")):
        # for labeL_PATH in tqdm(glob.glob("/data/aCardace/datasets/CityScapes/gtFine/val/*/*gtFine_labelIds_encoded.png")):
            label = np.array(Image.open(labeL_PATH))
            labeL_PATH = labeL_PATH.split("/")[-1]
            prediction = f"/data/aCardace/iterativive_da/qualitatives/UDAS/{uda}/"+labeL_PATH.replace("gtFine_labelIds_encoded", "leftImg8bit")
            prediction = np.array(Image.open(prediction))

            target_scorer.update(prediction.flatten(), label.flatten())

            correct += np.sum(prediction[label!=19].flatten() == label[label!=19].flatten())
            tot += np.sum(label!=19)
        # break
        # print()
        # print(band, uda)
        # print("Accuracy", correct/tot)
        miou, classes = target_scorer.get_score()
        scores.append(miou)
        # print("miou", miou)
    ys.append([scores])
    print(scores)

# %%
plt.rcParams.update({'font.size': 14.5})

linestyle = ["-"]
# colors = ['k', 'c', 'y', 'r', 'm', 'g']
# marker = ["s", ".", "p", "^", "D", "*"]
# names = ["StuffAndThings", "AdaptSegNet", "IAST", "LTIR", "MRNET", "Ours"]
names = ["Row 2", "Row 4", "Row 5", "Row 6", "Row 8"]
colors = ['k', 'c', 'm', 'y', 'r']
marker = ["s", ".", "D","p", "^"]

for j in range(len(udas)):
    s_uda = []
    for i, band in enumerate(bands):
        s_uda.append(ys[i][0][j])
    plt.plot(x, s_uda, color=colors[j], linestyle=linestyle[0], marker=marker[j])

plt.grid()
plt.legend(names, fontsize="small")
plt.ylabel("mean IOU (%)")
plt.xlabel("Trimap Width (pixels)")
plt.savefig("trimap_method_big.png", transparent=True, dpi=1200)
# %%
