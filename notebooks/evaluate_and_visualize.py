#%%
# import os
# os.chdir("..")
import sys
sys.path.append(".")
# from networks.baseline import Res_Deeplab
# from networks.net import Res_Deeplab
import pytorch_lightning as pl
# from models.segmentation_trainer import Model
from models.baseline import Model as Model
# from networks.baseline import Res_Deeplab-
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

pl.seed_everything(42)

#%%

initialize_config_dir(config_dir="/data/aCardace/iterativive_da/conf")
cfg = compose(config_name="config")

#%%

# setup model
target_scorer = mIoU(cfg.dataset.num_classes)
device = "cuda:1"
model = Model.load_from_checkpoint(checkpoint_path=cfg.resume_checkpoint)
# model_baseline = Res_Deeplab.load_from_checkpoint("/data/aCardace/iterativive_da/experiments/warping_gta/baseline_1/checkpoint/last.ckpt")
# saved_state_dict = torch.load("/data/aCardace/iterativive_da/experiments/warping_gta/baseline_1/checkpoint/last.ckpt")["state_dict"]
# saved_state_dict = torch.load("/data/aCardace/iterativive_da/experiments/warping_gta/warp_only+edges_7/checkpoint/last.ckpt")
# model = Res_Deeplab(cfg)
# model.load_state_dict(ckpt)


# new_params = model.state_dict().copy()
# for i in saved_state_dict:
#     # Scale.layer5.conv2d_list.3.weight
#     i_parts = i.split('.')
#     # print i_parts
#     # if not args.num_classes == 19 or not i_parts[1] == 'layer5':
#     new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
#         # print i_parts
# model.load_state_dict(new_params)

model.to(device)
model.eval()
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

val_dl_target = DataLoader(
    val_ds_target,
    batch_size=cfg.val_batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

#%%

target_scorer.reset()
for num_image, (image, label, image_path, label_path) in enumerate(tqdm(val_dl_target)):

    image = image.to(device)
    # scaled_image = tf.resize(image, (1024, 2048))
    with torch.no_grad():
        # logits = model(image)
        # logits = tf.resize(logits, tuple(cfg.dataset.out_val_size))
        # logits = F.interpolate(logits, size=tuple(cfg.dataset.out_val_size), mode="bilinear", align_corners=True)
        logits = model(image, tuple(cfg.dataset.out_val_size))


        # flipped = model(tf.hflip(image), size=tuple(cfg.dataset.out_val_size))
        # flipped = tf.hflip(flipped)
        # logits_scaled_up = model(scaled_image, size=tuple(cfg.dataset.out_val_size))

    prediction = logits.squeeze()

    # flipped_prediction = flipped.squeeze()
    # prediction_scaled_up_source = logits_scaled_up.squeeze()

    # prediction = 0.6*prediction + 0.2*flipped_prediction + 0.2*prediction_scaled_up_source
    # prediction = F.softmax(prediction, dim=0)
    _, prediction = torch.max(prediction, dim=0)
    # image_path = image_path[0].split("/")[-1]
    # print( prediction.cpu().numpy().astype(np.uint8).shape)

    prediction[label.squeeze().cpu().numpy()==19] = 19

    # image = Image.fromarray(val_dl_target.dataset.re_normalize(image.cpu().numpy().squeeze())).resize((512, 256))
    # colored_prediction = val_dl_target.dataset.colorize_mask(prediction.squeeze().cpu().numpy()).resize((512, 256))
    # colored_prediction = colored_prediction.resize((512, 256), Image.NEAREST)
    # colored_gt = val_dl_target.dataset.colorize_mask(label.squeeze().cpu().numpy())
    # colored_gt = colored_gt.resize((512, 256), Image.NEAREST)
    
    # display(colored_prediction)
    # display(colored_gt)
    
    # colored_prediction.save(f"/data/aCardace/iterativive_da/qualitatives/nthu/baseline_tokyo/{num_image}.png")
    # colored_gt.save(f"/data/aCardace/iterativive_da/qualitatives/nthu/gt_rome/{num_image}.png")

    # edges = edges.squeeze().cpu().numpy().astype(np.uint8)*255
    # edges = skimage.morphology.dilation(edges, np.ones((5, 5)))
    # edges = Image.fromarray(edges)
    # flow_uv, flow_u, flow_v = flow_to_color(flow.squeeze(dim=0).permute(1,2,0).cpu().numpy())
    # flow_uv = Image.fromarray(flow_uv).resize((512, 256))
    # display(flow_uv)
    # display(Image.fromarray(flow_u).resize((1024, 512)))
    # display(Image.fromarray(flow_v).resize((1024, 512)))

    # image.save(f"/data/aCardace/iterativive_da/qualitatives/nthu/rgb_taipei/{num_image}.png")
    # flow_uv.save(f"/data/aCardace/iterativive_da/qualitatives/nthu/flow_image/flow_{str(label_path)}")
    # image.save(f"/data/aCardace/iterativive_da/qualitatives/nthu/best_gta/image_{num_image}.png")
    # colored_gt.save(f"/data/aCardace/iterativive_da/qualitatives/nthu/source_gt_{num_image}.png")

    # flow_uv.save(f"/data/aCardace/iterativive_da/qualitatives/nthu/flow_2{num_image}.png")

    target_scorer.update(prediction.cpu().numpy().flatten(), label.numpy().flatten())
    # break
    # colored_prediction_baseline = val_dl_target.dataset.colorize_mask(prediction_baseline.squeeze().cpu().numpy()).resize((512, 256))
    # new_im = Image.new('RGB', (1024, 512))
    # new_im.paste(image, (0,0))
    # new_im.paste(flow_uv, (512,0))
    # new_im.paste(colored_prediction_baseline, (0,256))
    # new_im.paste(colored_prediction, (512,256))
    # label_path = label_path[0].split("/")[-1].strip()
    # new_im.save(f"/data/aCardace/iterativive_da/qualitatives/nthu/flow_image_combined/{str(label_path)}")

    # break


# %%
miou, classes = target_scorer.get_score()
print(miou)
print(classes)

# mask = np.ones(19, dtype=np.bool)
# mask[3] = False
# mask[4] = False
# mask[5] = False
# mask[9] = False
# mask[14] = False
# mask[16] = False
# miou, classes = target_scorer.get_score(mask=mask)
# print(classes)

# print(miou)

# %%
