#%%
import os
os.chdir("..")
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
import numpy as np
from utils.metrics import mIoU
import torch.nn.functional as F
from hydra.experimental import initialize_config_dir, compose
from PIL import Image
from utils.vis_flow import flow_to_color 
import PIL
pl.seed_everything(42)

#%%

initialize_config_dir(config_dir="/data/aCardace/iterativive_da/conf")
cfg = compose(config_name="config")

#%%

# setup model
target_scorer = mIoU(cfg.dataset.num_classes)
device = "cuda:1"
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
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)


#%%
predicted_label = np.zeros((len(val_dl_target), 512, 1024))
predicted_prob = np.zeros((len(val_dl_target), 512, 1024))
image_name = []

# target_scorer.reset()
for num_image, (image, label, image_path, label_path) in enumerate(tqdm(val_dl_target)):
    image = image.to(device)
    scaled_image = tf.resize(image, (512, 1024))
    label = tf.resize(label, (512, 1024), PIL.Image.NEAREST)

    with torch.no_grad():
        logits, edges = model.net(image, get_flow=False)
        logits = tf.resize(logits, tuple((512, 1024)))

        # logits= model(image, tuple((512, 1024)))

        flipped, _ = model(tf.hflip(image), size=tuple((512, 1024)))
        flipped = tf.hflip(flipped)
        logits_scaled_up, _ = model(scaled_image, size=tuple((512, 1024)))

    prediction = logits.squeeze()
    flipped_prediction = flipped.squeeze()
    prediction_scaled_up_source = logits_scaled_up.squeeze()

    prediction = 0.6*prediction + 0.2*flipped_prediction + 0.2*prediction_scaled_up_source
    prediction = F.softmax(prediction, dim=0)
    prob, prediction = torch.max(prediction, dim=0)
    predicted_label[num_image] = prediction.cpu().numpy()
    predicted_prob[num_image] = prob.cpu().numpy()
    image_name.append(label_path[0])
    # display(val_ds_target.colorize_mask(prediction.cpu().numpy().squeeze()))

    # target_scorer.update(prediction.cpu().numpy().flatten(), label.numpy().flatten())
    # break

# # %%
# mask = np.ones(16, dtype=np.bool)
# mask[3] = False
# mask[4] = False
# mask[5] = False

# miou, classes = target_scorer.get_score(mask=mask)
# print(miou)
# %%
thres = []
for i in range(16):
    x = predicted_prob[predicted_label==i]
    if len(x) == 0:
        thres.append(0)
        continue        
    x = np.sort(x)
    thres.append(x[np.int(np.round(len(x)*0.5))])
print(thres)
thres = np.array(thres)
thres[thres>0.9]=0.9
print(thres)
for index in range(len(val_dl_target)):
    name = image_name[index]
    label = predicted_label[index]
    prob = predicted_prob[index]
    for i in range(16):
        label[(prob<thres[i])*(label==i)] = 16  
    output = np.asarray(label, dtype=np.uint8)
    output = Image.fromarray(output)
    dest_path_label = image_name[index].replace("gtFine", "final_step_filtered_synthia", 1)
    os.makedirs(os.path.dirname(dest_path_label), exist_ok=True)
    output.save(dest_path_label)
    # print(dest_path_label)
    # break