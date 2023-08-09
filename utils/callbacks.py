from numpy.core.fromnumeric import size
import pytorch_lightning as pl
import torch
import numpy as np
from PIL import Image
import wandb

# from torch.nn import functional as F
from torchvision.transforms import functional as F


class ImagePredictionLogger(pl.Callback):
    def __init__(self, dataModule):
        super().__init__()
        self.dataModule = dataModule
        segmentation_classes = dataModule.train_ds.segmentation_classes
        self.l = {}
        for i, label in enumerate(segmentation_classes):
            self.l[i] = label

    def wb_mask(self, bg_img, pred_mask, true_mask):
        return wandb.Image(
            bg_img,
            masks={
                "prediction": {"mask_data": pred_mask, "class_labels": self.l},
                "ground truth": {"mask_data": true_mask, "class_labels": self.l},
            },
        )

    def log_plot(self, trainer, pl_module, samples, split):
        xb, yb, _, _ = samples
        xb = xb.to(device=pl_module.device)
        size = xb.size()[-2:]

        mask_list = []
        edge_list = []

        if yb.size() != size:
            yb = F.resize(yb, size=size, interpolation=Image.NEAREST)

        with torch.no_grad():
            predictions = pl_module(xb, size=size)
            predictions, edges = pl_module(xb, size=size)

        for i in range(predictions.size()[0]):
            prediction = predictions[i].cpu().numpy()
            prediction = prediction.transpose(1, 2, 0)
            prediction = np.asarray(np.argmax(prediction, axis=2), dtype=np.uint8)

            mask_list.append(self.wb_mask(xb[i].cpu(), prediction, yb[i].cpu().numpy()))
            edge = torch.sigmoid(edges[i])
            edge = edge.cpu().numpy()
            edge = (edge-edge.min())/(edge.max()-edge.min())
            edge_list.append(edge)

        trainer.logger.experiment.log(
            {
            f"predictions_{split}": mask_list, 
            f"edges_{split}": [wandb.Image(x) for x in edge_list], 
            "epoch": trainer.current_epoch},
            step=trainer.global_step,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        # if trainer.global_step != 0:
            samples = self.dataModule.get_val_samples(num_samples=4)
            self.log_plot(trainer, pl_module, samples, "val")
            samples = self.dataModule.get_train_samples()
            self.log_plot(trainer, pl_module, samples, "train")