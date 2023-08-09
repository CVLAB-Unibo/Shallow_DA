import numpy as np
from numpy.lib.utils import source
from datasets.segmentation_dataset import SegmentationDataset
from torch.utils.data import DataLoader, dataset
import pytorch_lightning as pl
from PIL import Image
import io
import matplotlib.pyplot as plt
import torch
from datasets.transformations import get_transforms


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.files = []
        self.cfg = cfg
        
    # def setup(self, stage=None):
        train_transform, val_transform, val_transform_target = get_transforms(self.cfg)

        self.train_ds = SegmentationDataset(
            self.cfg.dataset.data_dir,
            self.cfg.dataset.source_train_list,
            mean=self.cfg.dataset.mean,
            std=self.cfg.dataset.std,
            transform=train_transform,
            encode=cfg.dataset.encode,
            encoding=cfg.dataset.source_encoding,
            num_classes=cfg.dataset.num_classes      
        )
        self.val_ds_source = SegmentationDataset(
            self.cfg.dataset.data_dir,
            self.cfg.dataset.source_val_list,
            mean=self.cfg.dataset.mean,
            std=self.cfg.dataset.std,
            transform=val_transform,
            encode=cfg.dataset.encode,
            encoding=cfg.dataset.source_encoding,
            num_classes=cfg.dataset.num_classes      
        )
        self.val_ds_target = SegmentationDataset(
            self.cfg.dataset.data_dir,
            self.cfg.dataset.target_val_list,
            mean=self.cfg.dataset.mean,
            std=self.cfg.dataset.std,
            transform=val_transform_target,
            encode=cfg.dataset.encode,
            encoding=cfg.dataset.target_encoding,
            num_classes=cfg.dataset.num_classes      
        )

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        return train_dl

    def val_dataloader(self):
        val_dl_source = DataLoader(
            self.val_ds_source,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        val_dl_target = DataLoader(
            self.val_ds_target,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
        )
        return [val_dl_source, val_dl_target]

    def test_dataloader(self):
        test_dl_target = DataLoader(
            self.val_ds_target,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
        )
        return test_dl_target
        
    def get_val_samples(self, num_samples=1):
        xb = []
        yb = []
        eb = []
        i_val = iter(self.val_dataloader()[1])
        for i in range(num_samples):
            x, y, e, _ = next(i_val)
            xb.append(x)
            yb.append(y)
            eb.append(e)

        xb = torch.cat(xb)
        yb = torch.cat(yb)
        eb = torch.cat(eb)

        return xb, yb, eb, None

    def get_train_samples(self):
        return next(iter(self.train_dataloader()))

    def __len__(self):
        dl_temp = self.train_dataloader()
        return len(dl_temp)
