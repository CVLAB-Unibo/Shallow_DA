from platform import version
import pytorch_lightning as pl
import wandb
import torch
import random
import os
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from utils.callbacks import ImagePredictionLogger
from dataloader.segmentation_data_module import DataModule
# from models.segmentation_trainer import Model
from models.baseline import Model
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
# import cv2
import time
# cv2.setNumThreads(0)

logging.getLogger("lightning").setLevel(logging.WARNING)
# pl.seed_everything(6)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # setup model
    dm = DataModule(cfg)
    
    if cfg.test:
        # evaluate the model on a test set
        model = Model.load_from_checkpoint(checkpoint_path=cfg.resume_checkpoint, map_location="cuda:"+str(cfg.gpus[0]))
        # init trainer with whatever options
        trainer = pl.Trainer(
            gpus=cfg.gpus,
            benchmark=True,
            progress_bar_refresh_rate=50,
            precision=cfg.precision,
            amp_level=cfg.amp_level,
        )
        trainer.test(model=model, datamodule=dm)
    else:
        # fit the model
        #move in train branch
        # wandb.login()
        id = wandb.util.generate_id()
        resume=False
        if cfg.resume_checkpoint is not None:
            resume=True
            # id = cfg.id

        run = wandb.init(job_type="train", project=cfg.project_name, name=cfg.run_name + "_" + cfg.version_run, entity="adricarda", resume=resume, id=id)
        wandb_logger = WandbLogger(
            project=cfg.project_name,
            name=cfg.run_name + "_" + cfg.version_run,
            version=cfg.version_run,
            id = id
        )
        artifact = wandb.Artifact('Model', type='model')
        # artifact.add_file('networks/net.py')
        # artifact.add_file('models/segmentation_trainer.py')
        # run.log_artifact(artifact)

        print(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

        print(os.getcwd())
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoint")

        checkpoint_callback = ModelCheckpoint(
            monitor="valid/target_miou",
            dirpath=checkpoint_dir,
            filename="{epoch:02d}",
            save_top_k=1,
            save_last=True,
            mode="max",
            verbose=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        model = Model(cfg, dm=dm)
    
        trainer = pl.Trainer(
            default_root_dir=checkpoint_dir,
            logger=wandb_logger,  # W&B integration
            log_every_n_steps=1000,  # set the logging frequency
            gpus=cfg.gpus,  # use all GPUs
            max_epochs=cfg.epochs,
            benchmark=True,
            # val_check_interval=0.5,
            # automatic_optimization=False,
            progress_bar_refresh_rate=500,
            callbacks=[ImagePredictionLogger(dm), checkpoint_callback, lr_monitor],  # see Callbacks section
            resume_from_checkpoint=cfg.resume_checkpoint,
            precision=cfg.precision,
            amp_level=cfg.amp_level,
            num_sanity_val_steps=2,
        )
        trainer.fit(model, datamodule=dm)
        trainer.test(model=model, datamodule=dm, ckpt_path=None)

if __name__ == "__main__":
    main()
