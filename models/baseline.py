from numpy.core.fromnumeric import mean
import pytorch_lightning as pl

# pl.seed_everything(42)

import numpy as np
import torch
from torch.nn import functional as F
# from networks.baseline import Res_Deeplab
from networks.proda import Res_Deeplab
import torch.nn as nn
from utils.metrics import mIoU
from utils.losses import get_loss_fn
from utils.optimizers import get_optimizer
import wandb


class Model(pl.LightningModule):
    def __init__(self, cfg, dm=None):
        super().__init__()
        self.best_miou = 0
        self.save_hyperparameters(cfg)

        self.net = Res_Deeplab(cfg)
        self.loss_fn = get_loss_fn(cfg)
        self.source_scorer = mIoU(cfg.dataset.num_classes)
        self.source_scorer.reset()
        self.target_scorer = mIoU(cfg.dataset.num_classes)
        self.target_scorer.reset()
        self.dm = dm

    def forward(self, x, size):
        x = self.net(x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return x

    def loss(self, xs, ys, size):
        logits = self(xs, size)  # this calls self.forward
        loss = self.loss_fn(logits, ys)
        return logits, loss

    def on_train_epoch_start(self):
        if self.hparams.freeze_bn:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find("BatchNorm") != -1:
                    m.eval()
            self.net.apply(set_bn_eval)

    def eval_bn(self):
        def freeze_bn(m):
            if m.__class__.__name__.find('BatchNorm') != -1 or isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
        self.net.apply(freeze_bn)

    def training_step(self, batch, batch_idx):
        # opt = self.optimizers()
        self.eval_bn()
        xb, yb, _, _ = batch
        logits, loss = self.loss(xb, yb, tuple(self.hparams.dataset.out_train_size))

        if self.global_step % 500 == 0:
            self.logger.experiment.log(
                {"train/loss": loss.item()}, commit=True, step=self.global_step
            )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        xb, yb, _, _ = batch
        if dataloader_idx==0:
            logits, loss = self.loss(xb, yb, tuple(self.hparams.dataset.out_val_size))
        else:
            logits, loss = self.loss(xb, yb, tuple(self.hparams.dataset.out_val_size_target))

        prediction = logits.cpu().numpy().squeeze()
        prediction = prediction.transpose(1, 2, 0)
        prediction = np.asarray(np.argmax(prediction, axis=2), dtype=np.uint8)
        yb = yb.cpu().numpy()

        if dataloader_idx == 0:
            scorer = self.source_scorer
        else:
            scorer = self.target_scorer

        scorer.update(prediction.flatten(), yb.flatten())
        return loss

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, 1)

    def validation_epoch_end(self, validation_step_outputs):
        losses_source = validation_step_outputs[0]
        # avoid logging when training starts
        if len(losses_source) > 10:
            avg_loss = np.mean([x.item() for x in losses_source])
            miou, ious = self.source_scorer.get_score()
            print("SOURCE")
            print('mean iou: {:.2f}%'.format(np.mean(ious)))
            print(' '.join("{:.2f}".format(value) for value in ious))
            
            self.source_scorer.reset()
            self.logger.experiment.log(
                {"valid/source_loss": avg_loss}, commit=False, step=self.global_step
            )
            self.logger.experiment.log(
                {"valid/source_miou": miou}, commit=False, step=self.global_step
            )

            losses_target = validation_step_outputs[1]
            avg_loss = np.mean([x.item() for x in losses_target])
            miou, ious = self.target_scorer.get_score()
            print("TARGET")
            print('mean iou: {:.2f}%'.format(np.mean(ious)))
            print(' '.join("{:.2f}".format(value) for value in ious))
            self.target_scorer.reset()
            self.logger.experiment.log(
                {"valid/target_loss": avg_loss}, commit=False, step=self.global_step
            )

            if miou > self.best_miou and self.global_step != 0:
                self.logger.log_metrics({"best_miou": miou}, step=self.global_step)
                self.best_miou = miou

            self.log("valid/target_miou", miou)

    def test_epoch_end(self, outputs):
        print("TARGET")
        miou, mious = self.target_scorer.get_score()
        print(miou)
        print(mious)

        if not self.hparams.test:
            table = wandb.Table(columns=["class", "IoU"])
            table.add_data("mean", "{:.2f}%".format(miou))
            for c, iou in enumerate(mious):
                table.add_data(f"{c}", "{:.2f}%".format(iou))
            self.logger.experiment.log({"Final_results": table})
        self.target_scorer.reset()

    def configure_optimizers(self):
        opt = get_optimizer(self.net, self.hparams)
        # return opt
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=[param_group["lr"] for param_group in opt.param_groups],
            steps_per_epoch=len(self.dm),
            epochs=self.hparams.epochs,
            div_factor=5,
        )
        return [opt], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_miou"] = self.best_miou

    def on_load_checkpoint(self, checkpointed_state):
        self.best_miou = checkpointed_state["best_miou"]
