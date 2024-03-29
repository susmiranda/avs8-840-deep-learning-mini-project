from typing import Any
import math

import torch
from torch import nn, optim
import lightning as L
from transformers import get_cosine_schedule_with_warmup

from src.models.ssformer import SSTransformer
from utils.masking import AudioMaskingGenerator

class LightningSSformer(L.LightningModule):
    def __init__(self,
                 encoder: nn.Module,
                 config):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="none")
        self.model = SSTransformer(encoder, **config['hparams']['SSformer'])
        self.config = config
        self.mask_generator = AudioMaskingGenerator(mask_prob=config["hparams"]["SSformer"]["mask_prob"],
                                           mask_length=config["hparams"]["SSformer"]["mask_length"],
                                           attention_mask=None,
                                           min_masks=config["hparams"]["SSformer"]["min_masks"])
        
    def forward(self, inputs, target, mask):
        return self.model(inputs, target, mask)
            
    def create_mask(self, spec):
        batch_size = spec.size(dim=0)
        audio_length = spec.size(dim=-1)
        mask = self.mask_generator(shape=(batch_size, audio_length))
        mask = torch.cat([torch.zeros(batch_size, 1, device=mask.device), mask], dim=1).bool()
        return mask
    
    def compute_var(self, y: torch.Tensor):
        """
        Function for computing standard deviation of target
        :param y:
        :return: standard deviation of y
        """
        y = y.view(-1, y.size(-1))
        return torch.sqrt(y.var(dim=0) + 1e-6).mean()
    
    def training_step(self, batch, batch_idx):
        spec = batch
        mask = self.create_mask(spec)
        predictions, targets = self(spec, spec, mask)

        scale = math.sqrt(predictions.size(dim=-1))
        loss = self.criterion(predictions.float(), targets.float()).sum(dim=-1).sum().div(scale)

        # self.model.ema_step()

        with torch.no_grad():
            target_var = self.compute_var(targets.float())
            prediction_var = self.compute_var(predictions.float())

        self.log_dict({"train_loss": loss, "lr": self.optimizer.param_groups[0]["lr"],
                            "target_var": target_var, "prediction_var": prediction_var}, on_epoch=True, on_step=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        self.model.ema_step()
    
    def validation_step(self, batch, batch_idx):
        spec = batch
        mask = self.create_mask(spec)
        predictions, targets = self(spec, spec, mask)
        
        scale = math.sqrt(predictions.size(dim=-1))
        loss = self.criterion(predictions.float(), targets.float()).sum(dim=-1).sum().div(scale)

        target_var = self.compute_var(targets.float())
        prediction_var = self.compute_var(predictions.float())

        self.log_dict({"val_loss": loss,
                "val_target_var": target_var, "val_prediction_var": prediction_var}, on_epoch=True, on_step=True, sync_dist=True)        

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["hparams"]["optimizer"]["lr"],
                           betas=self.config["hparams"]["optimizer"]["betas"],
                           eps=self.config["hparams"]["optimizer"]["eps"],
                           weight_decay=self.config["hparams"]["optimizer"]["weight_decay"])
        scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.config["hparams"]["scheduler"]["n_warmup"], self.config["hparams"]["n_epochs"])
        return [self.optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

class LightningKWT(L.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 config):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.model = model
        self.config = config
        
    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        specs, targets = batch
        outputs = self(specs)
        loss = self.criterion(outputs, targets)
        correct = outputs.argmax(1).eq(targets.argmax()).sum().float()

        self.log_dict({"train_loss": loss, "lr": self.optimizer.param_groups[0]["lr"],
                            "tr_correct_predictions": correct}, on_epoch=True, on_step=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):    
        specs, targets = batch
        outputs = self(specs)
        val_loss = self.criterion(outputs, targets)
        correct = outputs.argmax(1).eq(targets.argmax()).sum().float()
        accuracy = correct / len(batch)
        self.log_dict({"val_loss": val_loss, "val_correct_predictions": correct, "val_accuracy": accuracy}, on_epoch=True, on_step=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["hparams"]["optimizer"]["lr"],
                           betas=self.config["hparams"]["optimizer"]["betas"],
                           eps=self.config["hparams"]["optimizer"]["eps"],
                           weight_decay=self.config["hparams"]["optimizer"]["weight_decay"])
        scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.config["hparams"]["scheduler"]["n_warmup"], self.config["hparams"]["n_epochs"])
        return [self.optimizer], [{"scheduler": scheduler, "interval": "epoch"}]