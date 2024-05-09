import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from typing import Tuple

from utils.algorithms import alg_10

class BaseCNN(pl.LightningModule):

    '''
    This class implements basic functionally for training.
    Probably not needed to touch this module too much.
    For actually changing architecture, do like in OurCNN below
    '''

    def __init__(self, preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__()

        # Set up some loss and processing
        self.preparer = preparer
        self.criteria = criteria

        self.val_data_loader_names = ['val_set', 'full_training']

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    def _evaluate_on_batch(self, outputs, labels):

        _, preds = torch.max(outputs, dim=-1)
        acc = torch.sum(preds == labels).item() / len(preds)

        return acc
    
    def training_step(self, batch, batch_idx):

        data, labels = batch

        outputs, _ = self(self.preparer(data))
        # Calculate the loss, the input is our target
        loss = self.criteria(outputs, labels)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        data, labels = batch

        outputs, cnn_out = self(self.preparer(data))

        loss = self.criteria(outputs, labels)
        self.log('val_loss_'+self.val_data_loader_names[dataloader_idx], loss, prog_bar=True, logger=True)

        acc = self._evaluate_on_batch(outputs, labels)
        self.log('val_acc_'+self.val_data_loader_names[dataloader_idx], acc, prog_bar=False, logger=True)

        mec_estimate = alg_10(cnn_out.detach(), labels)
        self.log('mec_'+self.val_data_loader_names[dataloader_idx], mec_estimate, prog_bar=False, logger=True)

        return loss
    
    def test_step(self, batch, batch_idx):

        data, labels = batch

        outputs, cnn_out = self(self.preparer(data))

        loss = self.criteria(outputs, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        acc = self._evaluate_on_batch(outputs, labels)
        self.log('test_acc', acc, prog_bar=False, logger=True)

        mec_estimate = alg_10(cnn_out.detach(), labels)
        self.log('mec', mec_estimate, prog_bar=False, logger=True)

        return loss
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        scheduler = {
            'scheduler': StepLR(optimizer, step_size=1, gamma=0.1**(1/100)),  # Change LR every 10 epochs
            'interval': 'epoch',  # 'step' or 'epoch'
            'frequency': 1
        }
        return [optimizer], [scheduler]
    
    #def configure_optimizers(self):
    #    return torch.optim.Adam(self.parameters(), lr=0.001)  
