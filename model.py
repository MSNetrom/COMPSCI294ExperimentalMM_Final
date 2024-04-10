import torch
from torchvision import transforms
import pytorch_lightning as pl

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def _evaluate_on_batch(self, batch):

        data, labels = batch

        outputs = self(self.preparer(data))

        _, preds = torch.max(outputs, dim=-1)
        acc = torch.sum(preds == labels).item() / len(preds)

        return acc
    
    def _step(self, batch):
            
        data, labels = batch

        # Forward pass
        outputs = self(self.preparer(data))

        # Calculate the loss, the input is our target
        loss = self.criteria(outputs, labels)

        return loss
    
    def training_step(self, batch, batch_idx):

        loss = self._step(batch)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        loss = self._step(batch)
        self.log('val_loss_'+self.val_data_loader_names[dataloader_idx], loss, prog_bar=True, logger=True)

        acc = self._evaluate_on_batch(batch)
        self.log('val_acc_'+self.val_data_loader_names[dataloader_idx], acc, prog_bar=False, logger=True)

        return loss
    
    def test_step(self, batch, batch_idx):

        loss = self._step(batch)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        acc = self._evaluate_on_batch(batch)
        self.log('test_acc', acc, prog_bar=False, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)  
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

class OurCNN(BaseCNN):

    def __init__(self, preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.conv_sequence = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2))

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(24*4*4, 50),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(50, 10))               


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.preparer(x)
        x = self.conv_sequence(x)
        x = self.full_connected_sequence(x.flatten(start_dim=1))

        return x