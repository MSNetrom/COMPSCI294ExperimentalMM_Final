from old_files.model import BaseCNN
import torch
from torch.optim.lr_scheduler import StepLR

# Optimal CNN

class OptimalCNN(BaseCNN):

    def __init__(self, preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.conv_sequence = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=(1, 0), padding_mode='reflect'),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=(1, 0), padding_mode='reflect'),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2))
        

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(6, 6),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(6, 15))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.preparer(x)
        x = self.conv_sequence(x[:, :, :, 1:-1])

        #print("Shape before flatten:", x.shape)

        return self.full_connected_sequence(x.flatten(start_dim=1))
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
    

class OptimalCNN2(BaseCNN):

    def __init__(self, preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.conv_sequence = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=0, padding_mode='reflect'),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                    #torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=(1, 0), padding_mode='reflect'),
                    #torch.nn.ReLU(),
                    #torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=0),
                    #torch.nn.ReLU(),
                    #orch.nn.MaxPool2d(kernel_size=3, stride=2))
        

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(9, 9),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(9, 15))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.preparer(x)
        x = self.conv_sequence(x)

        #print("Shape before flatten:", x.shape)

        return self.full_connected_sequence(x.flatten(start_dim=1))
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        scheduler = {
            'scheduler': StepLR(optimizer, step_size=1, gamma=0.1**(1/100)),  # Change LR every 100 epochs
            'interval': 'epoch',  # 'step' or 'epoch'
            'frequency': 1
        }
        return [optimizer], [scheduler]
    

class OptimalCNN3(BaseCNN):

    def __init__(self, preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.conv_sequence = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=2, padding=0, padding_mode='reflect'),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    torch.nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                    torch.nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=0),
                    torch.nn.ReLU())
                    #torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=(1, 0), padding_mode='reflect'),
                    #torch.nn.ReLU(),
                    #torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=0),
                    #torch.nn.ReLU(),
                    #orch.nn.MaxPool2d(kernel_size=3, stride=2))
        

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(9, 9),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(9, 15))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.preparer(x)
        x = self.conv_sequence(x[:, :, 10:-10, 10:-10])

        #print("Shape before flatten:", x.shape)

        return self.full_connected_sequence(x.flatten(start_dim=1))
    

class OptimalCNN4(BaseCNN):

    def __init__(self, hidden_size: int, preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.conv_sequence = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=2, padding=0, padding_mode='reflect'),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    torch.nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                    torch.nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=0),
                    torch.nn.ReLU())

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(9, hidden_size),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(hidden_size, 15))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.preparer(x)
        x = self.conv_sequence(x[:, :, 9:-9, 9:-9])

        #print("Shape before flatten:", x.shape)

        return self.full_connected_sequence(x.flatten(start_dim=1))
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        scheduler = {
            'scheduler': StepLR(optimizer, step_size=1, gamma=0.1**(1/100)),  # Change LR every 10 epochs
            'interval': 'epoch',  # 'step' or 'epoch'
            'frequency': 1
        }
        return [optimizer], [scheduler]