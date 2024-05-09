import torch
from typing import Tuple

from base_cnn import BaseCNN

class OptimalCNN(BaseCNN):

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
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.preparer(x)
        x = self.conv_sequence(x[:, :, 9:-9, 9:-9]).flatten(start_dim=1)

        return self.full_connected_sequence(x), x