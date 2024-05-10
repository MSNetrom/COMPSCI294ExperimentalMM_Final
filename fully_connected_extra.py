import torch
from typing import Tuple

from base_cnn import BaseCNN

class FullyConnectedExtra(BaseCNN):

    def __init__(self, preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.fully_connected_sequence = torch.nn.Sequential(torch.nn.Linear(4096, 10),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(10, 10),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(10, 10),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(10, 15))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.preparer(x)
        x = x.flatten(start_dim=1)

        return self.fully_connected_sequence(x), x