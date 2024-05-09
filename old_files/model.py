import torch
from torchvision import transforms
import pytorch_lightning as pl

    

class MECCNN(BaseCNN):

    def __init__(self, preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.conv_sequence = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2), # Half the size, factor of 2 now
                    torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2), # Half the size again, factor of 4 now
                    torch.nn.Conv2d(in_channels=12, out_channels=23, kernel_size=3, stride=1, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2)) # 4*4/(k/12)=8.43, k = 12*4*4/8.43 = 23

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(368, 50),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(50, 10),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(10, 10))               


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.preparer(x)
        x = self.conv_sequence(x)
        x = self.full_connected_sequence(x.flatten(start_dim=1))

        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

# have a look here: https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist
# chinese MNIST: https://www.kaggle.com/discussions/general/174582

class HighDimMECCNN(BaseCNN):

    def __init__(self, preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.conv_sequence = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=12, out_channels=6, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2), # Half the size, factor of 2 now
                    torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2), # Half the size again, factor of 4 now
                    torch.nn.Conv2d(in_channels=12, out_channels=23, kernel_size=3, stride=1, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=23, out_channels=23, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2)) # 4*4/(k/12)=8.43, k = 12*4*4/8.43 = 23
        
        self.skip_conv = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=7, stride=2, padding=0)

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(368, 50),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(50, 50),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(50, 10),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(10, 10))               


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.preparer(x)
        conv = self.conv_sequence(x)
        skip = self.skip_conv(x).flatten(start_dim=1)
        skip = torch.cat((skip, torch.zeros((skip.shape[0], 368 - skip.shape[1]))), dim=1)

        x = conv.flatten(start_dim=1) + skip

        x = self.full_connected_sequence(x)

        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)  
    

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
    
class ChineseCNN(BaseCNN):

    def __init__(self, preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.conv_sequence = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2))
        

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(64, 50),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(50, 15),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(15, 15))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.preparer(x)
        x = self.conv_sequence(x)

        return self.full_connected_sequence(x.flatten(start_dim=1))
    

class ChineseCrazy(BaseCNN):

    def __init__(self):

        super().__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=2),
            torch.nn.Conv2d(in_channels=1, out_channels=15, kernel_size=16, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Linear(15, 15)
        )
                                     