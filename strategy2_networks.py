
import torch
from base_cnn import BaseCNN
from typing import List, Tuple
from utils.cnn_size_calcs import cnn_mec_calc, full_cnn_seq_size

class Strat2Base(BaseCNN):
        
    def _fully_connected_mec(self) -> int:
        #print("In features 1", self.full_connected_sequence[0].in_features, "In features 2", self.full_connected_sequence[2].in_features)
        return self.full_connected_sequence[0].in_features + min(self.full_connected_sequence[2].in_features, self.full_connected_sequence[0].in_features)
    
    def get_mec(self) -> int:
        #print("CNN MEC:", cnn_mec_calc(self.conv_sequence, self.input_size), "Fully connected MEC:", self._fully_connected_mec())
        return cnn_mec_calc(self.conv_sequence, self.input_size) + self._fully_connected_mec()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.preparer(x)
        x = self.conv_sequence(x).flatten(start_dim=1)

        return self.full_connected_sequence(x), x
    
class CNNBestGuess(Strat2Base):

    def __init__(self, input_size: Tuple[int, int, int], preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.input_size = input_size

        self.conv_sequence = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.cnn_out_size = full_cnn_seq_size(in_size=input_size, cnn_sequence=self.conv_sequence)[-1]

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(self.cnn_out_size[0]*self.cnn_out_size[1]*self.cnn_out_size[2], 15),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(15, 15))
        
class CNNLow(Strat2Base):

    def __init__(self, input_size: Tuple[int, int, int], preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.input_size = input_size

        self.conv_sequence = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=3),
                    torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=3))
        
        self.cnn_out_size = full_cnn_seq_size(in_size=input_size, cnn_sequence=self.conv_sequence)[-1]

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(self.cnn_out_size[0]*self.cnn_out_size[1]*self.cnn_out_size[2], 15),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(15, 15))
        

class CNNHigher1(Strat2Base):

    def __init__(self, input_size: Tuple[int, int, int], preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.input_size = input_size

        self.conv_sequence = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.cnn_out_size = full_cnn_seq_size(in_size=input_size, cnn_sequence=self.conv_sequence)[-1]

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(self.cnn_out_size[0]*self.cnn_out_size[1]*self.cnn_out_size[2], 15),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(15, 15))
        
class CNNHigher2(Strat2Base):

    def __init__(self, input_size: Tuple[int, int, int], preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.input_size = input_size

        self.conv_sequence = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.cnn_out_size = full_cnn_seq_size(in_size=input_size, cnn_sequence=self.conv_sequence)[-1]

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(self.cnn_out_size[0]*self.cnn_out_size[1]*self.cnn_out_size[2], 15),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(15, 15))
        
class CNNHigher3(Strat2Base):

    def __init__(self, input_size: Tuple[int, int, int], preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.input_size = input_size

        self.conv_sequence = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=6, out_channels=8, kernel_size=3, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.cnn_out_size = full_cnn_seq_size(in_size=input_size, cnn_sequence=self.conv_sequence)[-1]

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(self.cnn_out_size[0]*self.cnn_out_size[1]*self.cnn_out_size[2], 15),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(15, 15))
        

class CNNHigher4(Strat2Base):

    def __init__(self, input_size: Tuple[int, int, int], preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.input_size = input_size

        self.conv_sequence = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=7, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=6, out_channels=10, kernel_size=5, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.cnn_out_size = full_cnn_seq_size(in_size=input_size, cnn_sequence=self.conv_sequence)[-1]

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(self.cnn_out_size[0]*self.cnn_out_size[1]*self.cnn_out_size[2], 15),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(15, 15))
        
        
class CNNVeryHigh1(Strat2Base):

    def __init__(self, input_size: Tuple[int, int, int], preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.input_size = input_size

        self.conv_sequence = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=11, stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.cnn_out_size = full_cnn_seq_size(in_size=input_size, cnn_sequence=self.conv_sequence)[-1]

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(self.cnn_out_size[0]*self.cnn_out_size[1]*self.cnn_out_size[2], 15),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(15, 15))
        
        
class CNNFullMec(Strat2Base):

    def __init__(self, input_size: Tuple[int, int, int], preparer: torch.nn.Module = torch.nn.Identity(), criteria: torch.nn.Module = torch.nn.CrossEntropyLoss()):
        super().__init__(preparer=preparer, criteria=criteria)

        self.input_size = input_size

        self.conv_sequence = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=13, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=9, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.cnn_out_size = full_cnn_seq_size(in_size=input_size, cnn_sequence=self.conv_sequence)[-1]

        self.full_connected_sequence = torch.nn.Sequential(torch.nn.Linear(self.cnn_out_size[0]*self.cnn_out_size[1]*self.cnn_out_size[2], 15),
                                                           torch.nn.ReLU(),
                                                           torch.nn.Linear(15, 15))
        

STRAT1_TEST_LIST: List[Strat2Base] = [CNNLow, CNNBestGuess, CNNHigher1, CNNHigher2, CNNHigher3, CNNHigher4, CNNVeryHigh1, CNNFullMec]


        

