from old_files.model import BaseCNN
import torch

class StandardCNNBlock(torch.nn.Module):

    def __init__(self, in_channels):
        super(StandardCNNBlock, self).__init__()

        self.conv_seq = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2))
        
    def forward(self, x):
        return self.conv_seq(x)
    
def gen_standard_blocks(in_channels, num_blocks):

    blocks = []
    for _ in range(num_blocks):
        blocks.append(StandardCNNBlock(in_channels))
        in_channels *= 2

    return torch.nn.Sequential(*blocks)

class DynamicCapacityCNN(BaseCNN):

    def __init__(self, num_blocks: int, preparer: torch.Tensor = torch.nn.Identity(), end_block = torch.nn.Identity()):
        super(DynamicCapacityCNN, self).__init__(preparer=preparer)

        self.cnn_sequence = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            gen_standard_blocks(4, num_blocks),
            end_block)


        # Dynamically check linear layer input size
        out_shape = self.cnn_sequence(torch.zeros((1, 1, 64, 64))).shape
        line_in_size = out_shape[1] * out_shape[2] * out_shape[3]
        mid_size = max(line_in_size // 2, 15)

        #assert line_in_size >= 15 and mid_size >= 15, "Input size too small for linear layer"

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(line_in_size, mid_size),
            torch.nn.ReLU(),
            torch.nn.Linear(mid_size, 15),
            torch.nn.ReLU(),
            torch.nn.Linear(15, 15)
        )

        print("Line in size:", line_in_size, "Mid size:", mid_size, "Out size:", 15)

        self.mec = mid_size*line_in_size + min(mid_size, mid_size*15)  

    def get_mec(self):
        return self.mec
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.preparer(x)
        x = self.cnn_sequence(x)

        #print("Shape after CNN:", x.shape)
        #print("Flat shape:", x.shape[1]*x.shape[2]*x.shape[3])

        return self.feed_forward(x.flatten(start_dim=1))
    

class DynamicCapacityCNN2(BaseCNN):

    def __init__(self, num_blocks: int, preparer: torch.Tensor = torch.nn.Identity(), end_block = torch.nn.Identity()):
        super(DynamicCapacityCNN2, self).__init__(preparer=preparer)

        self.cnn_sequence = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            gen_standard_blocks(4, num_blocks),
            end_block)


        # Dynamically check linear layer input size
        out_shape = self.cnn_sequence(torch.zeros((1, 1, 64, 64))).shape
        line_in_size = out_shape[1] * out_shape[2] * out_shape[3]

        #assert line_in_size >= 15 and mid_size >= 15, "Input size too small for linear layer"

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(line_in_size, 15),
        )

        print("Line in size:", line_in_size, "Out size:", 15)

        self.mec = line_in_size*line_in_size + min(line_in_size, 15*line_in_size) 

    def get_mec(self):
        return self.mec
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.preparer(x)
        x = self.cnn_sequence(x)

        #print("Shape after CNN:", x.shape)
        #print("Flat shape:", x.shape[1]*x.shape[2]*x.shape[3])

        return self.feed_forward(x.flatten(start_dim=1))
        
