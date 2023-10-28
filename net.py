import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    "Fully connected neural network for a input image: torch.Size([4, 3, 32, 32])"

    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution kernel
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(3, 6, 5) # torch.Size([4, 6, 28, 28])
        self.pool = nn.MaxPool2d(2, 2) # torch.Size([4, 6, 14, 14])
        self.conv2 = nn.Conv2d(6, 16, 5) # torch.Size([4, 16, 10, 10])
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # torch.Size([4, 120])
        self.fc2 = nn.Linear(120, 84) # torch.Size([4, 84])
        self.fc3 = nn.Linear(84, 10) # torch.Size([4, 10])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
