import torch.nn as nn

class ResNet18(nn.Module):
    """
    ResNet-18 is a convolutional neural network that won the 2015 ImageNet competition. 
    The network architecture consists of two sets of convolutional and average pooling layers,
    followed by a flattening convolutional layer, then two fully-connected layers and finally a softmax
    classifier. It uses skip connections to overcome the vanishing gradient problem.

    The network uses the Cross Entropy Loss function and the Adam optimizer.

    Output:
        torch.Size([b, 10]) -> Probability of each class for each image in the batch b
                               (b images, 10 classes)    
    """
    def __init__(self):
        super(ResNet18, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
        # padding_mode='zeros')

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3) # 3 input image channel, 64 output channels, 7x7 square convolution kernel
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1)
        ) # 64 input image channel, 64 output channels, 3x3 square convolution kernel

        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128, 1)
        ) # 64 input image channel, 128 output channels, 3x3 square convolution kernel

        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 1)
        ) # 128 input image channel, 256 output channels, 3x3 square convolution kernel

        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512, 1)
        ) # 256 input image channel, 512 output channels, 3x3 square convolution kernel

        self.fc = nn.Linear(512, 10)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1) # Max pooling layer with a 3x3 kernel and stride of 2
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass of the network.
        """
        x = self.conv1(x) # Size changes from (3, 32, 32) to (64, 16, 16)
        x = self.bn1(x) # Size changes from (64, 16, 16) to (64, 16, 16)
        x = self.relu(x) # Size changes from (64, 16, 16) to (64, 16, 16)
        x = self.maxpool(x) # Size changes from (64, 16, 16) to (64, 8, 8)

        x = self.layer1(x) # Size changes from (64, 8, 8) to (64, 8, 8)
        x = self.layer2(x) # Size changes from (64, 8, 8) to (128, 4, 4)
        x = self.layer3(x) # Size changes from (128, 4, 4) to (256, 2, 2) 
        x = self.layer4(x) # Size changes from (256, 2, 2) to (512, 1, 1)

        # x = self.avgpool(x) # Size changes from (512, 1, 1) to (512, 1, 1) 

        x = x.view(-1, 512) # Size changes from (512, 1, 1) to (1, 512)
        x = self.fc(x) # Size changes from (1, 512) to (1, 10)
        x = self.softmax(x) # Apply softmax to the output

        return x
    
class BasicBlock(nn.Module):
    """
    Basic block of the ResNet-18 network.
    """
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
        # padding_mode='zeros')
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1) # 3x3 convolution
        self.bn1 = nn.BatchNorm2d(out_channels) # Batch normalization
        self.relu = nn.ReLU(inplace=True) # ReLU activation function
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1) # 3x3 convolution
        self.bn2 = nn.BatchNorm2d(out_channels) # Batch normalization
        
        self.shortcut = nn.Sequential() # Identity

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride), # 1x1 convolution
                nn.BatchNorm2d(out_channels) # Batch normalization
            ) 
        
    def forward(self, x):
        """
        Forward pass of the network.
        """
        identity = x
        
        x = self.conv1(x) # Size changes from (in_channels, H, W) to (out_channels, H, W)
        x = self.bn1(x) # Size changes from (out_channels, H, W) to (out_channels, H, W)
        x = self.relu(x) # Size changes from (out_channels, H, W) to (out_channels, H, W)
        
        x = self.conv2(x) # Size changes from (out_channels, H, W) to (out_channels, H, W)
        x = self.bn2(x) # Size changes from (out_channels, H, W) to (out_channels, H, W)
        
        # In self.shortcut, the identity is passed through a 1x1 convolution and batch normalization
        # Here we apply a residual connection to the output of the second batch normalization layer
        # by adding the identity to the output of the second batch normalization layer.
        x += self.shortcut(identity) # Size changes from (out_channels, H, W) to (out_channels, H, W)
        x = self.relu(x)  # Size changes from (out_channels, H, W) to (out_channels, H, W)

        return x