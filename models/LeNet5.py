import torch.nn as nn

class LeNet5(nn.Module):
    """
    The network architecture is based on the LeNet-5 architecture. 

    The network consists of the following layers:
    1. Convolutional layer (6 filters, 5x5 kernel, ReLU activation function)
    2. Max pooling layer (2x2 kernel)
    3. Convolutional layer (16 filters, 5x5 kernel, ReLU activation function)
    4. Max pooling layer (2x2 kernel)
    5. Fully connected layer (120 neurons, ReLU activation function)
    6. Dropout layer (20% dropout probability)
    7. Fully connected layer (84 neurons, ReLU activation function)
    8. Dropout layer (20% dropout probability)
    9. Fully connected layer (10 neurons, Softmax activation function)

    The network uses the Cross Entropy Loss function and the Adam optimizer.

    Output:
        torch.Size([b, 10]) -> Probability of each class for each image in the batch b
                               (b images, 10 classes)
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution kernel
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.conv1 = nn.Conv2d(3, 6, 5) # 3 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input image channel, 16 output channels, 5x5 square convolution kernel
        self.max_pool = nn.MaxPool2d(2, 2) # Max pooling over a (2, 2) window

        self.fc1 = nn.Linear(16 * 5 * 5, 120) # an affine operation: y = Wx + b
        self.fc2 = nn.Linear(120 , 84) # an affine operation: y = Wx + b
        self.fc3 = nn.Linear(84, 10) # an affine operation: y = Wx + b

        self.dropout = nn.Dropout(0.2) # Dropout layer with 20% dropout probability
        self.relu = nn.ReLU() # ReLU activation function
        self.softmax = nn.Softmax(dim=1) # Softmax activation function


    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x) # Input: torch.Size([4, 3, 32, 32]) -> Output: torch.Size([4, 6, 28, 28])
        x = self.relu(x) # Input: torch.Size([4, 6, 28, 28]) -> Output: torch.Size([4, 6, 28, 28])
        x = self.max_pool(x) # Input: torch.Size([4, 6, 28, 28]) -> Output: torch.Size([4, 6, 14, 14])

        x = self.conv2(x) # Input: torch.Size([4, 6, 14, 14]) -> Output: torch.Size([4, 16, 10, 10])
        x = self.relu(x) # Input: torch.Size([4, 16, 10, 10]) -> Output: torch.Size([4, 16, 10, 10])
        x = self.max_pool(x) # Input: torch.Size([4, 16, 10, 10]) -> Output: torch.Size([4, 16, 5, 5])

        # Flatten the image
        x = x.view(-1, 16 * 5 * 5) # Input: torch.Size([4, 16, 5, 5]) -> Output: torch.Size([4, 400])

        # Fully connected layers
        x = self.fc1(x) # Input: torch.Size([4, 400]) -> Output: torch.Size([4, 120])
        x = self.relu(x) # Input: torch.Size([4, 120]) -> Output: torch.Size([4, 120])
        x = self.dropout(x) # Input: torch.Size([4, 120]) -> Output: torch.Size([4, 120])

        x = self.fc2(x) # Input: torch.Size([4, 120]) -> Output: torch.Size([4, 84])
        x = self.relu(x) # Input: torch.Size([4, 84]) -> Output: torch.Size([4, 84])
        x = self.dropout(x) # Input: torch.Size([4, 84]) -> Output: torch.Size([4, 84])

        x = self.fc3(x) # Input: torch.Size([4, 84]) -> Output: torch.Size([4, 10])
        x = self.softmax(x) # Input: torch.Size([4, 10]) -> Output: torch.Size([4, 10])

        return x
    
