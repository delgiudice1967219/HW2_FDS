import torch.nn as nn
import torch

class PoorPerformingCNN(nn.Module):
    def __init__(self):
        super(PoorPerformingCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Update the input to match the conv1 output 
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Update the input and output dimensions
        # The input dimensions after all the previous operation
        # are going to be 8x8x8 (HxWxC)
        # The output is 10 to match the number of classes
        self.fc1 = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        
        # We start with input images of 32x32x3
        # After the first conv and pool we get images
        # 16x16x4 (HxWxC)
        x = self.pool(self.relu1(self.conv1(x)))
        
        # After the second conv and pool we get images
        # 8x8x8 (HxWxC)
        x = self.pool(self.relu2(self.conv2(x)))
        
        # Stretch to 1 x 512
        x = x.view(-1, 8 * 8 * 8)

        # Fully Connected Layer that takes in input a 1 x 512 tensor
        # and gives us in output a 1 x 10 tensor 
        x = self.fc1(x)
        return x