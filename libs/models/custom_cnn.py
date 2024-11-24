import torch
from torch import nn
from torch.nn import functional as F

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Define the structure of our CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        
        # Batch Normalization
        self.conv1_bn = nn.BatchNorm2d(16)

        # Activation Function
        self.relu1 = nn.ReLU()

        # Dropout
        self.dropout1 = nn.Dropout(0.1)

        # Max Pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(0.2)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout(0.2)
        self.relu4 = nn.ReLU()

        self.fc1 = nn.Linear(128* 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu1(self.dropout1(self.conv1_bn(self.conv1(x))))
        x = self.pool(self.relu2(self.dropout2(self.conv2_bn(self.conv2(x)))))
        x = self.relu3(self.dropout3(self.conv3_bn(self.conv3(x))))
        x = self.pool(self.relu4(self.dropout4(self.conv4_bn(self.conv4(x)))))
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = self.fc2(x)
        return x