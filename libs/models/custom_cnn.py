import torch
from torch import nn

class CustomCNN(nn.Module):
    def _init_(self):
        super(CustomCNN, self)._init_()
        
        # Define the structure of our CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        
        # Batch Normalization
        self.conv1_bn = nn.BatchNorm2d(16)

        # The rectified linear unit function, used as our activation function
        self.relu1 = nn.ReLU()

        # Dropout with a probability of 0.1 for every neuron to be deactivated
        self.dropout1 = nn.Dropout(0.1)

        # Max Pool with stride 2, so we halve the images dimensions every time we apply it
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

        # Our scheme for this network was a double CONV-ReLU-CONV-ReLU-Pool, so in the end
        # we had 4 layers in total.
        
        # Starting with images of 32x32x3 (HeigthxWeigthxChannels) as input
        x = self.relu1(self.dropout1(self.conv1_bn(self.conv1(x))))
        
        # After our first CONV-ReLU-CONV-ReLU-Pool sequence we obtain images of 16x16x32 (HxWxC)
        x = self.pool(self.relu2(self.dropout2(self.conv2_bn(self.conv2(x)))))
        
        x = self.relu3(self.dropout3(self.conv3_bn(self.conv3(x))))
        
        # After our second CONV-ReLU-CONV-ReLU-Pool sequence we obtain images of 8x8x128 (HxWxC)
        x = self.pool(self.relu4(self.dropout4(self.conv4_bn(self.conv4(x)))))
        
        # Stretch to 1 x 8192
        x = x.view(-1, 128 * 8 * 8)

        # Fully Connected Layer that takes in input 1 x 8192 tensor
        # and gives us in output 1 x 64 tensor
        x = self.fc1(x)

        # Fully Connected Layer that takes in input 1 x 64 tensor
        # and gives us in output 1 x 10 tensor        
        x = self.fc2(x)
        return x