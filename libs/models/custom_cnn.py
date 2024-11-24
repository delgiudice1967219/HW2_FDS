import torch
from torch import nn

class CustomCNN(nn.Module):
    def _init_(self):
        super(CustomCNN, self)._init_()
        
        # Define the structure of our CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        
        # Batch Normalization
        self.conv1_bn = nn.BatchNorm2d(16)

        # The rectified linear unit function, used as our activation function for the custom model
        self.relu1 = nn.ReLU()

        # Dropout with a probability of 0.1 for every neuron to be deactivated
        self.dropout1 = nn.Dropout(0.1)

        # Max Pool with stride 2, so we halve the matrix dimension every time we apply this technique
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

        #Starting from images of 32x32 dimension and 3 channels(RGB) as input, we apply twice a MaxPool function with stride equal to 2,
        #we divide by 4 the starting matrix, resulting in a 8x8. In the meantime, we increase the output channels at every convolution
        #to extract the maximum information available, so we arrive at the last convolution with a number of channels at the output of 128.
        #From there, we apply a first fully connection to merge all the infos in 64 channels of output, and in the end we apply a second
        #fully connection to return the 10 channels of the 10 classes of the dataset we want to classify in our network.
        self.fc1 = nn.Linear(128* 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        #Our scheme for this network was a double CONV-ReLU-CONV-ReLU-Pool, so in the end
        #we had 4 layers in total.
        x = self.relu1(self.dropout1(self.conv1_bn(self.conv1(x))))
        x = self.pool(self.relu2(self.dropout2(self.conv2_bn(self.conv2(x)))))
        x = self.relu3(self.dropout3(self.conv3_bn(self.conv3(x))))
        x = self.pool(self.relu4(self.dropout4(self.conv4_bn(self.conv4(x)))))
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = self.fc2(x)
        return x