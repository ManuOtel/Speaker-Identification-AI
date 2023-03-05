"""Voice Recognition AI project.

This module contains the Convolutional Neural Network Architecture for the Voice Recognition AI project.

@Author: Emanuel-Ionut Otel
@Company: University of Southern Denmark
@Created: 2021-09-22
@Contact: emote21@student.sdu.dk
"""

#### ---- IMPORT AREA ---- ####
from torch import nn
from torchsummary import summary
#### ---- IMPORT AREA ---- ####

class CNN_Voice(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(8)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(8)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(16)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(16)
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(5712, 128)
        self.linear3 = nn.Linear(128, 32)
        self.linear4 = nn.Linear(32, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear3(x)
        logits = self.linear4(x)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":
    cnn = CNN_Voice()
    summary(cnn.cuda(), (1, 64, 216))
