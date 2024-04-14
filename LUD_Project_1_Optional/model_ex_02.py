import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=2048, out_features=128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        nn.init.constant_(self.conv3.bias, 0)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.maxpool1(x)

        x1 = F.relu(self.conv2(x))
        x1 = self.bn2(x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        x = torch.cat([x, x1], dim=1)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x