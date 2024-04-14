import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    '''
    You are supposed to implement a two-layer MLP here.
    compute the input channels and output channels, set the hidden channels to 128
    '''
    def __init__(self, in_channels=3*32*32, out_channels=10, hidden_channels=128):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2_1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2_2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2_3 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, out_channels)
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.kaiming_uniform_(self.fc2_1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2_1.bias, 0)
        nn.init.kaiming_uniform_(self.fc2_2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2_2.bias, 0)
        nn.init.kaiming_uniform_(self.fc2_3.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2_3.bias, 0)
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2_1(x))
        x = F.relu(self.fc2_2(x))
        x = F.relu(self.fc2_3(x))

        x = self.fc3(x)
        x = self.softmax(x)
        return x
