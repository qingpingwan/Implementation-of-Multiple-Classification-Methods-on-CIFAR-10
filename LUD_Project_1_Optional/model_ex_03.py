import torch
from torch import nn
import torch.nn.functional as F




class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        attention_weights = F.softmax(torch.bmm(query, key), dim=2)
        attention_output = torch.bmm(value, attention_weights.permute(0, 2, 1))
        attention_output = attention_output.view(batch_size, channels, height, width)
        out = self.gamma * attention_output + x

        return out




class CNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.self_attention = SelfAttention(in_channels=16)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=2048, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
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

        x = self.self_attention(x)  # 添加self-attention层

        x = self.maxpool1(x)
        x1 = F.relu(self.conv2(x))
        x1 = self.conv3(x1)
        x = torch.cat([x, x1], dim=1)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x