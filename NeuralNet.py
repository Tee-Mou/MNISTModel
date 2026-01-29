import torch
from torch import nn
import torch.nn.functional as F

# nn.BCEWithLogitsLoss

class MNISTModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = nn.MaxPool2d(kernel_size=2, padding=1)
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(kernel_size=5, padding=2, stride=1, in_channels=1, out_channels=32)
        self.conv2 = nn.Conv2d(kernel_size=5, padding=2, stride=1, in_channels=32, out_channels=32)
        self.fc1 = nn.Linear(in_features=2048, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
