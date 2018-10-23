
import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """An implementation of CNN network
    
    Args:
        num_actions: int, the number of possible actions.
    """
    def __init__(self, num_actions):
        super().__init__()
        orthogonal = nn.init.orthogonal_
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        orthogonal(self.conv1.weight)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        orthogonal(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        orthogonal(self.conv3.weight)
        self.fc = nn.Linear(3136, 512)
        orthogonal(self.fc.weight)
        self.logits = nn.Linear(512, num_actions)
        orthogonal(self.logits.weight)
        self.value = nn.Linear(512, 1)
        orthogonal(self.value.weight)

    def forward(self, x):
        x = x / 255.
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        logits = self.logits(x)
        value = self.value(x)
        return logits, value