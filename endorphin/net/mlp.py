import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, num_actions):
        super(MLP, self).__init__()
        self.fc = nn.Linear(4, 128)
        self.logits = nn.Linear(128, num_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        logits = self.logits(x)
        value = self.value(x)
        return logits, value
