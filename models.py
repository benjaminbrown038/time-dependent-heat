# models.py
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden=128, depth=3):
        super().__init__()

        layers = [nn.Linear(3, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
