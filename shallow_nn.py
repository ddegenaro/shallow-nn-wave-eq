import torch
from torch import nn

class Wave2D(nn.Module):

    def __init__(self, width: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, width),
            nn.ReLU(),
            nn.Linear(width, 1)
        )
        

    def forward(self, t, x, y, z):
        return self.network(
            torch.transpose(
                torch.vstack([t, x, y, z]), 0, 1
            )
        )
    
if __name__ == "__main__":
    net = Wave2D(10)

    # randomly init, but "predicting" displacement at the origin for 10 time units
    print(net(
        torch.tensor([i for i in range(10)], dtype=torch.float32),
        torch.tensor([0] * 10, dtype=torch.float32),
        torch.tensor([0] * 10, dtype=torch.float32),
        torch.tensor([0] * 10, dtype=torch.float32)
    ))