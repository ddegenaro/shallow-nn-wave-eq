import torch
from torch import nn

class Wave3D(nn.Module):

    def __init__(self, width: int = 10, c: float = 1.):
        super().__init__()

        self.position_weights = nn.Parameter(torch.randn((3, width)))
        self.output_weights = nn.Linear(width, 1)
        self.c = c
        self.activation = nn.ReLU()

    def forward(self, t, x, y, z) -> torch.Tensor:

        pos_out = torch.hstack([x, y, z]) @ self.position_weights

        time_out = t @ torch.sqrt(self.c*(self.position_weights**2).sum(0, keepdim=True))

        return self.output_weights(
            self.activation(time_out + pos_out)
        )
    
    def u(self, t, x, y, z) -> torch.Tensor:
        return self.forward(t, x, y, z)
    
if __name__ == "__main__":
    net = Wave3D(10)

    # predict at some random times/places
    t, x, y, z = torch.randn((32, 1)), torch.randn((32, 1)), torch.randn((32, 1)), torch.randn((32, 1))

    out = net.u(t, x, y ,z)

    print(out.shape)