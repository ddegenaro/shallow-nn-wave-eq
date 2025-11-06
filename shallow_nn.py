import torch
from torch import nn

class Wave3D(nn.Module):

    def __init__(self, width: int = 10, c: float = 1.):
        super().__init__()

        self.position_weights_plus = nn.Parameter(torch.randn((3, width)))
        self.position_weights_minus = nn.Parameter(torch.randn((3, width)))
        self.output_weights = nn.Linear(width, 1)
        self.c = c
        self.activation = nn.ReLU()

    def forward(self, t, x, y, z) -> torch.Tensor:

        pos_out = torch.hstack([x, y, z]) @ self.position_weights_plus

        time_out = t @ (
            torch.sqrt(self.c * (self.position_weights_plus ** 2).sum(0, keepdim=True)) -
            torch.sqrt(self.c * (self.position_weights_minus ** 2).sum(0, keepdim=True))
        )

        return self.output_weights(
            self.activation(time_out + pos_out)
        )
    
    def u(self, t, x, y, z) -> torch.Tensor:
        return self.forward(t, x, y, z)
    
if __name__ == "__main__":

    batch_size = 32

    net = Wave3D(10)

    # predict at some random times/places
    t, x, y, z = (
        torch.randn((batch_size,1)),
        torch.randn((batch_size,1)),
        torch.randn((batch_size,1)),
        torch.randn((batch_size,1))
    )

    out = net.u(t, x, y ,z)
    
    assert out.shape[0] == batch_size