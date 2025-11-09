import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

class WaveLoss(_WeightedLoss):
    def __init__(self, weight: torch.Tensor):
        super().__init__(weight)

    def forward( # TODO: incorporate first derivative here or directly in Wave module
        self,
        input: torch.Tensor,
        target: torch.Tensor # TODO: likely need more args for initial conditions
    ):
        data_loss = self.weight[0] * F.mse_loss(input, target)
        return 0 + data_loss # TODO: weight and sum losses



class Wave(nn.Module):

    def __init__(
        self,
        width: int = 10,
        c: float = 1.,
        input_dim: int = 1,
        output_dim: int = 1,
        activation: nn.Module = nn.ELU()
    ):
        super().__init__()

        self.width = width
        self.c = c
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        self.position_weight_plus = nn.Parameter(torch.randn((self.input_dim, self.width)))
        self.position_weight_minus = nn.Parameter(torch.randn((self.input_dim, self.width)))

        self.output_weight = nn.Linear(width, self.output_dim)

    def forward(self, t, x):

        pos_out_plus = x @ self.position_weight_plus
        pos_out_minus = x @ self.position_weight_minus

        time_out_plus = self.c * (
            t @ torch.sqrt((self.position_weight_plus ** 2).sum(0, keepdim=True))
        )
        time_out_minus = - self.c * (
            t @ torch.sqrt((self.position_weight_minus ** 2).sum(0, keepdim=True))
        )

        return self.output_weight(
            self.activation(time_out_plus + pos_out_plus) + 
            self.activation(time_out_minus + pos_out_minus)
        )
    
if __name__ == "__main__":

    batch_size = 32
    width = 10
    c = 1.
    input_dim = 3
    output_dim = 1
    activation = nn.ELU()

    # predict at some random times/places
    t, x = (
        torch.randn((batch_size, 1)), # time is 1D
        torch.randn((batch_size, input_dim))
    )

    model = Wave(
        width = width,
        c = c,
        input_dim = input_dim,
        output_dim = output_dim,
        activation = activation
    )

    output = model(t, x)

    print(f'model(t in {t.shape}, x in {x.shape}) -> output in {output.shape}')