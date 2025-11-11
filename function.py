import torch

c = 1.0

def f(
    t: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    return 2 * c**2 * t**2 + x**2 + y**2