from typing import Callable, Union

import torch

def random_data(
    num_samples: int = 100,
    spatial_dim: int = 2,
    mins: list[float] = [0., 0., 0.],
    maxes: list[float] = [1., 1., 1.],
    f: Callable = None,
    noise_scale: float = 1e-3
) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
    
    """
    Generate a random physical dataset.

    Args:
        num_samples (`int`): The number of samples to generate.
        dim (`int`): The number of spatial dimensions to involve.
        mins (`list[float]`): The minimum value each of the spatio-temporal dimensions should take. The first dimension is interpreted as time.
        maxes (`list[float]`): The corresponding maximum values.
        f (`Callable`): The function for generating targets. Expected signature is `f(t, x, [y, z])`.

    Returns:
        `tuple[torch.Tensor, Union[torch.Tensor, None]]` where the first entry is the inputs and the second is the targets (`None` if `f` is `None`).
    """

    l_mins, l_maxes = len(mins), len(maxes)
    assert l_mins == l_maxes == 1 + spatial_dim, (
        f'spatial_dim should be one less than number of mins/maxes, but got: spatial_dim={spatial_dim}, {l_mins} mins, {l_maxes} maxes'
    )

    sample = torch.rand(
        (1 + spatial_dim, num_samples), dtype=torch.float32
    )

    for i in range(1 + spatial_dim):
        sample[i, :] *= (maxes[i] - mins[i])
        sample[i, :] += mins[i]

    if f is not None:
        targets: torch.Tensor = f(*sample).transpose(0, -1)
        targets += torch.randn(*targets.shape) * noise_scale
        return sample.transpose(0, -1), targets
    else:
        return sample.transpose(0, -1), None