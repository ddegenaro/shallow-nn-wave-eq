from typing import Callable

import torch

def random_data(
    num_samples: int = 2,
    dim: int = 3,
    mins: list[float] = [0., 0., 0.],
    maxes: list[float] = [1., 1., 1.],
    f: Callable = None
):
    assert len(mins) == len(maxes) == dim, (
        f'Dimensions must match, but got: dim={dim}, {len(mins)} mins, {len(maxes)} maxes'
    )

    sample = torch.rand(
        (num_samples, dim), dtype=torch.float32
    )

    for i in range(num_samples):
        sample[:, i] *= (maxes[i] - mins[i])
        sample[:, i] += mins[i]

    if f is not None:
        targets = []

        for i in range(num_samples):
            targets.append(f(*sample[:, i]))

        return sample, torch.tensor(targets, dtype=torch.float32)
    
    return sample, None