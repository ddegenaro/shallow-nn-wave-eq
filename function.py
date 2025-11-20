import torch

# c MUST BE DEFINED
c: float = 1.

# # u(t, x, [y, z]) MUST BE DEFINED
def u(t, x, y):
    return f(x + c * t) + f(y + c * t)
    
def f(s):
    return torch.exp(-10. * s**2) / 5.

# def u(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     return f(c * sqrt2 * t + x + y) + g(-c * sqrt5 * t + x - 2 * y)

# # DEFINE HELPER VARS AND FUNCS AS NEEDED
# a = 5.0
# b = 0.0

# sqrt2 = torch.sqrt(torch.tensor(2.))
# sqrt5 = torch.sqrt(torch.tensor(5.))

# def f(s: torch.Tensor) -> torch.Tensor:
#     return torch.exp(-(s - a) ** 2)

# def g(s: torch.Tensor) -> torch.Tensor:
#     return 3 * torch.exp(-4 * (s - b) ** 2)

# def u(
#     t: torch.Tensor,
#     x: torch.Tensor,
#     y: torch.Tensor
# ) -> torch.Tensor:
#     return 2 * c**2 * t**2 + x**2 + y**2
