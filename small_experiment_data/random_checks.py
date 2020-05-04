import torch

x = torch.tensor([[1], [2], [3]])
print(x)
y=x.expand(3,-1)
print(y)