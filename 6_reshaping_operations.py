import torch

t = torch.tensor([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3]
], dtype=torch.float32)
print(t)

print(t.size())
print(t.shape)
print(len(t.shape))

print(torch.tensor(t.shape).prod())
print(t.numel())

print("---")
print(t.reshape([1, 12]))
print(t.reshape([1, 12]).shape)
print(t.reshape([1, 12]).squeeze())
print(t.reshape([1, 12]).squeeze().shape)
print(t.reshape([1, 12]).squeeze().unsqueeze(dim=0))
print(t.reshape([1, 12]).squeeze().unsqueeze(dim=0).shape)

print(t.reshape(2, 2, 3))

# flatten


def flatten(t):  # reshape & squeeze
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t


print(flatten(t))
print(t.reshape([1, 12]))  # only reshape



