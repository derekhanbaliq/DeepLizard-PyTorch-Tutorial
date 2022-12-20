import torch

t = torch.tensor([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3]
], dtype=torch.float32)
print(t)

print(t.size())  # torch.Size([3, 4])
print(t.shape)  # torch.Size([3, 4])
print(len(t.shape))

print(torch.tensor(t.shape).prod())  # shape -> product
print(t.numel())

print("---")
print(t.reshape([2, 6]))
print(t.reshape([2, 6]).shape)
print(t.reshape([1, 12]).squeeze())  # dimension becomes 1
print(t.reshape([1, 12]).squeeze().shape)
print(t.reshape([1, 12]).squeeze().unsqueeze(dim=0))
print(t.reshape([1, 12]).squeeze().unsqueeze(dim=0).shape)

print("---")
print(t.reshape(2, 2, 3))


def flatten(t):  # reshape & squeeze
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t


print(flatten(t))
print(t.reshape([1, 12]))  # only reshape



