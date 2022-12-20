import torch

t1 = torch.tensor([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
])

t2 = torch.tensor([
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2]
])

t3 = torch.tensor([
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3]
])

t = torch.stack((t1, t2, t3))  # concatenate
print(t.shape)
print(t)

print("-"*50)
t = t.reshape(3, 1, 4, 4)
print(t)
print(t[0])
print(t[0][0])
print(t[0][0][0])
print(t[0][0][0][0])

print("-"*50)
print(t.flatten(start_dim=1).shape)  # 3 x (1 x 4 x 4)
t = t.flatten(start_dim=1)
print(t)

