import torch
import numpy as np

t1 = torch.tensor([
    [1, 2],
    [3, 4]
], dtype=torch.float32)

t2 = torch.tensor([
    [9, 8],
    [7, 6]
], dtype=torch.float32)

print(t1.shape)

print(t1 + t2)
print(t1 + 2)
t = t1 + torch.tensor(
    np.broadcast_to(2, t1.shape)
    , dtype=torch.float32
)
print(t)

print('*'*50)

t1 = torch.tensor([
    [1, 1],
    [1, 1]
], dtype=torch.float32)
print(t1.shape)

t2 = torch.tensor([2, 4], dtype=torch.float32)
print("t2 =", t2.shape)

t2 = np.broadcast_to(t2.numpy(), t1.shape)
print("t2_new =", t2.shape)

print(t1 + t2)


