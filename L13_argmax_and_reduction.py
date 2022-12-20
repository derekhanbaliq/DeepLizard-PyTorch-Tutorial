import torch
import numpy as np

# t = torch.tensor([
#     [0, 1, 0],
#     [2, 0, 2],
#     [0, 3, 0]
# ], dtype=torch.float32)

t = torch.tensor([
    [1, 0, 0, 2],
    [0, 3, 3, 0],
    [4, 0, 0, 5]
], dtype=torch.float32)

print(t.max())
print(t.argmax())
print("---")
print(t.max(dim=0))  # squeeze row and find the max
print(t.max(dim=1))
print(t.argmax(dim=1))
print("---")

t = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=torch.float32)
print(t.mean())
print(t.mean().item())  # item() get the value
