import torch

t = torch.tensor([1, 2, 3])
print(t)

t = t.cuda()  # to be performed on the GPU
print(t)

