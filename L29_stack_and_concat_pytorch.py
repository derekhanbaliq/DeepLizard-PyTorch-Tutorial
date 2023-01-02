import torch

t1 = torch.tensor([1, 1, 1])

print(t1.unsqueeze(dim=0))
print(t1.unsqueeze(dim=1))

print(t1.shape)
print(t1.unsqueeze(dim=0).shape)
print(t1.unsqueeze(dim=1).shape)

print("-"*50)

t1 = torch.tensor([1, 1, 1])
t2 = torch.tensor([2, 2, 2])
t3 = torch.tensor([3, 3, 3])
print(torch.cat((t1, t2, t3), dim=0))
print(torch.stack((t1, t2, t3), dim=0))
print(torch.cat((t1.unsqueeze(0), t2.unsqueeze(0), t3.unsqueeze(0)), dim=0))  # == stack
