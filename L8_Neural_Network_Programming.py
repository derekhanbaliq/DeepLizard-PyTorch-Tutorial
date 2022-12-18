import torch

t = torch.Tensor()
print(type(t))

print("t.dtype = {}".format(t.dtype))  # data type
print("t.device = {}".format(t.device))
print("t.layout = {}".format(t.layout))

device = torch.device('cuda:0')
print("device = {}".format(device))

