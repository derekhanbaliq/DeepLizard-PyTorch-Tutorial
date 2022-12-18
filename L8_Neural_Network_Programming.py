import torch
import numpy as np

t = torch.Tensor()
print(type(t))

print("t.dtype = {}".format(t.dtype))  # data type
print("t.device = {}".format(t.device))
print("t.layout = {}".format(t.layout))

device = torch.device('cuda:0')
print("device = {}".format(device))

print("---")
data = np.array([1, 2, 3])
print("type(data) = {}".format(type(data)))

t1 = torch.Tensor(data)
t2 = torch.tensor(data)
t3 = torch.as_tensor(data)
t4 = torch.from_numpy(data)
print("torch.Tensor(data) = {}".format(t1))
print("torch.tensor(data) = {}".format(t2))
print("torch.as_tensor(data) = {}".format(t3))
print("torch.from_numpy(data) = {}".format(t4))
print(t1.dtype)
print(t2.dtype)
print(t3.dtype)
print(t4.dtype)

print("---")
print("torch.eye(2) = \n{}".format(torch.eye(2)))
print("torch.zeros(2, 2) = \n{}".format(torch.zeros(2, 2)))
print("torch.ones(2, 2) = \n{}".format(torch.ones(2, 2)))
print("torch.rand(2, 2) = \n{}".format(torch.rand(2, 2)))

print("---")
t5 = torch.tensor(np.array([1, 2, 3]), dtype=torch.float64)
print(type(t5))



