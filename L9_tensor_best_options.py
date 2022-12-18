import torch
import numpy as np

print(torch.tensor(np.array([1, 2, 3])))
print(torch.tensor(np.array([1., 2., 3.])))
print(torch.tensor(np.array([1, 2, 3]), dtype=torch.float64))

print("---")
data = np.array([1, 2, 3])
t1 = torch.Tensor(data)
t2 = torch.tensor(data)
t3 = torch.as_tensor(data)
t4 = torch.from_numpy(data)
data[0: 3] = [0, 0, 0]  # change the data
print("torch.Tensor(data) = {}".format(t1))
print("torch.tensor(data) = {}".format(t2))
print("torch.as_tensor(data) = {}".format(t3))
print("torch.from_numpy(data) = {}".format(t4))

print("---")
print("torch.eye(2) = \n{}".format(torch.eye(2)))
print("torch.zeros(2, 2) = \n{}".format(torch.zeros(2, 2)))
print("torch.ones(2, 2) = \n{}".format(torch.ones(2, 2)))
print("torch.rand(2, 2) = \n{}".format(torch.rand(2, 2)))

