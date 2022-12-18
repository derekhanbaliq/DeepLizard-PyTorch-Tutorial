import torch

dd = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
print(dd)
print("type(dd) = {}".format(type(dd)))

t = torch.tensor(dd)
print(t)
print("type(t) = {}".format(type(t)))
print("t.shape = {}".format(t.shape))


print("t.reshape(1, 9) = {}".format(t.reshape(1, 9)))

print("t.reshape(1, 9).shape = {}".format(t.reshape(1, 9).shape))

