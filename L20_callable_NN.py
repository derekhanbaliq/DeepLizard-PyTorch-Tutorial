import torch
import torch.nn as nn

# exm 1
# define a linear layer that accepts 4 in features and transforms these into 3 out features
in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
weight_matrix = torch.tensor([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6]
], dtype=torch.float32)
print(weight_matrix.matmul(in_features))

print("-"*50)

# exm 2
fc = nn.Linear(in_features=4, out_features=3, bias=False)
fc.weight = nn.Parameter(weight_matrix)
print(fc(in_features))  # object instance / callable Python objects

print("-"*50)

fc = nn.Linear(in_features=4, out_features=3)  # linear layer
t = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
output = fc(t)
print(output)

