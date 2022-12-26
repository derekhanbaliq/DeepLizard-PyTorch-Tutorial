import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)  # callable function
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # implement the forward pass
        return t

    # def __repr__(self):
    #     return "lizard_net"


network = Network()
print(network)

print("-" * 50)

print(network.conv1)

# print("-"*50)

# print(network.conv2.weight)

print("-" * 50)

print(network.conv1.weight.shape)  # rank-4 tensor, [Number of filters, Depth, Height, Width]
print(network.conv2.weight.shape)
# 1. All filters are represented using a single tensor.
# 2. Filters have depth that accounts for the input channels.

print(network.fc1.weight.shape)
print(network.fc2.weight.shape)
print(network.out.weight.shape)

# print("-" * 50)
#
# in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
# weight_matrix = torch.tensor([
#     [1, 2, 3, 4],
#     [2, 3, 4, 5],
#     [3, 4, 5, 6]
# ], dtype=torch.float32)
#
# print(weight_matrix.matmul(in_features))

print("-" * 50)

for param in network.parameters():
    print(param.shape)

for name, param in network.named_parameters():
    print(name, '\t\t', param.shape)

