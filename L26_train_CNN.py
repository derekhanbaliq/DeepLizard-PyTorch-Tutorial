import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)  # display options for output
torch.set_grad_enabled(True)  # already on by default

print(torch.__version__)
print(torchvision.__version__)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t0):
        # input layer
        t = t0

        # hidden conv layer
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # hidden conv layer
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # hidden linear layer
        t = F.relu(self.fc1(t.reshape(-1, 12 * 4 * 4)))

        # hidden linear layer
        t = F.relu(self.fc2(t))

        # output layer
        t = self.out(t)
        # t = F.softmax(t, dim=1)

        return t


train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',  # location on disk
    train=True,  # 60000 for training & 10000 for testing
    download=True,  # download if it's not present at the location we specified
    transform=transforms.Compose([
        transforms.ToTensor()
    ])  # A composition of transformations that should be performed on the dataset elements.
)

network = Network()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
batch = next(iter(train_loader))
images, labels = batch

preds = network(images)
print("get_num_correct = ", get_num_correct(preds, labels))

# calculating the loss
print("calculating the loss: ")
loss = F.cross_entropy(preds, labels)
print("loss = ", loss.item())

# calculating the gradients
print("-"*50)
print("calculating the gradients: ")
print(network.conv1.weight.grad)
loss.backward()
print(network.conv1.weight.grad.shape)

# updating the weights
print("-"*50)
print("updating the weights: ")

print("get_num_correct = ", get_num_correct(preds, labels))
print("old loss = ", loss.item())

optimizer = optim.Adam(network.parameters(), lr=0.01)  # lr is a hyperparameter
optimizer.step()  # updating the weights

preds = network(images)
loss = F.cross_entropy(preds, labels)
print("optimized loss = ", loss.item())
print("get_num_correct = ", get_num_correct(preds, labels))

