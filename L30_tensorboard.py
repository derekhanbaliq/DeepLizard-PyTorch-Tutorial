import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)  # display options for output
torch.set_grad_enabled(True)  # already on by default

from torch.utils.tensorboard import SummaryWriter

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
# step 1: get batch from the training set
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
optimizer = optim.Adam(network.parameters(), lr=0.01)  # input the weights

images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)

tb = SummaryWriter()
tb.add_image('images', grid)
tb.add_graph(network, images)

for epoch in range(5):  # step 7: do many epochs

    total_loss = 0
    total_correct = 0

    for batch in train_loader:
        images, labels = batch

        preds = network(images)  # step 2: pass batch
        loss = F.cross_entropy(preds, labels)  # step 3: calculate loss

        optimizer.zero_grad()  # zero out the existing grads instead of accumulating them
        loss.backward()  # step 4: calculate gradients of the loss function
        optimizer.step()  # step 5: update weights using the gradients to reduce the loss

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    tb.add_scalar('Loss', total_loss, epoch)
    tb.add_scalar('Number Correct', total_correct, epoch)
    tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)

    tb.add_histogram('conv1.bias', network.conv1.bias, epoch)
    tb.add_histogram('conv1.weight', network.conv1.weight, epoch)
    tb.add_histogram('conv1.weight.grad', network.conv1.weight.grad, epoch)

    # step 6: do these for every batch in 1 epoch
    print("epoch = ", epoch, ", total_correct = ", total_correct, ", loss = ", total_loss)

