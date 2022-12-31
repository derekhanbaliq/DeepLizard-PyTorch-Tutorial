import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)  # display options for output
torch.set_grad_enabled(True)  # already on by default

# print(torch.__version__)
# print(torchvision.__version__)


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
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)  # input the weights

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

    # step 6: do these for every batch in 1 epoch
    print("epoch = ", epoch, ", total_correct = ", total_correct, ", loss = ", total_loss)

print(train_set)
print(train_set.targets)
print(len(train_set))
print(len(train_set.targets))

print("-" * 50)


@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds


prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
train_preds = get_all_preds(network, prediction_loader)
print(train_preds.shape)

print(train_preds.requires_grad)
print(train_preds.grad)
print(train_preds.grad_fn)

print("-" * 50)

preds_correct = get_num_correct(train_preds, train_set.targets)
print("total correct: ", preds_correct)
print("accuracy: ", preds_correct / len(train_set))

print("-" * 50)

stacked = torch.stack((train_set.targets, train_preds.argmax(dim=1)), dim=1)
print(stacked.shape)  # torch.Size([60000, 2])
print(stacked)  # true label & predicted label
# print(stacked[0].tolist())  # [9, 9]

print("-" * 50)

cmt = torch.zeros(10, 10, dtype=torch.int32)
# print(cmt)

for p in stacked:
    true_label, predicted_label = p.tolist()
    cmt[true_label, predicted_label] = cmt[true_label, predicted_label] + 1

print(cmt)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix

print("-" * 50)

cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
print(type(cm))
print(cm)

print("-" * 50)

names = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, names)
