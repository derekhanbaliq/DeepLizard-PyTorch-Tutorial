import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data


train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',  # location on disk
    train=True,  # 60000 for training & 10000 for testing
    download=True,  # download if it's not present at the location we sepecified
    transform=transforms.Compose([transforms.ToTensor()])  # A composition of transformations that should be
    # performed on the dataset elements.
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000, shuffle=True)

