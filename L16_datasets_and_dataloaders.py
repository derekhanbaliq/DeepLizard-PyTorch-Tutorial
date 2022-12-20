import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data

import numpy as np
import matplotlib.pyplot as plt


train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',  # location on disk
    train=True,  # 60000 for training & 10000 for testing
    download=True,  # download if it's not present at the location we sepecified
    transform=transforms.Compose([transforms.ToTensor()])  # A composition of transformations that should be
    # performed on the dataset elements.
)

# train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000, shuffle=True)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)

torch.set_printoptions(linewidth=120)

# print(len(train_set))  # 60000

# print(train_set.train_labels)  # train_labels have changed name to targets
# print(train_set.targets)  # tensor([9, 0, 0,  ..., 3, 0, 5])

# how many of each label exists in the dataset
# print(train_set.train_labels.bincount())
# print(train_set.targets.bincount())  # tensor([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000])

# print("-"*50)

# sample = next(iter(train_set))  # 返回迭代器的下一个项目 函数要和生成迭代器的 iter() 函数一起使用。
# # iter(), returns an object representing a stream of data
# # next(), get the next data element in the stream of data
# print(len(sample))
# print(type(sample))
# image, label = sample  # image-label pairs / deconstructing the object
# print("image.shape = {}".format(image.shape))
# print("image.label = {}".format(image.label))
#
# plt.imshow(image.squeeze(), cmap='gray')
# plt.show()
# print("label = {}".format(label))

print("-"*50)

batch = next(iter(train_loader))  # this is train loader! Previous one is training set!
print(len(batch))
print(type(batch))

images, labels = batch
print(images.shape)
print(labels.shape)  # 10 images

print("-"*50)

grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(15, 15))
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.show()
print('labels: ', labels)


