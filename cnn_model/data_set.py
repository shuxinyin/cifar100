import numpy as np
import torch
import os
import pickle
from torch.utils.data import Dataset


# 1. load data function
def load_cifar100(data_dir, file_name):
    with open(os.path.join(data_dir, file_name), 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        labels = data[b'fine_labels']
        images = data[b'data']
    print(f"data length: {len(labels)}, {len(images)}")
    return labels, images


# 2. define data_set class
class Cifar100(Dataset):
    def __init__(self, dirname, filename, train=True):
        super(Cifar100).__init__()
        if train:
            self.labels, self.images = load_cifar100(dirname, filename)
        else:
            self.labels, self.images = load_cifar100(dirname, filename)
        self.images = self.images.reshape(-1, 3, 32, 32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        image = image.astype(np.float32)
        image = torch.from_numpy(image).div(255.0)  # transform ndarray to tensor
        label = self.labels[index]
        label = int(label)
        return image, label


# 3. pack self-define dataset into torch DataLoader
from torch.utils.data import DataLoader

dirname = "../data/cifar-100-data"
with open(os.path.join(dirname, "train"), 'rb') as f:
    print()
train_dataset = Cifar100(dirname, "train")
test_dataset = Cifar100(dirname, "test")

train_dataloader = DataLoader(train_dataset, shuffle=True,
                              batch_size=16, num_workers=0)
test_dataloader = DataLoader(test_dataset, shuffle=True,
                             batch_size=16, num_workers=0)

# with open("../data/cifar-100-data/train", 'rb') as f:
#     d = pickle.load(f, encoding='bytes')
# print(f"data shape, {type(d)}, {d.keys()}, {d[b'data'][0]}")
