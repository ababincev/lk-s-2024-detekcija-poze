import os
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, heat_maps_dir, img_dir, transform, target_transform=None):
        self.heat_maps_dir = heat_maps_dir
        self.img_dir = img_dir
        self.transform = transform = transforms.Compose([
            transforms.Resize(transform),
            ])
        self.target_transform = None

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        image = read_image(img_path)


        map_path = os.path.join(self.heat_maps_dir, os.listdir(self.heat_maps_dir)[idx])
        mapa = np.load(map_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mapa = self.target_transform(mapa)
        #random_array[:, :, :3] = image[:, :, :] #ili  random_array[:, :, 0]
        #return random_aray, mapaimport os
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, heat_maps_dir, img_dir, transform, center_map,  target_transform=None):
        self.heat_maps_dir = heat_maps_dir
        self.img_dir = img_dir
        self.transform = transform = transforms.Compose([
            transforms.Resize(transform),
            ])
        self.target_transform = None
        self.center_map = center_map

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        image = read_image(img_path)
        image = image / 255

        map_path = os.path.join(self.heat_maps_dir, os.listdir(self.heat_maps_dir)[idx])
        mapa = np.load(map_path, allow_pickle=True)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mapa = self.target_transform(mapa)
        #self.center_map[:, :, :3] = image[:, :, :] #ili  random_array[:, :, 0]
        return image, mapa