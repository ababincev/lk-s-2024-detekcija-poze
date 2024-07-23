import os
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
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
        return len([name for name in os.listdir(self.img_dir)])

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
        mapa = torch.tensor(mapa)
        #self.center_map[:, :, :3] = image[:, :, :] #ili  random_array[:, :, 0]
        return [image, mapa]    

    
class DataLoader1():
    def __init__(self, image_dataset, batch_size):
        self.used = []
        self.image_dataset = image_dataset
        self.batch_size = batch_size
        self.batch = []

#     def vrati(self, i):
#         self.batch = []
#         for x in os.listdir(self.image_dataset.img_dir):
#             if x not in self.used:
#                 idx = self.batch_size * i + len(self.batch)
#                 batch1 = self.image_dataset[idx]
#                 self.batch.append(batch1)
#                 self.used.append(x)

#             if len(self.batch) >= self.batch_size:
#                 y = self.batch
#                 return y