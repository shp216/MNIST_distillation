import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset class using only cached data
class MNISTDataset(Dataset):
    def __init__(self, cached_images, cached_labels):
        self.cached_images = cached_images
        self.cached_labels = cached_labels

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        img = self.cached_images[idx]
        label = self.cached_labels[idx]

        return img, label