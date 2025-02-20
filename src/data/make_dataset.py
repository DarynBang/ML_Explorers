import torch
from torch.utils.data import Dataset

# Custom Dataset class for 3D inputs
class Custom3DDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)  # Convert to tensor
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to tensor

    def __len__(self):
        return len(self.data)  # Number of samples

    def __getitem__(self, idx):
        x = self.data[idx].unsqueeze(0)  # Add channel dimension -> (1, 28, 28, 28)
        y = self.labels[idx]  # Label
        return x, y

# Custom Dataset class for 2D inputs
class Custom2DDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)  # Convert to tensor
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to tensor

    def __len__(self):
        return len(self.data)  # Number of samples

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]  # Label
        return x, y
