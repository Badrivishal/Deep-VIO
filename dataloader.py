import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RandomNoiseDataset(Dataset):
    def __init__(self, num_samples, seq_len):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_image = torch.randn(self.seq_len, 1, 960, 752)
        x_imu = torch.randn(self.seq_len, 1, 6)
        y = torch.randn(self.seq_len, 1, 7)
        return x_image, x_imu, y

if __name__ == "__main__":

    # Create dataset
    dataset = RandomNoiseDataset(num_samples=1000, seq_len=10)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Usage
    for batch in dataloader:
        x_image, x_imu, y = batch
        print(f"x_image shape: {x_image.shape}")
        print(f"x_imu shape: {x_imu.shape}")
        print(f"y shape: {y.shape}")
        break
