"""
CNN autoencoder architecture and dataset wrapper.

This is the BatchNorm-equipped architecture from the tuning notebook
(notebooks/04_cnn_tuning.ipynb), which outperformed the simpler single-block
version from the earlier baseline notebook.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SensorDataset(Dataset):
    """Wraps windowed sensor data as (channels, timesteps) tensors for Conv1d."""

    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class ConvAutoencoder1D(nn.Module):
    """Unsupervised Conv1d autoencoder trained to reconstruct normal windows."""

    def __init__(self, n_features):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, n_features, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
