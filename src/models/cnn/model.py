"""
CNN autoencoder architecture and dataset wrapper.

Round 3: replaced the single shallow Conv1d block (kernel=3, dilation=1,
effective receptive field of only 3 timesteps out of a 200-step window) with
a stacked dilated residual (TCN-style) architecture. The project's own
window-size tuning (notebooks/04_cnn_tuning.ipynb) already established that
anomalies here are slow, sustained drifts rather than sharp spikes - F1 kept
improving as window size grew from 20 to 200 - which means the model needs
to actually see a wide temporal context, not just be handed one. Dilated
convolutions expand the receptive field exponentially by stacking, without
enlarging the window or increasing memory the way a wider window would.
See reports/hybrid_model_postmortem.pdf for the full root-cause writeup.
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


class DilatedResidualBlock(nn.Module):
    """Two dilated Conv1d layers with a residual connection.

    Uses symmetric ("same") padding since this is a reconstruction task, not
    causal forecasting - the model is allowed to use both past and future
    context within the window. A 1x1 conv projects the residual when the
    channel count changes.
    """

    def __init__(self, in_channels, out_channels, dilation, kernel_size=3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ConvAutoencoder1D(nn.Module):
    """Unsupervised dilated-TCN autoencoder trained to reconstruct normal windows.

    Encoder dilation stack (1, 2, 4, 8) gives an effective receptive field of
    ~63 timesteps before the bottleneck pool - roughly 20x the previous
    architecture's 3-timestep receptive field - while keeping the same
    "compress then reconstruct" bottleneck structure as before.
    """

    def __init__(self, n_features):
        super().__init__()

        self.encoder = nn.Sequential(
            DilatedResidualBlock(n_features, 32, dilation=1),
            DilatedResidualBlock(32, 32, dilation=2),
            DilatedResidualBlock(32, 32, dilation=4),
            DilatedResidualBlock(32, 64, dilation=8),
            nn.MaxPool1d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            DilatedResidualBlock(32, 32, dilation=4),
            DilatedResidualBlock(32, 32, dilation=2),
            nn.Conv1d(32, n_features, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
