"""Reconstruction-error scoring for a trained CNN autoencoder."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.cnn.model import SensorDataset


def get_reconstruction_errors(model, X, device, batch_size=64):
    """Per-window mean squared reconstruction error — the CNN anomaly score."""
    loader = DataLoader(SensorDataset(X), batch_size=batch_size)

    model.eval()
    errors = []
    loss_fn = nn.MSELoss(reduction="none")

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)

            loss = loss_fn(out, batch)
            loss = loss.mean(dim=[1, 2]).cpu().numpy()

            errors.extend(loss)

    return np.array(errors)
