"""Reconstruction-error scoring for a trained CNN autoencoder."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.cnn.model import SensorDataset


def get_reconstruction_errors(model, X, device, batch_size=64, per_sensor=False):
    """Reconstruction error - the CNN anomaly score.

    Round 4: added `per_sensor`. Averaging error over every sensor AND
    timestep into one scalar (the original behavior, per_sensor=False)
    lets a naturally noisy sensor's ordinary error swamp a genuinely
    alarming spike on a normally-precise sensor. `per_sensor=True` averages
    only over time, keeping one error value per sensor per window, so a
    Mahalanobis score can later weigh each sensor by its own normal
    variability instead of treating them all as equally scaled.
    """
    loader = DataLoader(SensorDataset(X), batch_size=batch_size)

    model.eval()
    errors = []
    loss_fn = nn.MSELoss(reduction="none")

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)

            loss = loss_fn(out, batch)  # (batch, n_sensors, timesteps)
            dims = [2] if per_sensor else [1, 2]
            loss = loss.mean(dim=dims).cpu().numpy()

            errors.extend(loss)

    return np.array(errors)
