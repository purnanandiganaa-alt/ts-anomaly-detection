"""Training loop for the CNN autoencoder."""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.cnn.model import ConvAutoencoder1D


def train_autoencoder(
    X_train, n_features, device, epochs, batch_size, lr=1e-3, verbose=True,
    eval_every=None, eval_fn=None, collect_states=False,
):
    """Train a ConvAutoencoder1D on normal (unlabeled) windows.

    Round 2 fix: training loss falls smoothly for the full 50 epochs with no
    plateau, consistent with the autoencoder gradually learning to
    reconstruct some anomalous patterns too (train isn't guaranteed
    anomaly-free), which shrinks the normal/anomalous error gap the score
    depends on. If `eval_fn` is given, it's called every `eval_every` epochs
    with the current model and should return a validation score (whatever
    metric the caller's eval_fn computes - Round 4 uses F-beta, not
    necessarily F1); the epoch with the best score is kept (via a
    state_dict snapshot) instead of blindly returning the final epoch.
    Omit `eval_fn` to keep the old behavior.

    `collect_states=True` additionally returns every candidate state_dict on
    the `eval_every` schedule as (epoch, deep-copied state_dict) pairs, so a
    caller can choose among them *afterwards* using only a subset of the
    validation runs (nested/leave-one-run-out selection). Snapshots are taken
    on the schedule whether or not `eval_fn` is supplied; training itself is
    unaffected either way, since `eval_fn`'s value only drives `best_state`.

    Returns the trained model (best checkpoint if eval_fn is used), or
    (model, states) when collect_states=True.
    """
    from src.models.cnn.model import SensorDataset

    train_loader = DataLoader(SensorDataset(X_train), batch_size=batch_size, shuffle=True)

    model = ConvAutoencoder1D(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_score = -1.0
    best_state = None
    states = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} loss: {total_loss / len(train_loader):.6f}")

        if eval_every and (epoch + 1) % eval_every == 0:
            if collect_states:
                states.append((epoch + 1, copy.deepcopy(model.state_dict())))
            if eval_fn is not None:
                val_score = eval_fn(model)
                if verbose:
                    print(f"  -> validation score at epoch {epoch + 1}: {val_score:.4f}")
                if val_score > best_score:
                    best_score = val_score
                    best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        if verbose:
            print(f"Restoring best checkpoint (validation score={best_score:.4f})")
        model.load_state_dict(best_state)

    return (model, states) if collect_states else model
