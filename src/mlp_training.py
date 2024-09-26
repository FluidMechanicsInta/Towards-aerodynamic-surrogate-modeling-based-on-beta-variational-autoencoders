
import torch
import torch.nn as nn
import time


def loss_function(reconstruction, data):
    """
    This function computes the MLP loss between the reconstruction of the model and the ground truth.

    Args:
        reconstruction: reconstructed data from the model.
        data: ground truth.

    Returns:
        loss of the model.
    """
    mse_loss = nn.MSELoss(reduction="mean").cuda()

    mse = mse_loss(reconstruction, data)

    return mse


def train_epoch(model, data, optimizer, device):
    """
    This function computes a training epoch of the model.

    Args:
        model: model to be trained.
        data: input data to the model.
        optimizer: optimizer to tune parameters.
        device: whether to use cpu or gpu ("cuda").

    Returns:
        Epoch-averaged loss and the time it took to train.
    """
    start_epoch_time = time.time()

    loss_batch = []  # Store batch loss

    try:

        for batch in data:

            batch = batch.to(device, non_blocking=True)

            predictions = model(batch[:, :2])
            loss = loss_function(predictions, batch[:, 2:])

            loss_batch.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return (sum(loss_batch) / len(loss_batch),
                time.time() - start_epoch_time)

    except TypeError:
        pass


def test_epoch(model, data, device):
    """
    This function tests a model for an epoch.

    Args:
        model: model to be tested.
        data: input data to the model.
        device: whether to use cpu or gpu ("cuda").

    Returns:
        Epoch-averaged loss and the time it took to test.
    """
    start_epoch_time = time.time()

    with torch.no_grad():
        loss_batch = []  # Store batch loss

        try:

            for batch in data:

                batch = batch.to(device, non_blocking=True)

                rec = model(batch[:, :2])
                loss = loss_function(rec, batch[:, 2:])

                loss_batch.append(loss.item())

            return (sum(loss_batch) / len(loss_batch),
                    time.time() - start_epoch_time)

        except TypeError:
            pass


def print_progress(epoch, epochs, loss, loss_test, elapsed, elapsed_test):
    """
    This function is used to print information on the terminal.

    Args:
        epoch: current epoch.
        epochs: total epochs.
        loss: total loss during training.
        loss_test: total loss during testing.
        elapsed: time it took to train.
        elapsed_test: time it took to test.

    Returns:
        None
        """
    print(f"Epoch: {epoch:3d}/{epochs:d}, Loss: {loss:2.4f}, Loss_test: {loss_test:2.4f}, "
          f"time train: {elapsed:2.3f}, time test: {elapsed_test:2.3f}")

    return
