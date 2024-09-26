import torch
import torch.nn as nn
import time


def loss_function(reconstruction, data, mean, log_variance, beta):
    """
    This function computes the beta-VAE loss between the reconstruction of the model and the ground truth.

    Args:
        reconstruction: reconstructed data from the model.
        data: ground truth.
        mean: mean of the probability distribution in the latent space.
        log_variance: log-variance of the probability distribution in the latent space.
        beta: beta hyperparameter.

    Returns:
        loss: total loss of the beta-VAE
        MSE: reconstruction term of the total loss.
        KLD: divergence term of the total loss.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mse_loss = nn.MSELoss(reduction="mean").to(device)
    mse = mse_loss(reconstruction, data)

    kld = -0.5 * torch.mean(1 + log_variance - mean.pow(2) - log_variance.exp())

    loss = mse + kld * beta

    return loss, mse, kld


def train_epoch(model, data, optimizer, beta, device):
    """
    This function computes a training epoch of the model.

    Args:
        model: model to be trained.
        data: input data to the model.
        optimizer: optimizer to tune parameters.
        beta: beta hyperparameter.
        device: whether to use cpu or gpu ("cuda").

    Returns:
        Epoch-averaged losses and the time it took to train.
    """
    start_epoch_time = time.time()

    loss_batch = []  # Store batch loss
    mse_batch = []  # Store batch MSE
    kld_batch = []  # Store batch KLD

    for batch in data:

        batch = batch.to(device, non_blocking=True)

        rec, mean, log_variance = model(batch)
        loss, mse, kld = loss_function(rec, batch, mean, log_variance, beta)

        loss_batch.append(loss.item())
        mse_batch.append(mse.item())
        kld_batch.append(kld.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (sum(loss_batch) / len(loss_batch),
            sum(mse_batch) / len(mse_batch),
            sum(kld_batch) / len(kld_batch),
            time.time() - start_epoch_time)


def test_epoch(model, data, beta, device):
    """
    This function tests a model for an epoch.

    Args:
        model: model to be tested.
        data: input data to the model.
        beta: beta hyperparameter.
        device: whether to use cpu or gpu ("cuda").

    Returns:
        Epoch-averaged losses and the time it took to test.
    """
    start_epoch_time = time.time()

    with torch.no_grad():

        loss_batch = []  # Store batch loss
        mse_batch = []  # Store batch MSE
        kld_batch = []  # Store batch KLD

        for batch in data:

            batch = batch.to(device, non_blocking=True)

            rec, mean, log_variance = model(batch)
            loss, mse, kld = loss_function(rec, batch, mean, log_variance, beta)

            loss_batch.append(loss.item())
            mse_batch.append(mse.item())
            kld_batch.append(kld.item())

    return (sum(loss_batch) / len(loss_batch),
            sum(mse_batch) / len(mse_batch),
            sum(kld_batch) / len(kld_batch),
            time.time() - start_epoch_time)


def print_progress(epoch, epochs, loss, loss_test, mse, kld, elapsed, elapsed_test):
    """
    This function is used to print information on the terminal.

    Args:
        epoch: current epoch.
        epochs: total epochs.
        loss: total loss during training.
        loss_test: total loss during testing.
        mse: mse term during training.
        kld: divergence term during training.
        elapsed: time it took to train.
        elapsed_test: time it took to test.

    Returns:
        None
    """
    print(f"Epoch: {epoch:3d}/{epochs:d}, Loss: {loss:2.4f}, Loss_test: {loss_test:2.4f}, "
          f"MSE: {mse:2.4f}, KLD: {kld:2.4f}, time train: {elapsed:2.3f}, time test: {elapsed_test:2.3f}")

    return


class betaScheduler:
    """Schedule beta, linear growth to max value"""

    def __init__(self, start_value, end_value, warmup):
        self.start_value = start_value
        self.end_value = end_value
        self.warmup = warmup

    def get_beta(self, epoch, prints=False):

        if epoch < self.warmup:
            beta = self.start_value + (self.end_value - self.start_value) * epoch / self.warmup
            if prints:
                print(beta)
            return beta
        else:
            return self.end_value
