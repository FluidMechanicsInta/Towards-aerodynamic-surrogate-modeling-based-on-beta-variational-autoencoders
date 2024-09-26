import torch
import torch.nn as nn
from torchsummary import summary

"""
Structure of the script: 

Classes:
    1. VAE model
    2. VAE_prePCA model
    3. MLP model
    4. MLP_prePCA model

Functions:
    1. save_checkpoint
    2. load_checkpoint
    
"""

device = "cuda" if torch.cuda.is_available() else "cpu"


class VAE(nn.Module):
    """
    This class is used to generate copies of the VAE model. The beta hyperparameter is specified directly in the loss
    function.

    Args:
        data_size (int): size of input vector. It should be a row.
        latent_dim (int): size of desired latent dimension.

    Returns:
        instance of VAE model.
    """
    def __init__(self, data_size: int, latent_dim: int):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = self.build_encoder(data_size, latent_dim)
        self.decoder = self.build_decoder(data_size, latent_dim)

    @staticmethod
    def build_encoder(data_size, latent_dim):
        """
        This function is used to create the encoding network of the model.

        Args:
            data_size: input size to the encoder.
            latent_dim: dimension to which dimensionality reduction is conducted.

        Returns:
            encoder object of the model.
        """
        encoder = nn.Sequential(
            nn.Linear(data_size, 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim * 2),
        )
        return encoder

    @staticmethod
    def build_decoder(data_size, latent_dim):
        """
        This function is used to create the decoding network of the model.

        Args:
            data_size: output size of the decoder. It is the same as the inputs' of the decoder.
            latent_dim: dimension to which dimensionality reduction is conducted.

        Returns:
            encoder object of the model.
        """
        decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, data_size),

        )
        return decoder

    @staticmethod
    def sample(mean, log_variance):
        """
        This function is used to perform the reparametrization trick.

        Args:
            mean: vector of mean values in the latent space.
            log_variance: vector of log-variances in the latent space.

        Returns:
            z vector.
        """
        std = torch.exp(0.5 * log_variance)
        epsilon = torch.rand_like(std)

        return mean + epsilon * std

    def forward(self, data):
        mean_log_variance = self.encoder(data)
        mean, log_variance = torch.chunk(input=mean_log_variance, chunks=2, dim=-1)
        z = self.sample(mean, log_variance)
        reconstruction = self.decoder(z)

        return reconstruction, mean, log_variance


class VAE_PrePCA(nn.Module):
    """
    This class is used to generate copies of the VAE model when the PCA pre-processing is carried out. The beta
    hyperparameter is specified directly in the loss function. The only difference with the VAE model is the
    architecture of both encoding and decoding nets.

    Args:
        data_size (int): size of input vector. It should be a row.
        latent_dim (int): size of desired latent dimension.

    Returns:
        instance of VAE model.

    """
    def __init__(self, data_size, latent_dim):
        super(VAE_PrePCA, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = self.build_encoder(data_size, latent_dim)
        self.decoder = self.build_decoder(data_size, latent_dim)

    @staticmethod
    def build_encoder(data_size, latent_dim):
        encoder = nn.Sequential(
            nn.Linear(data_size, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Linear(16, latent_dim * 2),
        )
        return encoder

    @staticmethod
    def build_decoder(data_size, latent_dim):
        decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ELU(),
            nn.Linear(16, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, data_size)
        )
        return decoder

    @staticmethod
    def sample(mean, log_variance):
        std = torch.exp(0.5 * log_variance)
        epsilon = torch.rand_like(std)

        return mean + epsilon * std

    def forward(self, data):
        mean_log_variance = self.encoder(data)
        mean, log_variance = torch.chunk(input=mean_log_variance, chunks=2, dim=-1)
        z = self.sample(mean, log_variance)
        reconstruction = self.decoder(z)

        return reconstruction, mean, log_variance


class MLP(nn.Module):
    """
    This class is used to generate copies of the MLP model against which the VAE is compared. Its structure is identical
    to the decoding network of the VAE.

    Args:
        data_size: size of input vector. It should be a row.

    Returns:
        instance of MLP model.

    """
    def __init__(self, data_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, data_size),
        )

    def forward(self, x):
        return self.net(x)


class MLP_PrePCA(nn.Module):
    """
    This class is used to generate copies of the MLP model against which the VAE is compared when the PCA pre-processing
    is carried out. The only difference with the MLP model is the architecture of the network.

    Args:
        data_size: size of input vector. It should be a row.

    Returns:
        instance of MLP model.

    """

    def __init__(self, data_size: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ELU(),
            nn.Linear(16, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, data_size)
        )

    def forward(self, x):
        return self.net(x)


def save_checkpoint(state, path_name, verbose=True):
    """
    This function is used to save the parameters of a trained model.

    Args:
        state: dictionary with the parameters of the model and the optimizer.
        path_name: path to save the state.
        verbose: True if log messages are desired.

    Returns:
        .pth.tar file containing the frozen parameters of both the model and the optimizer.
    """
    if verbose:
        print('Saving checkpoint')

    torch.save(state, path_name)

    if verbose:
        print('Saved checkpoint')

    return


def load_checkpoint(model, path_name, optimizer=None, verbose=True):
    """
    This function is used to load parameters to a model.

    Args:
        model: model to which the parameters are loaded.
        path_name: path to the .pth.tar file (from the save_checkpoint function).
        optimizer: to load parameters to the optimizer as well.
        verbose: True if log messages are desired.

    Returns:
        model loaded with path_name parameters.

    """
    if verbose:
        print('Loading checkpoint')
        print(path_name)

    checkpoint = torch.load(path_name)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_dict'])

    if verbose:
        print('Loaded checkpoint')

    return


if __name__ == "__main__":

    print(f"Selected device: {device}\n")

    example_vae = VAE(data_size=49574, latent_dim=2).to(device)
    print("VAE model")
    summary(example_vae, input_size=(1, 49574))

    print("\nVAE_PrePCA model")
    example_vae_pca = VAE_PrePCA(data_size=371, latent_dim=2).to(device)
    summary(example_vae_pca, input_size=(1, 371))
