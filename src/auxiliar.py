import os
import GPy

from src.model import VAE, VAE_PrePCA, load_checkpoint


def find_file(path_to_folder, beta, latent_dimension, pca_bool):

    if pca_bool:
        pca_bool = "PCAnet"
    else:
        pca_bool = "_net"

    desired_file = "Not found"

    for doc in os.listdir(path_to_folder):

        if (
            (f"_beta{beta}_" in doc)
            and (f"_dim{latent_dimension}_" in doc)
            and (pca_bool in doc)
        ):
            desired_file = doc
            return os.path.join(path_to_folder, desired_file)

    return desired_file


def get_vae_model(data_size, latent_dimension, pca_bool, checkpoint=None, optimizer=None):

    if pca_bool:

        model = VAE_PrePCA(data_size=data_size, latent_dim=latent_dimension)

        if checkpoint is not None:
            load_checkpoint(model=model, path_name=checkpoint, optimizer=optimizer, verbose=True)

    else:

        model = VAE(data_size=data_size, latent_dim=latent_dimension)

        if checkpoint is not None:
            load_checkpoint(model=model, path_name=checkpoint, optimizer=optimizer, verbose=True)

    return model, optimizer


def gpr_ma(ma_train, gammas_train, ma_test):
    """
    This function is used to obtain the outputs from the GPR model.

    Args:
        ma_train: flight conditions in training dataset.
        gammas_train: latent space coordinates.
        ma_test: flight conditions in testing dataset.

    Returns:
        regression predictions for both training and testing datasets.
    """
    kernel = (GPy.kern.Linear(input_dim=2, ARD=True) *
              GPy.kern.Matern32(input_dim=2, variance=1.0, lengthscale=1.0, ARD=True))

    model = GPy.models.GPRegression(X=ma_train, Y=gammas_train, kernel=kernel, normalizer=True)
    model.optimize_restarts(num_restarts=10, verbose=False)

    pred_train, _ = model.predict(ma_train)
    pred_test, _ = model.predict(ma_test)

    return pred_train, pred_test
