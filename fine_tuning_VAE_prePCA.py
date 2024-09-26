import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import pickle
import os

import src.model as lib_model
import src.vaes_fine_tuning as lib_training
import src.data as lib_data
from src import auxiliar

torch.manual_seed(seed=42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Training parameters
batch_size = 64
lr = 2e-4
lr_end = 1e-5
epochs = 1
beta_sweep = [0.1]
latent_dim_sweep = [2]

# Working directories
general_outputs_folder = "./outputs/fine_tuning_VAE_prePCA"
os.makedirs(name=general_outputs_folder, exist_ok=True)

logs_folder = f"./{general_outputs_folder}/01_logs_ft_VAE_prePCA"
os.makedirs(name=logs_folder, exist_ok=True)

checkpoints_folder = f"./{general_outputs_folder}/02_checkpoints_ft_VAE_prePCA"
os.makedirs(name=checkpoints_folder, exist_ok=True)

# Path to old checkpoint files
old_checkpoints_folder = "./outputs/training_VAE_prePCA/02_checkpoints_VAE_prePCA"

# Data
(MA_train, MA_test, Cp_train, Cp_test,
 MA_mean, MA_std, Cp_mean, Cp_std) = lib_data.load_filtered_dataset(path="./00_data/filtered_dataset.pkl")

PCA_train, PCA_test, PCA_mean, PCA_std = lib_data.compute_pca(cp_train=Cp_train, cp_test=Cp_test)

PCA_train_scaled = (PCA_train - PCA_mean) / PCA_std
PCA_test_scaled = (PCA_test - PCA_mean) / PCA_std

n_train = PCA_train_scaled.shape[0]
n_test = PCA_test_scaled.shape[0]

print(f"N train: {n_train:d}, N test: {n_test:d}")

# File containing the latent space coordinates of the models
with open("./00_data/gammas.pkl", "rb") as handle:
    gammas = pickle.load(handle)

for beta in beta_sweep:

    for latent_dim in latent_dim_sweep:
        
        # Get old checkpoint file
        old_checkpoint_file = auxiliar.find_file(path_to_folder=old_checkpoints_folder, beta=beta, 
                                                 latent_dimension=latent_dim, pca_bool=True)

        # Get old model
        old_model = lib_model.VAE_PrePCA(data_size=PCA_train_scaled.shape[1], latent_dim=latent_dim)
        lib_model.load_checkpoint(model=old_model, path_name=old_checkpoint_file, optimizer=None, verbose=False)

        # Get decoder model
        decoder = old_model.decoder.to(device)
        decoder_params = list(decoder.parameters())

        # Get optimizer
        optimizer = torch.optim.Adam(
            params=[{"params": decoder_params,
                     "weight_decay": 0}],
            lr=lr, weight_decay=0)

        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs, div_factor=2,
                                            final_div_factor=lr / lr_end, pct_start=0.2)

        # Get regressor model and its predictions

        gammas_train_ground_truth = gammas[f"beta{beta}_ls{2}_pca{True}"]["gammas_train"]
        gammas_test_ground_truth = gammas[f"beta{beta}_ls{2}_pca{True}"]["gammas_test"]

        print("Getting GPR model...")
        predicted_train, predicted_test = auxiliar.gpr_ma(MA_train, gammas_train_ground_truth, MA_test)
        print("Done")

        # Set up dataset
        np_training_dataset = np.concatenate((predicted_train, PCA_train_scaled), axis=1)
        np_testing_dataset = np.concatenate((predicted_test, PCA_test_scaled), axis=1)

        # Set up dataloader
        dataset_train = torch.utils.data.DataLoader(dataset=torch.from_numpy(np_training_dataset).to(torch.float32).
                                                    to(device), batch_size=batch_size, shuffle=True, num_workers=0)
        
        dataset_test= torch.utils.data.DataLoader(dataset=torch.from_numpy(np_testing_dataset).to(torch.float32).
                                                  to(device), batch_size=batch_size, shuffle=False, num_workers=0)

        # Train loop
        str_date = time.strftime("%Y%m%d_%H_%M")

        model_name = (f"{str_date}_ft_PCAnet0_beta{beta}_dim{latent_dim}_lr{lr}OneCycleLR{lr_end}_bs{batch_size}_"
                      f"epochs{epochs}")
        print(f"Model: {model_name}")

        logger = SummaryWriter(log_dir=f"{logs_folder}/{model_name}")

        for epoch in range(1, epochs + 1):

            decoder.train()
            loss, elapsed = lib_training.train_epoch(model=decoder, data=dataset_train, optimizer=optimizer,
                                                     device=device)

            decoder.eval()
            loss_test, elapsed_test = lib_training.test_epoch(model=decoder, data=dataset_test, device=device)

            scheduler.step()

            lib_training.print_progress(epoch=epoch, epochs=epochs, loss=loss, loss_test=loss_test, elapsed=elapsed,
                                        elapsed_test=elapsed_test)

            logger.add_scalar(tag="General loss/Total", scalar_value=loss, global_step=epoch)
            logger.add_scalar(tag="General loss/Total_test", scalar_value=loss_test, global_step=epoch)
            logger.add_scalar(tag="Optimizer/LR", scalar_value=scheduler.get_last_lr()[0], global_step=epoch)

        state_checkpoint = {"state_dict": decoder.state_dict(), "optimizer_dict": optimizer.state_dict()}
        checkpoint_file = f"{checkpoints_folder}/{model_name}_epoch_final.pth.tar"
        lib_model.save_checkpoint(state=state_checkpoint, path_name=checkpoint_file)
        print(f"Checkpoint. Final epoch, loss: {loss}, test loss: {loss_test}, saving checkpoint {checkpoint_file}")

print("\nEND")
